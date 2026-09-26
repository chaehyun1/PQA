# ** 한번 돌리면 다시 직접 돌릴 필요 없음**
# 해당 파일 직접 실행시 KG 트리플 구성 + 영화 임베딩 사전 계산 → kg_memory.json 저장까지만 수행됨. 나머지는 8_Brain_Module.py에서 호출됨 

# using external knowledge (ex. influence of others and prior beliefs about items)
# Knowledge-Graph Memory Module
# - KG: G = {(h, r, t) | h,t ∈ V, r ∈ E}
# - Entities: user_{uid}, movie_{mid}, genre_{name}
# - Relations: liked / disliked / neutral (user->movie), has_genre (movie->genre)
# - liked: rating >= 4, disliked: rating <= 2, neutral: rating == 3
# - 초기화: 전체 train data + movie genre 트리플로 구성
# - 시뮬레이션 중 새 interaction 발생 시 트리플 추가
#
# [Graph-Aware Dynamic Item Retrieval]
# 쿼리 아이템 x, 유저 u에 대해 후보 아이템들의 score 계산:
#   s_item  = PathSim(x, y) = 2 * |shared_genres(x,y)| / (|genres(x)| + |genres(y)|) (query 아이템 x와 후보 아이템 y의 공유 장르 기반)
#   s_user  = |genres(y) ∩ user_liked_genres| / |genres(y)| (타겟 유저가 liked한 영화들의 장르를 모두 모아서, 후보 아이템 y의 장르와 겹치는 비율)
#   s_path  = α * s_item + (1-α) * s_user     (α=0.8)
#   s_sem   = cosine_similarity(embed(x), embed(y))  (text-embedding-3-small)
#   final   = 0.75 * s_path + 0.25 * s_sem
# top-k₂ = 3 아이템 반환

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from openai import OpenAI

ALPHA = 0.8
NODE_EMB_WEIGHT = 0.25
TOP_K2 = 3
EMBEDDING_MODEL = "text-embedding-3-small"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "MovieLens")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")


class KGMemory:
    def __init__(self, client):
        self.client = client
        self.graph = defaultdict(lambda: defaultdict(set))  # h -> r -> set of t
        self.movie_embeddings = {}  # remapped mid -> np.ndarray

    # ── 트리플 관리 ──────────────────────────────────────────────

    def add_triple(self, h, r, t):
        self.graph[h][r].add(t)

    def add_user_movie(self, uid, mid, score):
        """유저-영화 interaction 트리플 추가."""
        if score >= 4:
            rel = "liked"
        elif score <= 2:
            rel = "disliked"
        else:
            rel = "neutral"
        self.add_triple(f"user_{uid}", rel, f"movie_{mid}")

    def add_movie_genres(self, mid, genres):
        """영화-장르 트리플 추가. genres: list of str"""
        for genre in genres:
            self.add_triple(f"movie_{mid}", "has_genre", f"genre_{genre}")

    # ── 초기화 ───────────────────────────────────────────────────

    def initialize(self, train, all_ratings, rev_user_id_map, rev_movie_id_map, movie_info):
        """전체 train data + movie genre로 KG 초기화."""
        # movie-genre 트리플 (전체 영화)
        for remapped_mid, info in movie_info.items():
            genres = [g.strip() for g in info["genres"].split("|") if g.strip()]
            self.add_movie_genres(remapped_mid, genres)

        # user-movie 트리플 (전체 train)
        for remapped_uid, item_list in train.items():
            orig_uid = rev_user_id_map[remapped_uid]
            for remapped_mid in item_list:
                orig_mid = rev_movie_id_map.get(remapped_mid)
                if orig_mid is None:
                    continue
                score = all_ratings.get(orig_uid, {}).get(orig_mid)
                if score is None:
                    continue
                self.add_user_movie(remapped_uid, remapped_mid, score)

    def precompute_movie_embeddings(self, movie_info):
        """movie_info의 영화 텍스트 임베딩을 배치로 사전 계산. 이미 있는 mid는 스킵."""
        mids = [mid for mid in movie_info.keys() if mid not in self.movie_embeddings]
        if not mids:
            print("  All embeddings already cached, skipping.")
            return
        print(f"  Computing embeddings for {len(mids)} new movies (skipping {len(movie_info) - len(mids)} cached)...")
        texts = [
            f"{movie_info[mid]['title']} ({movie_info[mid]['genres']})"
            for mid in mids
        ]
        # 배치 처리 (OpenAI 최대 2048개 제한)
        batch_size = 2048
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            embeddings.extend([np.array(d.embedding) for d in resp.data])
        for mid, emb in zip(mids, embeddings):
            self.movie_embeddings[mid] = emb

    # ── 유사도 계산 ───────────────────────────────────────────────

    def _genres_of(self, mid):
        return self.graph.get(f"movie_{mid}", {}).get("has_genre", set())

    def _item_item_similarity(self, mid_x, mid_y):
        """PathSim: 장르 공유 기반 item-item similarity."""
        genres_x = self._genres_of(mid_x)
        genres_y = self._genres_of(mid_y)
        denom = len(genres_x) + len(genres_y)
        if denom == 0:
            return 0.0
        return 2 * len(genres_x & genres_y) / denom

    def _user_item_similarity(self, uid, mid_y):
        """유저가 liked한 영화들의 장르 vs 후보 영화 장르 겹치는 비율."""
        liked_movies = self.graph.get(f"user_{uid}", {}).get("liked", set())
        user_genres = set()
        for movie_node in liked_movies:
            user_genres |= self.graph.get(movie_node, {}).get("has_genre", set())
        genres_y = self._genres_of(mid_y)
        if not genres_y:
            return 0.0
        return len(genres_y & user_genres) / len(genres_y)

    def _semantic_similarity(self, mid_x, mid_y):
        """사전 계산된 임베딩 기반 cosine similarity."""
        emb_x = self.movie_embeddings.get(mid_x)
        emb_y = self.movie_embeddings.get(mid_y)
        if emb_x is None or emb_y is None:
            return 0.0
        denom = np.linalg.norm(emb_x) * np.linalg.norm(emb_y)
        if denom == 0:
            return 0.0
        return float(np.dot(emb_x, emb_y) / denom)

    # ── 검색 ─────────────────────────────────────────────────────

    def retrieve(self, uid, query_mid, candidate_mids):
        """
        쿼리 아이템 query_mid와 유저 uid 기준으로 candidate_mids 중 top-k₂ 반환.
        반환: list of {"movie_id": mid, "score": float}
        """
        scores = []
        for mid in candidate_mids:
            s_item = self._item_item_similarity(query_mid, mid)
            s_user = self._user_item_similarity(uid, mid)
            s_path = ALPHA * s_item + (1 - ALPHA) * s_user
            s_sem = self._semantic_similarity(query_mid, mid)
            final = (1 - NODE_EMB_WEIGHT) * s_path + NODE_EMB_WEIGHT * s_sem
            scores.append((mid, round(final, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"movie_id": mid, "score": score} for mid, score in scores[:TOP_K2]]

    # ── 직렬화 ────────────────────────────────────────────────────

    def to_dict(self):
        # graph: defaultdict -> 일반 dict로 변환
        graph_serializable = {
            h: {r: list(t_set) for r, t_set in r_dict.items()}
            for h, r_dict in self.graph.items()
        }
        embeddings_serializable = {
            str(mid): emb.tolist() for mid, emb in self.movie_embeddings.items()
        }
        return {"graph": graph_serializable, "embeddings": embeddings_serializable}

    def from_dict(self, data):
        for h, r_dict in data["graph"].items():
            for r, t_list in r_dict.items():
                self.graph[h][r] = set(t_list)
        self.movie_embeddings = {
            int(mid): np.array(emb) for mid, emb in data["embeddings"].items()
        }


def load_data():
    with open(os.path.join(DATA_DIR, "user_id_map.pkl"), "rb") as f:
        user_id_map = pickle.load(f)
    rev_user_id_map = {v: k for k, v in user_id_map.items()}  # remapped -> original

    with open(os.path.join(DATA_DIR, "movie_id_map.pkl"), "rb") as f:
        movie_id_map = pickle.load(f)
    rev_movie_id_map = {v: k for k, v in movie_id_map.items()}  # remapped -> original

    train = {}
    with open(os.path.join(DATA_DIR, "train.txt")) as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            train[uid] = [int(x) for x in parts[1:]]

    all_ratings = {}
    with open(os.path.join(DATA_DIR, "ratings.dat"), encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            uid, mid, rating = int(parts[0]), int(parts[1]), float(parts[2])
            if uid not in all_ratings:
                all_ratings[uid] = {}
            all_ratings[uid][mid] = rating

    movie_df = pd.read_csv(os.path.join(DATA_DIR, "movie_detail.csv"))
    movie_info = {
        int(row["movie_id"]): {"title": row["title"], "genres": row["genres"]}
        for _, row in movie_df.iterrows()
    }

    return rev_user_id_map, rev_movie_id_map, train, all_ratings, movie_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "kg_memory.json"))
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading data...")
    rev_user_id_map, rev_movie_id_map, train, all_ratings, movie_info = load_data()

    print("Initializing KG...")
    kg = KGMemory(client)
    # 기존 파일이 있으면 임베딩 먼저 로드 (재계산 방지)
    if os.path.exists(args.output):
        print(f"  Found existing {args.output}, loading embeddings...")
        with open(args.output, encoding="utf-8") as f:
            kg.from_dict(json.load(f))
    kg.initialize(train, all_ratings, rev_user_id_map, rev_movie_id_map, movie_info)
    print(f"  Triples loaded. Entities: {len(kg.graph)}")

    print("Precomputing movie embeddings...")
    kg.precompute_movie_embeddings(movie_info) # 일단 title이랑 genre만 넣음 
    print(f"  Embeddings computed for {len(kg.movie_embeddings)} movies.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kg.to_dict(), f, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
