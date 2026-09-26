# ** 한번 돌리면 다시 직접 돌릴 필요 없음**
# 해당 파일 직접 실행시 KG 트리플 구성 + 아이템 임베딩 사전 계산 → kg_memory.json 저장까지만 수행됨. 나머지는 8_Brain_Module.py에서 호출됨
# CDs_and_Vinyl 버전

# using external knowledge (ex. influence of others and prior beliefs about items)
# Knowledge-Graph Memory Module
# - KG: G = {(h, r, t) | h,t ∈ V, r ∈ E}
# - Entities: user_{uid}, item_{iid}, category_{name}
# - Relations: liked / disliked / neutral (user->item), has_category (item->category)
# - liked: rating >= 4, disliked: rating <= 2, neutral: rating == 3
# - 초기화: 전체 train data + item category 트리플로 구성
# - 시뮬레이션 중 새 interaction 발생 시 트리플 추가
#
# [Graph-Aware Dynamic Item Retrieval]
# 쿼리 아이템 x, 유저 u에 대해 후보 아이템들의 score 계산:
#   s_item  = PathSim(x, y) = 2 * |shared_categories(x,y)| / (|cats(x)| + |cats(y)|)
#   s_user  = |cats(y) ∩ user_liked_cats| / |cats(y)|
#   s_path  = α * s_item + (1-α) * s_user     (α=0.8)
#   s_sem   = cosine_similarity(embed(x), embed(y))  (text-embedding-3-small)
#   final   = 0.75 * s_path + 0.25 * s_sem
# top-k₂ = 3 아이템 반환

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from openai import OpenAI

ALPHA = 0.8
NODE_EMB_WEIGHT = 0.25
TOP_K2 = 3
EMBEDDING_MODEL = "text-embedding-3-small"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "CDs_and_Vinyl")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")


class KGMemory:
    def __init__(self, client):
        self.client = client
        self.graph = defaultdict(lambda: defaultdict(set))  # h -> r -> set of t
        self.item_embeddings = {}  # remapped iid(int) -> np.ndarray

    # ── 트리플 관리 ──────────────────────────────────────────────

    def add_triple(self, h, r, t):
        self.graph[h][r].add(t)

    def add_user_item(self, uid, iid, score):
        """유저-아이템 interaction 트리플 추가."""
        if score >= 4:
            rel = "liked"
        elif score <= 2:
            rel = "disliked"
        else:
            rel = "neutral"
        self.add_triple(f"user_{uid}", rel, f"item_{iid}")

    def add_item_categories(self, iid, categories):
        """아이템-카테고리 트리플 추가. categories: list of str"""
        for cat in categories:
            self.add_triple(f"item_{iid}", "has_category", f"category_{cat}")

    # ── 초기화 ───────────────────────────────────────────────────

    def initialize(self, train, review, item_info):
        """전체 train data + item category로 KG 초기화."""
        # item-category 트리플 (전체 아이템)
        for iid, info in item_info.items():
            categories = info.get("categories", [])
            self.add_item_categories(iid, categories)

        # user-item 트리플 (전체 train)
        for uid, item_list in train.items():
            uid_key = str(uid)
            user_reviews = review.get(uid_key, {})
            for iid in item_list:
                iid_key = str(iid)
                r = user_reviews.get(iid_key, {}).get("rating")
                if r is None:
                    continue
                self.add_user_item(uid, iid, float(r))

    def precompute_item_embeddings(self, item_info):
        """item_info의 아이템 텍스트 임베딩을 배치로 사전 계산. 이미 있는 iid는 스킵."""
        iids = [iid for iid in item_info.keys() if iid not in self.item_embeddings]
        if not iids:
            print("  All embeddings already cached, skipping.")
            return
        print(f"  Computing embeddings for {len(iids)} new items (skipping {len(item_info) - len(iids)} cached)...")
        texts = [
            f"{item_info[iid]['title']} ({', '.join(item_info[iid]['categories'])})"
            for iid in iids
        ]
        batch_size = 2048
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            embeddings.extend([np.array(d.embedding) for d in resp.data])
        for iid, emb in zip(iids, embeddings):
            self.item_embeddings[iid] = emb

    # ── 유사도 계산 ───────────────────────────────────────────────

    def _categories_of(self, iid):
        return self.graph.get(f"item_{iid}", {}).get("has_category", set())

    def _item_item_similarity(self, iid_x, iid_y):
        """PathSim: 카테고리 공유 기반 item-item similarity."""
        cats_x = self._categories_of(iid_x)
        cats_y = self._categories_of(iid_y)
        denom = len(cats_x) + len(cats_y)
        if denom == 0:
            return 0.0
        return 2 * len(cats_x & cats_y) / denom

    def _user_item_similarity(self, uid, iid_y):
        """유저가 liked한 아이템들의 카테고리 vs 후보 아이템 카테고리 겹치는 비율."""
        liked_items = self.graph.get(f"user_{uid}", {}).get("liked", set())
        user_cats = set()
        for item_node in liked_items:
            user_cats |= self.graph.get(item_node, {}).get("has_category", set())
        cats_y = self._categories_of(iid_y)
        if not cats_y:
            return 0.0
        return len(cats_y & user_cats) / len(cats_y)

    def _semantic_similarity(self, iid_x, iid_y):
        """사전 계산된 임베딩 기반 cosine similarity."""
        emb_x = self.item_embeddings.get(iid_x)
        emb_y = self.item_embeddings.get(iid_y)
        if emb_x is None or emb_y is None:
            return 0.0
        denom = np.linalg.norm(emb_x) * np.linalg.norm(emb_y)
        if denom == 0:
            return 0.0
        return float(np.dot(emb_x, emb_y) / denom)

    # ── 검색 ─────────────────────────────────────────────────────

    def retrieve(self, uid, query_iid, candidate_iids):
        """
        쿼리 아이템 query_iid와 유저 uid 기준으로 candidate_iids 중 top-k₂ 반환.
        반환: list of {"item_id": iid, "score": float}
        """
        scores = []
        for iid in candidate_iids:
            s_item = self._item_item_similarity(query_iid, iid)
            s_user = self._user_item_similarity(uid, iid)
            s_path = ALPHA * s_item + (1 - ALPHA) * s_user
            s_sem = self._semantic_similarity(query_iid, iid)
            final = (1 - NODE_EMB_WEIGHT) * s_path + NODE_EMB_WEIGHT * s_sem
            scores.append((iid, round(final, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"item_id": iid, "score": score} for iid, score in scores[:TOP_K2]]

    # ── 직렬화 ────────────────────────────────────────────────────

    def to_dict(self):
        graph_serializable = {
            h: {r: list(t_set) for r, t_set in r_dict.items()}
            for h, r_dict in self.graph.items()
        }
        embeddings_serializable = {
            str(iid): emb.tolist() for iid, emb in self.item_embeddings.items()
        }
        return {"graph": graph_serializable, "embeddings": embeddings_serializable}

    def from_dict(self, data):
        for h, r_dict in data["graph"].items():
            for r, t_list in r_dict.items():
                self.graph[h][r] = set(t_list)
        self.item_embeddings = {
            int(iid): np.array(emb) for iid, emb in data["embeddings"].items()
        }


def load_data():
    # train.json: {"History": {uid: [iids]}, "Time": {...}}
    with open(os.path.join(DATA_DIR, "train.json")) as f:
        train_raw = json.load(f)
    train_history = train_raw.get("History", {})
    train = {int(uid): [int(x) for x in items] for uid, items in train_history.items()}

    # review.json: uid(str) -> {iid(str) -> {rating, title, text}}
    with open(os.path.join(DATA_DIR, "review.json")) as f:
        review = json.load(f)

    # meta.json: iid(str) -> {title, categories, store, price, average_rating}
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    item_info = {
        int(iid): {
            "title": info.get("title", f"Unknown (id={iid})"),
            "categories": info.get("categories", []),
        }
        for iid, info in meta.items()
    }

    return train, review, item_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "kg_memory.json"))
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading data...")
    train, review, item_info = load_data()

    print("Initializing KG...")
    kg = KGMemory(client)
    # 기존 파일이 있으면 임베딩 먼저 로드 (재계산 방지)
    if os.path.exists(args.output):
        print(f"  Found existing {args.output}, loading embeddings...")
        with open(args.output, encoding="utf-8") as f:
            kg.from_dict(json.load(f))
    kg.initialize(train, review, item_info)
    print(f"  Triples loaded. Entities: {len(kg.graph)}")

    print("Precomputing item embeddings...")
    kg.precompute_item_embeddings(item_info)
    print(f"  Embeddings computed for {len(kg.item_embeddings)} items.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kg.to_dict(), f, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
