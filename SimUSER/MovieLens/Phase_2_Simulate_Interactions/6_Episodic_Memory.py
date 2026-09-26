# 해당 파일 직접 실행 시 train history 기반으로 각 유저의 초기 episodic memory 구성 후, result/episodic_memory_init.json에 저장

# Episodic Memory Module
# - 초기 train history로 메모리 초기화 (liked/disliked/neutral plain text)
# - 새로운 RS interaction 발생 시 메모리에 추가
# - Self-ask 검색 전략: LLM이 follow-up 질문 3개 생성 → 원래 쿼리 + follow-up 각각 임베딩 → cosine similarity → top-5 반환
#
# [저장 포맷]
# 초기 history:
#   - score >= 4: "I liked {item_name} based on my review score of {score}."
#   - score <= 2: "I disliked {item_name} based on my review score of {score}."
#   - score == 3: "I felt neutral about {item_name} based on my review score of {score}."
# RS interaction:
#   "The recommender system recommended the following {item_type} to me on page {page_number}:
#    {name_all_items}, among them, I selected {watched_items} and rate them {ratings} respectively.
#    I dislike the rest {item_type} items: {dislike_items}."

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from openai import OpenAI

NUM_FOLLOW_UP = 4 # 참고: 프롬프트 때문인지 계속 page 내에 있는 아이템 각각에 대한 것과 관련된 것으로 하나씩 질문을 해서, page 내 아이템 수랑 맞추는게 좋을듯
TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-small"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "MovieLens")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")


class EpisodicMemory:
    def __init__(self, client):
        self.client = client
        self.memories = []  # list of {"text": str, "embedding": np.ndarray} / 유저의 train history에 있는 과거 interaction 

    def _get_embedding(self, text):
        # 단일 텍스트 임베딩 (add() 등 단건 호출용)
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return np.array(resp.data[0].embedding)

    def _get_embeddings_batch(self, texts):
        # 여러 텍스트를 한 번의 API 호출로 임베딩 (비용 절감)
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts) # 1536차원
        return [np.array(d.embedding) for d in resp.data]

    def _cosine_similarity(self, a, b):
        # 두 임베딩 벡터 간 cosine 유사도 계산
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _format_history_entry(self, movie_name, score):
        # score 기준으로 liked / disliked / neutral 텍스트 포맷 생성
        if score >= 4:
            return f"I liked {movie_name} based on my review score of {score}."
        elif score <= 2:
            return f"I disliked {movie_name} based on my review score of {score}."
        else:
            return f"I felt neutral about {movie_name} based on my review score of {score}."

    def add(self, text):
        # 단일 텍스트를 임베딩하여 메모리에 추가
        embedding = self._get_embedding(text)
        self.memories.append({"text": text, "embedding": embedding})

    def initialize_from_history(self, history):
        """train history 기반으로 초기 메모리 구성. 배치 임베딩으로 API 1회 호출."""
        if not history:
            return
        # 전체 history 텍스트 생성 후 한 번에 임베딩
        texts = [self._format_history_entry(movie_name, score) for movie_name, score in history]
        embeddings = self._get_embeddings_batch(texts)
        for text, embedding in zip(texts, embeddings):
            self.memories.append({"text": text, "embedding": embedding})

    def add_rs_interaction(self, page_number, item_type, name_all_items, watched_items, ratings_list, dislike_items):
        """시뮬레이션 중 RS interaction 발생 시 메모리에 추가."""
        # 추천 페이지 전체 / 시청 / 평점 / 비시청 항목을 하나의 텍스트로 조합
        all_items_str = ", ".join(name_all_items)
        watched_str = ", ".join(watched_items)
        ratings_str = ", ".join(str(r) for r in ratings_list)
        dislike_str = ", ".join(dislike_items)
        text = (
            f"The recommender system recommended the following {item_type} to me on page {page_number}: "
            f"{all_items_str}, among them, I selected {watched_str} and rate them {ratings_str} respectively. "
            f"I dislike the rest {item_type} items: {dislike_str}."
        )
        self.add(text)

    def _generate_follow_ups(self, query, llm_model):
        # LLM으로 원래 쿼리에 대한 follow-up 질문 NUM_FOLLOW_UP개 생성 (self-ask 전략)
        prompt = (
            f"A user is deciding whether to watch movies on a recommendation page.\n\n"
            f"Recommended movies: {query}\n\n"
            f"Generate {NUM_FOLLOW_UP} questions to retrieve the user's past experiences "
            f"relevant to these movies. Questions should be concrete and targeted, such as:\n"
            f"- Have I enjoyed [ genre] movies in the past?\n"
            f"- How did I feel when I watched movies with [theme/mood]?\n"
            f"- Do I tend to like movies from [specific era/style]?\n\n"
            f"Generate exactly {NUM_FOLLOW_UP} questions, one per line, without numbering or bullet points."
        )
        try:
            resp = self.client.responses.create(model=llm_model, input=prompt)
            lines = [l.strip() for l in resp.output_text.strip().split("\n") if l.strip()]
            return lines[:NUM_FOLLOW_UP]
        except Exception:
            print("Not create any follow-up questions")
            return []

    def retrieve(self, query, llm_model, top_k=TOP_K, follow_ups=None):
        """Self-ask 전략으로 top-k 메모리 반환.
        follow_ups를 전달하면 재생성 없이 기존 follow-up을 사용."""
        if not self.memories:
            return [], []

        # follow-up이 없으면 새로 생성
        if follow_ups is None:
            follow_ups = self._generate_follow_ups(query, llm_model)
        all_queries = [query] + follow_ups

        # 모든 쿼리를 한 번에 임베딩
        query_embeddings = self._get_embeddings_batch(all_queries)

        # 각 메모리에 대해 모든 쿼리 임베딩 중 max cosine similarity를 점수로 사용
        scores = []
        for mem in self.memories:
            sim = max(self._cosine_similarity(qe, mem["embedding"]) for qe in query_embeddings)
            scores.append(sim)

        # 상위 top_k 메모리 반환
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [
            {"text": self.memories[i]["text"], "score": round(scores[i], 4)}
            for i in top_indices
        ] # 어떤 아이템을 좋아하고 싫어하는지 유사도 기반으로 5개 선정함 (아이템 제목만 알려줌)
        return results, follow_ups

    def to_dict(self):
        # 직렬화: embedding을 list로 변환하여 JSON 저장 가능하게
        return [{"text": m["text"], "embedding": m["embedding"].tolist()} for m in self.memories]

    def from_dict(self, data):
        # 역직렬화: JSON에서 로드 후 embedding을 np.array로 복원
        self.memories = [{"text": d["text"], "embedding": np.array(d["embedding"])} for d in data]


def load_data():
    with open(os.path.join(DATA_DIR, "user_id_map.pkl"), "rb") as f:
        user_id_map = pickle.load(f)
    rev_user_id_map = {v: k for k, v in user_id_map.items()}  # remapped -> original

    with open(os.path.join(DATA_DIR, "movie_id_map.pkl"), "rb") as f:
        movie_id_map = pickle.load(f)
    rev_movie_id_map = {v: k for k, v in movie_id_map.items()}  # remapped -> original

    # train.txt: remapped uid -> list of remapped movie IDs
    train = {}
    with open(os.path.join(DATA_DIR, "train.txt")) as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            train[uid] = [int(x) for x in parts[1:]]

    # ratings.dat: original uid -> {original mid -> rating}
    all_ratings = {}
    with open(os.path.join(DATA_DIR, "ratings.dat"), encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            uid, mid, rating = int(parts[0]), int(parts[1]), float(parts[2])
            if uid not in all_ratings:
                all_ratings[uid] = {}
            all_ratings[uid][mid] = rating

    # movie_detail.csv: remapped movie ID -> title
    movie_df = pd.read_csv(os.path.join(DATA_DIR, "movie_detail.csv"))
    movie_info = {int(row["movie_id"]): row["title"] for _, row in movie_df.iterrows()}

    return rev_user_id_map, rev_movie_id_map, train, all_ratings, movie_info


def build_history(uid, rev_user_id_map, rev_movie_id_map, train, all_ratings, movie_info):
    """train data 기반으로 (movie_name, score) 리스트 반환."""
    orig_uid = rev_user_id_map[uid]
    history = []
    for remapped_mid in train.get(uid, []):
        orig_mid = rev_movie_id_map.get(remapped_mid)
        if orig_mid is None:
            continue
        score = all_ratings.get(orig_uid, {}).get(orig_mid)
        if score is None:
            continue
        title = movie_info.get(remapped_mid, f"Unknown (id={remapped_mid})")
        history.append((title, int(score)))
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_users", type=str,
                        default=os.path.join(DATA_DIR, "user_sets.txt"),
                        help="Path to user_sets.txt")
    parser.add_argument("--input", type=str, default=os.path.join(RESULT_DIR, "selected_persona.json"))
    parser.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "episodic_memory_init.json"))
    parser.add_argument("--num_users", type=int, default=None, help="처리할 유저 수 (None이면 전체)")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading data...")
    rev_user_id_map, rev_movie_id_map, train, all_ratings, movie_info = load_data()

    with open(args.input, encoding="utf-8") as f:
        selected_data = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.target_users, 'r') as f:
        target = set(int(line.strip()) for line in f if line.strip())
    user_ids = [int(k) for k in selected_data.keys() if int(k) in target]
    user_ids = user_ids[:args.num_users]

    # 기존 파일이 있으면 로드 (이미 처리된 유저는 스킵)
    results = {}
    if os.path.exists(args.output):
        print(f"  Found existing {args.output}, loading cached entries...")
        with open(args.output, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                results[int(k)] = v

    missing_uids = [uid for uid in user_ids if uid not in results]
    cached_count = len(user_ids) - len(missing_uids)
    print(f"  {cached_count} cached / {len(missing_uids)} to process")

    if not missing_uids:
        print("All users already cached.")
        return

    print(f"Initializing episodic memory for {len(missing_uids)} users...")
    for i, uid in enumerate(missing_uids):
        history = build_history(uid, rev_user_id_map, rev_movie_id_map, train, all_ratings, movie_info)
        memory = EpisodicMemory(client)
        memory.initialize_from_history(history)
        results[uid] = memory.to_dict()
        print(f"  [{i+1}/{len(missing_uids)}] user {uid}: {len(memory.memories)} memory entries")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
