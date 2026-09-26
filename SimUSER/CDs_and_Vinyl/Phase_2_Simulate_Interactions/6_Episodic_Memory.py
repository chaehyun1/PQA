# 해당 파일 직접 실행 시 train history 기반으로 각 유저의 초기 episodic memory 구성 후, result/episodic_memory_init.json에 저장 (CDs_and_Vinyl 버전)

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
#    {name_all_items}, among them, I selected {listened_items} and rate them {ratings} respectively.
#    I dislike the rest {item_type} items: {dislike_items}."

import os
import json
import argparse
import numpy as np
import pandas as pd
from openai import OpenAI

NUM_FOLLOW_UP = 4 # 참고: 프롬프트 때문인지 계속 page 내에 있는 아이템 각각에 대한 것과 관련된 것으로 하나씩 질문을 해서, page 내 아이템 수랑 맞추는게 좋을듯
TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-small"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "CDs_and_Vinyl")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")


class EpisodicMemory:
    def __init__(self, client):
        self.client = client
        self.memories = []  # list of {"text": str, "embedding": np.ndarray} / 유저의 train history에 있는 과거 interaction

    def _get_embedding(self, text):
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return np.array(resp.data[0].embedding)

    def _get_embeddings_batch(self, texts):
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts) # 1536차원
        return [np.array(d.embedding) for d in resp.data]

    def _cosine_similarity(self, a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _format_history_entry(self, item_name, score):
        if score >= 4:
            return f"I liked {item_name} based on my review score of {score}."
        elif score <= 2:
            return f"I disliked {item_name} based on my review score of {score}."
        else:
            return f"I felt neutral about {item_name} based on my review score of {score}."

    def add(self, text):
        embedding = self._get_embedding(text)
        self.memories.append({"text": text, "embedding": embedding})

    def initialize_from_history(self, history):
        """train history 기반으로 초기 메모리 구성. 배치 임베딩으로 API 1회 호출."""
        if not history:
            return
        texts = [self._format_history_entry(item_name, score) for item_name, score in history]
        embeddings = self._get_embeddings_batch(texts)
        for text, embedding in zip(texts, embeddings):
            self.memories.append({"text": text, "embedding": embedding})

    def add_rs_interaction(self, page_number, item_type, name_all_items, listened_items, ratings_list, dislike_items):
        """시뮬레이션 중 RS interaction 발생 시 메모리에 추가."""
        all_items_str = ", ".join(name_all_items)
        listened_str = ", ".join(listened_items)
        ratings_str = ", ".join(str(r) for r in ratings_list)
        dislike_str = ", ".join(dislike_items)
        text = (
            f"The recommender system recommended the following {item_type} to me on page {page_number}: "
            f"{all_items_str}, among them, I selected {listened_str} and rate them {ratings_str} respectively. "
            f"I dislike the rest {item_type} items: {dislike_str}."
        )
        self.add(text)

    def _generate_follow_ups(self, query, llm_model):
        # LLM으로 원래 쿼리에 대한 follow-up 질문 NUM_FOLLOW_UP개 생성 (self-ask 전략)
        prompt = (
            f"A user is deciding whether to listen to albums on a recommendation page.\n\n"
            f"Recommended albums: {query}\n\n"
            f"Generate {NUM_FOLLOW_UP} questions to retrieve the user's past experiences "
            f"relevant to these albums. Questions should be concrete and targeted, such as:\n"
            f"- Have I enjoyed [genre] albums in the past?\n"
            f"- How did I feel when I listened to albums with [theme/mood]?\n"
            f"- Do I tend to like albums from [specific era/style]?\n\n"
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

        if follow_ups is None:
            follow_ups = self._generate_follow_ups(query, llm_model)
        all_queries = [query] + follow_ups

        query_embeddings = self._get_embeddings_batch(all_queries)

        scores = []
        for mem in self.memories:
            sim = max(self._cosine_similarity(qe, mem["embedding"]) for qe in query_embeddings)
            scores.append(sim)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [
            {"text": self.memories[i]["text"], "score": round(scores[i], 4)}
            for i in top_indices
        ]
        return results, follow_ups

    def to_dict(self):
        return [{"text": m["text"], "embedding": m["embedding"].tolist()} for m in self.memories]

    def from_dict(self, data):
        self.memories = [{"text": d["text"], "embedding": np.array(d["embedding"])} for d in data]


def load_data():
    # train.json: {"History": {uid: [iids]}, "Time": {...}}
    with open(os.path.join(DATA_DIR, "train.json")) as f:
        train_raw = json.load(f)
    train_history = train_raw.get("History", {})
    train = {int(uid): [int(x) for x in items] for uid, items in train_history.items()}

    # review.json: uid(str) -> {iid(str) -> {rating, title, text}}
    with open(os.path.join(DATA_DIR, "review.json")) as f:
        review = json.load(f)

    # meta.json: iid(str) -> {title, categories, ...}
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    item_info = {iid: info.get("title", f"Unknown (id={iid})") for iid, info in meta.items()}

    return train, review, item_info


def build_history(uid, train, review, item_info):
    """train data 기반으로 (item_name, score) 리스트 반환."""
    uid_key = str(uid)
    user_reviews = review.get(uid_key, {})
    history = []
    for iid in train.get(uid, []):
        iid_key = str(iid)
        r = user_reviews.get(iid_key, {}).get("rating")
        if r is None:
            continue
        title = item_info.get(iid_key, f"Unknown (id={iid_key})")
        history.append((title, int(float(r))))
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_users", type=str, default=os.path.join(DATA_DIR, "user_sets.txt"), help="Path to user_sets.txt")
    parser.add_argument("--input", type=str, default=os.path.join(RESULT_DIR, "selected_persona.json"))
    parser.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "episodic_memory_init.json"))
    parser.add_argument("--num_users", type=int, default=None, help="처리할 유저 수")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading data...")
    train, review, item_info = load_data()

    with open(args.input, encoding="utf-8") as f:
        selected_data = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.target_users, 'r') as f:
        target = set(int(line.strip()) for line in f if line.strip())
    user_ids = [int(k) for k in selected_data.keys() if int(k) in target]
    user_ids = user_ids[:args.num_users]

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
        history = build_history(uid, train, review, item_info)
        memory = EpisodicMemory(client)
        memory.initialize_from_history(history)
        results[uid] = memory.to_dict()
        print(f"  [{i+1}/{len(missing_uids)}] user {uid}: {len(memory.memories)} memory entries")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
