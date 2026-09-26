# 논문 Self-Consistent Persona Evaluation 부분 참고 (단, 식이 좀 이상함?)
# 생성된 페르소나 후보 중 하나 선택하기
#
# s(p, u) = Σ r̂(ι, p) - Σ r̂(ī, p)
#
# [로직]
# 각 candidate persona p에 대해 score s(p, u)를 계산한다.
# 1. target user의 history에서 J번, 매번 ρ개씩 랜덤 샘플링하여 LLM에게 얼마나 persona와 맞는지 평가 → 점수 합산
# 2. 나를 제외한 유저 중 N명을 랜덤 선택, 각 유저마다 J번 × ρ개 샘플링하여 평가 후 sum → N개의 sum을 평균하여 other_score 산출
# 3. s(p, u) = target 점수 합 - other 점수 합
# 4. 5개의 candidate 중 s가 가장 높은 persona를 해당 유저의 최종 persona로 선택

import os
import json
import random
import argparse
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

MAX_WORKERS = 5  # candidate 5개를 병렬로 처리


# NOTE: J x N_OTHER x num_users를 어떻게 설정하는지에 따라서 API 호출 수 크게 달라짐 
J = 2       # 샘플링 반복 수
RHO = 10     # 한 subset당 아이템 수
N_OTHER = 5 # other users에서 랜덤하게 선택할 유저 수

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "MovieLens")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")


def load_data():
    movie_df = pd.read_csv(os.path.join(DATA_DIR, "movie_detail.csv"))
    movie_info = {int(row["movie_id"]): {"title": row["title"], "genres": row["genres"], "avg_rating": row["rating"], "summary": row["summary"]}
                  for _, row in movie_df.iterrows()}

    train = {}
    with open(os.path.join(DATA_DIR, "train.txt")) as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            train[uid] = [int(x) for x in parts[1:]]

    return movie_info, train


def format_interactions(items, movie_info):
    """remapped item ID 리스트를 텍스트로 변환."""
    lines = []
    for remapped_mid in items:
        info = movie_info.get(remapped_mid, {})
        title = info.get("title", f"Unknown (id={remapped_mid})")
        genres = info.get("genres", "Unknown")
        avg_rating = info.get("avg_rating")
        summary = info.get("summary", "")
        avg_rating_str = str(round(avg_rating, 2)) if avg_rating is not None else "N/A"
        lines.append(f"  - {title} ({genres}), avg rating: {avg_rating_str}, summary: {summary}")
    return "\n".join(lines)


def build_rating_prompt(persona, interaction_text):
    return f"""You are evaluating how well a movie interaction history aligns with a given persona.

## Persona
{persona}

## Movie Interaction History
{interaction_text}

## Instructions
Rate how well this interaction history aligns with the persona on a scale from 1 to 10.
- 1 means the interactions are completely inconsistent with the persona.
- 10 means the interactions perfectly match the persona.

Respond in the following format:
Score: <integer from 1 to 10>
Reason: <brief explanation>"""


def get_llm_score(client, prompt, model):
    resp = client.responses.create(model=model, input=prompt)
    text = resp.output_text.strip()

    print(f"[LLM Raw Response]\n{text}")
    match = re.search(r"Score:\s*(\d+)", text)
    if match:
        score = min(10, max(1, int(match.group(1))))
        return score
    
    # fallback: 첫 번째 숫자
    print("[Parsed] Score regex failed, trying fallback...")
    match = re.search(r"\d+", text)
    if match:
        score = min(10, max(1, int(match.group())))
        print(f"[Parsed] Fallback matched: {match.group()} -> clamped: {score}")
        return score
    
    print("[Parsed] All parsing failed, returning default 5")
    return 5  # 파싱 실패 시 중간값


def score_persona(client, model, persona, uid,
                  train, movie_info):
    """s(p, u) 계산."""
    user_items = train.get(uid, [])

    # target user: J개 subset 샘플링
    target_score = 0
    for _ in range(J):
        subset = random.sample(user_items, min(RHO, len(user_items)))
        text = format_interactions(subset, movie_info)
        prompt = build_rating_prompt(persona, text)
        target_score += get_llm_score(client, prompt, model)

    # n명의 other user 랜덤 선택, 각 유저마다 j번 샘플링 후 sum → n개의 sum을 평균
    other_uids = random.sample([u for u in train.keys() if u != uid], N_OTHER)
    per_user_sums = []
    for other_uid in other_uids:
        other_items = train.get(other_uid, [])
        user_sum = 0
        for _ in range(J):
            subset = random.sample(other_items, min(RHO, len(other_items)))
            text = format_interactions(subset, movie_info)
            prompt = build_rating_prompt(persona, text)
            user_sum += get_llm_score(client, prompt, model)
        per_user_sums.append(user_sum)
    other_score = sum(per_user_sums) / len(per_user_sums)

    return round(target_score - other_score, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_users", type=str,
                        default=os.path.join(DATA_DIR, "user_sets.txt"),
                        help="Path to user_sets.txt")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--input", type=str, default=os.path.join(RESULT_DIR, "persona_candidates.json"))
    parser.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "selected_persona.json"))
    parser.add_argument("--num_users", type=int, default=None, help="처리할 유저 수 (None이면 전체)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading data...")
    movie_info, train = load_data()

    with open(args.input, encoding="utf-8") as f:
        candidates_data = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = {}
    with open(args.target_users, 'r') as f:
        target = set(int(line.strip()) for line in f if line.strip())
    user_ids = [int(k) for k in candidates_data.keys() if int(k) in target]
    user_ids = user_ids[:args.num_users]

    def evaluate_user(uid):
        candidates = candidates_data[str(uid)]["candidates"]

        # fast-path: candidate가 1개뿐이면 LLM 평가 생략하고 그대로 선택
        if len(candidates) <= 1:
            scores = [None] * len(candidates)
            return uid, candidates, scores, 0

        def eval_candidate(args_tuple):
            j_idx, persona = args_tuple
            s = score_persona(client, args.model, persona, uid, train, movie_info)
            return j_idx, s

        scores = [None] * len(candidates)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(eval_candidate, (j_idx, persona)): j_idx
                       for j_idx, persona in enumerate(candidates)}
            for future in as_completed(futures):
                j_idx, s = future.result()
                scores[j_idx] = s
                print(f"  user {uid} | candidate {j_idx+1}/{len(candidates)} | score={s}")

        best_idx = scores.index(max(scores))
        return uid, candidates, scores, best_idx

    print(f"Evaluating personas for {len(user_ids)} users...")
    for i, uid in enumerate(user_ids):
        uid, candidates, scores, best_idx = evaluate_user(uid)
        results[uid] = {
            "selected_persona": candidates[best_idx],
            "scores": scores,
            "best_idx": best_idx,
        }
        score_str = scores[best_idx] if scores[best_idx] is not None else "skipped (single candidate)"
        print(f"  [{i+1}/{len(user_ids)}] user {uid} -> best candidate {best_idx+1} (score={score_str})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
