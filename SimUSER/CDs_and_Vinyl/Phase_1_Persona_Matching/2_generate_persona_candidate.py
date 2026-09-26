# short summary를 historical data와 함께 프롬프트에 넣어서 LLM이 페르소나 후보를 생성하도록 함
# 가능한 personality 제공 (CDs_and_Vinyl 데이터셋에는 demographic 정보 없음)
# 5개의 candidate persona 생성

# personality의 경우 Agent4Rec에서 쓴 세 가지 그대로 이용함 (CDs/Vinyl 도메인 맞게 수정)
# short summary의 경우 마찬가지로 데이터셋에 있는 taste 및 high rating을 사용함

import os
import argparse
import json
import pandas as pd
from openai import OpenAI

NUM_HISTORY = 15 # interaction history에서 최신 기준 몇개까지 볼 지

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "CDs_and_Vinyl")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")


activity_dict = {
    0: "An Incredibly Elusive Occasional Listener, so seldom attracted by CD and vinyl recommendations that it's a legendary event when you actually listen to a record. Your listening habits are extraordinarily infrequent, and you will exit the recommender system immediately if you feel even slightly unsatisfied.",
    1: "An Occasional Listener, seldom attracted by CD and vinyl recommendations. You are only curious about albums that strictly align with your taste. Your listening habits are not very frequent, and you tend to exit the recommender system if you have a few unsatisfying listening experiences.",
    2: "A Music Enthusiast with an insatiable appetite for CDs and vinyl, willing to listen to nearly every album recommended to you. Music is a central part of your life, and album recommendations are integral to your experience. You are tolerant of the recommender system and are not likely to leave even with some unsatisfying recommendations."
}
conformity_dict = {
    0: "A Dedicated Follower who heavily relies on an album's historical ratings and popular reviews, rarely expressing independent opinions. You usually give ratings that are the same as the general consensus.",
    1: "A Balanced Evaluator who considers both popular reviews and personal preferences when rating albums. You sometimes give ratings that are different from the historical average.",
    2: "A Maverick Critic who completely ignores popular opinion and evaluates albums solely based on your own taste. You usually give ratings that are significantly different from an album's historical ratings."
}
diversity_dict = {
    0: "An Exceedingly Discerning Selective Listener who chooses CDs and vinyl with a level of selectivity that borders on exclusivity. Your album choices are meticulously curated to match your personal taste, leaving almost no room for genre diversity.",
    1: "A Niche Explorer who occasionally explores different music genres but mostly sticks to your preferred styles and artists.",
    2: "A Sonic Trailblazer, a relentless seeker of the unique and the obscure in the world of music. Your CD and vinyl collection is so diverse and avant-garde that it defies easy categorization."
}


def load_data():
    # personality.json: {activity: {uid: level}, diversity: {...}, conformity: {...}}
    with open(os.path.join(DATA_DIR, "personality.json")) as f:
        personality = json.load(f)
    user_stat = {}
    for uid_str in personality["activity"].keys():
        uid = int(uid_str)
        user_stat[uid] = {
            "activity": activity_dict[int(personality["activity"][uid_str])],
            "diversity": diversity_dict[int(personality["diversity"][uid_str])],
            "conformity": conformity_dict[int(personality["conformity"][uid_str])],
        }

    # all_persona.csv: user_id, taste, reasons, high_rating, low_rating
    persona_df = pd.read_csv(os.path.join(DATA_DIR, "all_persona.csv"))
    persona_map = {int(row["user_id"]): row for _, row in persona_df.iterrows()}

    # meta.json: item_id (remapped, str) -> {title, categories, store, price, average_rating}
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    item_info = {}
    for iid, info in meta.items():
        item_info[iid] = {
            "title": info.get("title", f"Unknown (id={iid})"),
            "categories": ", ".join(info.get("categories", [])) or "Unknown",
            "store": info.get("store", ""),
            "price": info.get("price", ""),
            "avg_rating": info.get("average_rating"),
        }

    # review.json: user_id (str) -> {item_id (str) -> {rating, title, text}}
    with open(os.path.join(DATA_DIR, "review.json")) as f:
        review = json.load(f)

    # train.json: {"History": {uid: [iids]}, "Time": {uid: [timestamps]}}
    with open(os.path.join(DATA_DIR, "train.json")) as f:
        train_raw = json.load(f)
    train_history = train_raw.get("History", {})
    train_time = train_raw.get("Time", {})

    return user_stat, persona_map, item_info, review, train_history, train_time


def get_recent_interactions(uid, item_info, review, train_history, train_time, n=NUM_HISTORY):
    """최신순으로 최대 n개의 interaction history 반환."""
    uid_key = str(uid)
    items = train_history.get(uid_key, [])
    times = train_time.get(uid_key, [])
    user_reviews = review.get(uid_key, {})

    records = []
    for idx, iid in enumerate(items):
        iid_key = str(iid)
        ts = int(times[idx]) if idx < len(times) else 0
        rating = user_reviews.get(iid_key, {}).get("rating")
        info = item_info.get(iid_key, {})
        title = info.get("title", f"Unknown (id={iid_key})")
        categories = info.get("categories", "Unknown")
        store = info.get("store", "")
        price = info.get("price", "")
        avg_rating = info.get("avg_rating")
        records.append((ts, title, categories, store, price, rating, avg_rating))
    records.sort(key=lambda x: x[0], reverse=True)
    return records[:n]


def build_prompt(stat, taste, high_rating, history):
    return f"""You are given information about a music (CDs/Vinyl) user. Based on this information, generate a persona description that summarizes who this user is.

## User Information
- Activity (The activity characteristic pertains to the frequency of your music listening habits): {stat['activity']}
- Diversity (The diversity characteristic gauges your likelihood of listening to albums that may not align with your usual taste): {stat['diversity']}
- Conformity (The conformity characteristic measures the degree to which your ratings are influenced by historical ratings): {stat['conformity']}
- Music taste: {taste}
- High rating pattern: {high_rating}

## Interaction History (ordered from oldest to most recent)
{history}

## Instructions
Generate a concise persona description (2 sentences) that integrates all the above information into a coherent user profile. Output only the persona description, nothing else.
"""


def generate_candidates(client, prompt, model, n=1): # 후보 5개 생성
    candidates = []
    for _ in range(n):
        resp = client.responses.create(model=model, input=prompt)
        candidates.append(resp.output_text.strip())
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_users", type=str, default=os.path.join(DATA_DIR, "user_sets.txt"), help="Path to user_sets.txt")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--num_users", type=int, default=None, help="처리할 유저 수")
    parser.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "persona_candidates.json"))
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading data...")
    user_stat, persona_map, item_info, review, train_history, train_time = load_data()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = {}
    with open(args.target_users, 'r') as f:
        user_ids = [int(line.strip()) for line in f if line.strip()]
    user_ids = user_ids[:args.num_users]

    print(f"Generating persona candidates for {len(user_ids)} users...")
    for i, uid in enumerate(user_ids):
        if uid not in user_stat or uid not in persona_map:
            print(f"  [{i+1}/{len(user_ids)}] user {uid} skipped (missing stat or persona)")
            continue
        stat = user_stat[uid]
        row = persona_map[uid]
        taste = str(row["taste"]).strip()
        high_rating = str(row["high_rating"]).strip()

        history_records = get_recent_interactions(uid, item_info, review, train_history, train_time)
        history_str = "\n".join(
            f"  - {title} | store: {store} | price: ${price} | categories: {categories}, user rating: {rating if rating is not None else 'N/A'}, avg rating: {round(avg_rating, 2) if avg_rating is not None else 'N/A'}"
            for _, title, categories, store, price, rating, avg_rating in reversed(history_records)
        )

        prompt = build_prompt(stat, taste, high_rating, history_str)
        candidates = generate_candidates(client, prompt, args.model)
        results[uid] = {
            "stat": stat,
            "taste": taste,
            "high_rating": high_rating,
            "candidates": candidates,
        }
        print(f"  [{i+1}/{len(user_ids)}] user {uid} done")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
