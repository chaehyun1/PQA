# short summary를 historical data와 함께 프롬프트에 넣어서 LLM이 페르소나 후보를 생성하도록 함
# 가능한 age, personality, occupation도 제공
# 5개의 candidate persona 생성

# personality의 경우 Agent4Rec에서 쓴 세 가지 그대로 이용함
# short summary의 경우 마찬가지로 데이터셋에 있는 taste 및 high rating을 사용함

import os
import pickle
import argparse
import json
import pandas as pd
from openai import OpenAI

NUM_HISTORY = 15 # interaction history에서 최신 기준 몇개까지 볼 지

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "MovieLens")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")

AGE_MAP = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

OCC_MAP = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}

ACTIVITY_DICT = {
    1: "An Incredibly Elusive Occasional Viewer, so seldom attracted by movie recommendations that it's almost a legendary event when you do watch a movie. Your movie-watching habits are extraordinarily infrequent. And you will exit the recommender system immediately even if you just feel little unsatisfied.",
    2: "An Occasional Viewer, seldom attracted by movie recommendations. Only curious about watching movies that strictly align the taste. The movie-watching habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.",
    3: "A Movie Enthusiast with an insatiable appetite for films, willing to watch nearly every movie recommended to you. Movies are a central part of your life, and movie recommendations are integral to your existence. You are tolerant of recommender system, which means you are not easy to exit recommender system even if you have some unsatisfied memory."
}

CONFORMITY_DICT = {
    1: "A Dedicated Follower who gives ratings heavily relies on movie historical ratings, rarely expressing independent opinions. Usually give ratings that are same as historical ratings.",
    2: "A Balanced Evaluator who considers both historical ratings and personal preferences when giving ratings to movies. Sometimes give ratings that are different from historical rating.",
    3: "A Maverick Critic who completely ignores historical ratings and evaluates movies solely based on own taste. Usually give ratings that are a lot different from historical ratings."
}

DIVERSITY_DICT = {
    1: "An Exceedingly Discerning Selective Viewer who watches movies with a level of selectivity that borders on exclusivity. The movie choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
    2: "A Niche Explorer who occasionally explores different genres and mostly sticks to preferred movie types.",
    3: "A Cinematic Trailblazer, a relentless seeker of the unique and the obscure in the world of movies. The movie choices are so diverse and avant-garde that they defy categorization."
}


def load_data():
    # user_id_map: original MovieLens ID -> remapped ID
    with open(os.path.join(DATA_DIR, "user_id_map.pkl"), "rb") as f:
        user_id_map = pickle.load(f)
    rev_user_id_map = {v: k for k, v in user_id_map.items()}  # remapped -> original

    # users.dat: original ID -> (gender, age, occupation)
    user_demo = {}
    with open(os.path.join(DATA_DIR, "users.dat"), encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            uid, gender, age, occ = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            user_demo[uid] = {"gender": gender, "age": AGE_MAP[age], "occupation": OCC_MAP[occ]}

    # user_statistic.csv: remapped ID -> activity, diversity, conformity
    stat_df = pd.read_csv(os.path.join(DATA_DIR, "user_statistic.csv"))
    user_stat = {}
    for _, row in stat_df.iterrows():
        user_stat[int(row["user_id"])] = {
            "activity": ACTIVITY_DICT[int(row["activity"])],
            "diversity": DIVERSITY_DICT[int(row["diversity"])],
            "conformity": CONFORMITY_DICT[int(row["conformity"])],
        }

    # all_personas_like_modify.csv: row index = remapped user ID (0-based)
    persona_df = pd.read_csv(os.path.join(DATA_DIR, "all_personas_like_modify.csv"))

    # movie_id_map: original MovieLens movie ID -> remapped ID
    with open(os.path.join(DATA_DIR, "movie_id_map.pkl"), "rb") as f:
        movie_id_map = pickle.load(f)
    rev_movie_id_map = {v: k for k, v in movie_id_map.items()}  # remapped -> original

    # movie_detail.csv: remapped movie ID (0-based index) -> title, genres
    movie_df = pd.read_csv(os.path.join(DATA_DIR, "movie_detail.csv"))
    movie_info = {int(row["movie_id"]): {"title": row["title"], "genres": row["genres"], "avg_rating": row["rating"]}
                  for _, row in movie_df.iterrows()}

    # ratings.dat: original user ID -> {original movie ID -> (rating, timestamp)}
    ratings = {}
    with open(os.path.join(DATA_DIR, "ratings.dat"), encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            uid, mid, rating, ts = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
            if uid not in ratings:
                ratings[uid] = {}
            ratings[uid][mid] = (rating, ts)

    # train.txt: remapped user ID -> list of remapped movie IDs
    train = {}
    with open(os.path.join(DATA_DIR, "train.txt")) as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            train[uid] = [int(x) for x in parts[1:]]

    return rev_user_id_map, user_demo, user_stat, persona_df, rev_movie_id_map, movie_info, ratings, train


def get_recent_interactions(uid, orig_uid, train, rev_movie_id_map, movie_info, ratings, n=NUM_HISTORY):
    """최신순으로 최대 n개의 interaction history 반환."""
    items = train.get(uid, [])
    records = []
    for remapped_mid in items:
        orig_mid = rev_movie_id_map.get(remapped_mid)
        if orig_mid is None:
            continue
        rating, ts = ratings.get(orig_uid, {}).get(orig_mid, (None, 0))
        info = movie_info.get(remapped_mid, {})
        title = info.get("title", f"Unknown (id={remapped_mid})")
        genres = info.get("genres", "Unknown")
        avg_rating = info.get("avg_rating")
        records.append((ts, title, genres, rating, avg_rating))
    records.sort(key=lambda x: x[0], reverse=True) 
    return records[:n]


def build_prompt(demo, stat, taste, high_rating, history):
    return f"""You are given information about a movie user. Based on this information, generate a persona description that summarizes who this user is.

## User Information
- Age: {demo['age']}
- Gender: {demo['gender']}
- Occupation: {demo['occupation']}
- Activity (The activity characteristic pertains to the frequency of your movie-watching habits): {stat['activity']}
- Diversity (The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste): {stat['diversity']}
- Conformity (The conformity characteristic measures the degree to which your ratings are influenced by historical ratings): {stat['conformity']}
- Movie taste: {taste}
- High rating pattern: {high_rating}

## Interaction History (ordered from oldest to most recent)
{history}

## Instructions
Generate a concise persona description (2 sentences) that integrates all the above information into a coherent user profile. Output only the persona description, nothing else.
"""


def generate_candidates(client, prompt, model, n=1):  # 후보 1개 생성
    candidates = []
    for _ in range(n):
        resp = client.responses.create(model=model, input=prompt)
        candidates.append(resp.output_text.strip())
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_users", type=str,
                        default=os.path.join(DATA_DIR, "user_sets.txt"),
                        help="Path to user_sets.txt")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num_users", type=int, default=None, help="처리할 유저 수 (None이면 전체)")
    parser.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "persona_candidates.json"))
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading data...")
    rev_user_id_map, user_demo, user_stat, persona_df, rev_movie_id_map, movie_info, ratings, train = load_data()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = {}
    with open(args.target_users, 'r') as f:
        user_ids = [int(line.strip()) for line in f if line.strip()]
    user_ids = user_ids[:args.num_users]

    print(f"Generating persona candidates for {len(user_ids)} users...")
    for i, uid in enumerate(user_ids):
        orig_uid = rev_user_id_map[uid]
        demo = user_demo[orig_uid]
        stat = user_stat[uid]
        row = persona_df.iloc[uid]
        taste = str(row["taste"]).strip()
        high_rating = str(row["high_rating"]).strip()

        history_records = get_recent_interactions(uid, orig_uid, train, rev_movie_id_map, movie_info, ratings)
        history_str = "\n".join(
            f"  - {title} ({genres}), user rating: {rating if rating is not None else 'N/A'}, avg rating: {round(avg_rating, 2) if avg_rating is not None else 'N/A'}"
            for _, title, genres, rating, avg_rating in reversed(history_records)
        )

        prompt = build_prompt(demo, stat, taste, high_rating, history_str)
        candidates = generate_candidates(client, prompt, args.model)
        results[uid] = {
            "demo": demo,
            "stat": stat,
            "candidates": candidates,
        }
        print(f"  [{i+1}/{len(user_ids)}] user {uid} done")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
