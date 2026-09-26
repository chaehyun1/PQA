# profile: 앞서 선택된 페르소나와 3가지 속성
# 3가지 속성: pickiness, habits, unique tastes
# pickiness level sampled in {not picky, moderately picky, extremely picky} based on avg rating
#   - not picky: 평균 평점 >= 4.5
#   - moderately picky: 3.5 <= 평균 평점 < 4.5
#   - extremely picky: 평균 평점 < 3.5
# Habits: personality 3가지 (activity, diversity, conformity)
# unique tastes: 이미 데이터에 있음 (all_personas_like_modify.csv의 taste)

import os
import json
import pickle
import argparse
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "MovieLens")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")

PICKINESS_DICT = {
    "not picky": "You are not picky and tend to give high ratings to most movies you watch.",
    "moderately picky": "You are moderately picky and give ratings based on how well movies match your preferences.",
    "extremely picky": "You are extremely picky and rarely give high ratings unless a movie truly impresses you.",
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


def get_pickiness(avg_rating):
    if avg_rating >= 4.5:
        return "not picky"
    elif avg_rating >= 3.5:
        return "moderately picky"
    else:
        return "extremely picky"


def load_data():
    # user_id_map: original -> remapped
    with open(os.path.join(DATA_DIR, "user_id_map.pkl"), "rb") as f:
        user_id_map = pickle.load(f)
    rev_user_id_map = {v: k for k, v in user_id_map.items()}  # remapped -> original

    # movie_id_map: original movie ID -> remapped ID
    with open(os.path.join(DATA_DIR, "movie_id_map.pkl"), "rb") as f:
        movie_id_map = pickle.load(f)
    rev_movie_id_map = {v: k for k, v in movie_id_map.items()}  # remapped -> original

    # train.txt에서 각 remapped user의 train 아이템 목록 파악
    train_items = {}  # remapped uid -> set of remapped movie IDs
    with open(os.path.join(DATA_DIR, "train.txt")) as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            train_items[uid] = set(int(x) for x in parts[1:])

    # ratings.dat: original user ID -> {original movie ID -> rating}
    all_ratings = {}
    with open(os.path.join(DATA_DIR, "ratings.dat"), encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            uid, mid, rating = int(parts[0]), int(parts[1]), float(parts[2])
            if uid not in all_ratings:
                all_ratings[uid] = {}
            all_ratings[uid][mid] = rating

    # train 데이터에 해당하는 rating만 추출
    user_ratings = {}
    for remapped_uid, item_set in train_items.items():
        orig_uid = rev_user_id_map[remapped_uid]
        ratings_for_user = []
        for remapped_mid in item_set:
            orig_mid = rev_movie_id_map.get(remapped_mid)
            if orig_mid and orig_uid in all_ratings and orig_mid in all_ratings[orig_uid]:
                ratings_for_user.append(all_ratings[orig_uid][orig_mid])
        user_ratings[orig_uid] = ratings_for_user

    # user_statistic.csv: remapped ID -> activity, diversity, conformity
    stat_df = pd.read_csv(os.path.join(DATA_DIR, "user_statistic.csv"))
    user_stat = {int(row["user_id"]): {
        "activity": int(row["activity"]),
        "diversity": int(row["diversity"]),
        "conformity": int(row["conformity"]),
    } for _, row in stat_df.iterrows()}

    # all_personas_like_modify.csv: row index = remapped user ID
    persona_df = pd.read_csv(os.path.join(DATA_DIR, "all_personas_like_modify.csv"))

    return rev_user_id_map, user_ratings, user_stat, persona_df


def build_profile(uid, rev_user_id_map, user_ratings, user_stat, persona_df, selected_persona):
    orig_uid = rev_user_id_map[uid]

    # pickiness
    ratings = user_ratings.get(orig_uid, [])
    avg_rating = sum(ratings) / len(ratings) if ratings else 3.0
    pickiness_level = get_pickiness(avg_rating)

    # habits
    stat = user_stat[uid]
    habits = {
        "activity": ACTIVITY_DICT[stat["activity"]],
        "diversity": DIVERSITY_DICT[stat["diversity"]],
        "conformity": CONFORMITY_DICT[stat["conformity"]],
    }

    # unique tastes
    row = persona_df.iloc[uid]
    unique_tastes = str(row["taste"]).strip()

    return {
        "uid": uid,
        "persona": selected_persona,
        "pickiness": {
            "level": pickiness_level,
            "description": PICKINESS_DICT[pickiness_level],
            "avg_rating": round(avg_rating, 2),
        },
        "habits": habits,
        "unique_tastes": unique_tastes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_users", type=str,
                        default=os.path.join(DATA_DIR, "user_sets.txt"),
                        help="Path to user_sets.txt")
    parser.add_argument("--input", type=str, default=os.path.join(RESULT_DIR, "selected_persona.json"))
    parser.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "user_profiles.json"))
    parser.add_argument("--num_users", type=int, default=None, help="처리할 유저 수 (None이면 전체)")
    args = parser.parse_args()

    print("Loading data...")
    rev_user_id_map, user_ratings, user_stat, persona_df = load_data()

    with open(args.input, encoding="utf-8") as f:
        selected_data = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = {}
    with open(args.target_users, 'r') as f:
        target = set(int(line.strip()) for line in f if line.strip())
    user_ids = [int(k) for k in selected_data.keys() if int(k) in target]
    user_ids = user_ids[:args.num_users]

    print(f"Building profiles for {len(user_ids)} users...")
    for uid in user_ids:
        selected_persona = selected_data[str(uid)]["selected_persona"]
        profile = build_profile(uid, rev_user_id_map, user_ratings, user_stat, persona_df, selected_persona)
        results[uid] = profile
        print(f"  user {uid}: pickiness={profile['pickiness']['level']} (avg={profile['pickiness']['avg_rating']})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
