# profile: 앞서 선택된 페르소나와 3가지 속성 (CDs_and_Vinyl 버전)
# 3가지 속성: pickiness, habits, unique tastes
# pickiness level sampled in {not picky, moderately picky, extremely picky} based on avg rating
#   - not picky: 평균 평점 >= 4.5
#   - moderately picky: 3.5 <= 평균 평점 < 4.5
#   - extremely picky: 평균 평점 < 3.5
# Habits: personality 3가지 (activity, diversity, conformity)
# unique tastes: 이미 데이터에 있음 (all_persona.csv의 taste)

import os
import json
import argparse
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "CDs_and_Vinyl")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")

PICKINESS_DICT = {
    "not picky": "You are not picky and tend to give high ratings to most albums you listen to.",
    "moderately picky": "You are moderately picky and give ratings based on how well albums match your preferences.",
    "extremely picky": "You are extremely picky and rarely give high ratings unless an album truly impresses you.",
}

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


def get_pickiness(avg_rating):
    if avg_rating >= 4.5:
        return "not picky"
    elif avg_rating >= 3.5:
        return "moderately picky"
    else:
        return "extremely picky"


def load_data():
    # train.json: {"History": {uid: [iids]}, "Time": {...}}
    with open(os.path.join(DATA_DIR, "train.json")) as f:
        train_raw = json.load(f)
    train_history = train_raw.get("History", {})

    # review.json: uid (str) -> {iid (str) -> {rating, title, text}}
    with open(os.path.join(DATA_DIR, "review.json")) as f:
        review = json.load(f)

    # train 데이터에 해당하는 rating 추출: uid(int) -> [rating, ...]
    user_ratings = {}
    for uid_str, items in train_history.items():
        uid = int(uid_str)
        user_reviews = review.get(uid_str, {})
        ratings_for_user = []
        for iid in items:
            iid_key = str(iid)
            if iid_key in user_reviews:
                r = user_reviews[iid_key].get("rating")
                if r is not None:
                    ratings_for_user.append(float(r))
        user_ratings[uid] = ratings_for_user

    # personality.json: activity/diversity/conformity {uid(str): level(int)}
    with open(os.path.join(DATA_DIR, "personality.json")) as f:
        personality = json.load(f)
    user_stat = {}
    for uid_str in personality["activity"].keys():
        uid = int(uid_str)
        user_stat[uid] = {
            "activity": int(personality["activity"][uid_str]),
            "diversity": int(personality["diversity"][uid_str]),
            "conformity": int(personality["conformity"][uid_str]),
        }

    # all_persona.csv: user_id, taste, reasons, high_rating, low_rating
    persona_df = pd.read_csv(os.path.join(DATA_DIR, "all_persona.csv"))
    persona_map = {int(row["user_id"]): row for _, row in persona_df.iterrows()}

    return user_ratings, user_stat, persona_map


def build_profile(uid, user_ratings, user_stat, persona_map, selected_persona):
    # pickiness
    ratings = user_ratings.get(uid, [])
    avg_rating = sum(ratings) / len(ratings) if ratings else 3.0
    pickiness_level = get_pickiness(avg_rating)

    # habits
    stat = user_stat[uid]
    habits = {
        "activity": activity_dict[stat["activity"]],
        "diversity": diversity_dict[stat["diversity"]],
        "conformity": conformity_dict[stat["conformity"]],
    }

    # unique tastes
    row = persona_map[uid]
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
    parser.add_argument("--target_users", type=str, default=os.path.join(DATA_DIR, "user_sets.txt"))
    parser.add_argument("--input", type=str, default=os.path.join(RESULT_DIR, "selected_persona.json"))
    parser.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "user_profiles.json"))
    parser.add_argument("--num_users", type=int, default=None, help="처리할 유저 수")
    args = parser.parse_args()

    print("Loading data...")
    user_ratings, user_stat, persona_map = load_data()

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
        profile = build_profile(uid, user_ratings, user_stat, persona_map, selected_persona)
        results[uid] = profile
        print(f"  user {uid}: pickiness={profile['pickiness']['level']} (avg={profile['pickiness']['avg_rating']})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
