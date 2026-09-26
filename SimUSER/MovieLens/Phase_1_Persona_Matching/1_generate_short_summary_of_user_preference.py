# 논문의 Persona Extraction 부분 참고, short summary 생성하는 코드임 
# 이미 MovieLens 데이터셋에서 결과(taste, rating pattern)이 주어졌기 때문에(논문 내용 참고), **돌릴 필요 없음**
import os
import pickle
import argparse
import json
import random
from openai import OpenAI
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "MovieLens")
NUM_SAMPLE = 20  

def load_data():
    """Load all necessary data files."""
    with open(os.path.join(DATA_DIR, "user_id_map.pkl"), "rb") as f:
        user_id_map = pickle.load(f)
    with open(os.path.join(DATA_DIR, "movie_id_map.pkl"), "rb") as f:
        movie_id_map = pickle.load(f)

    # Reverse maps (remapped -> original)
    rev_user = {v: k for k, v in user_id_map.items()}
    rev_movie = {v: k for k, v in movie_id_map.items()}

    # Movie details (remapped_id -> title, genres)
    movie_df = pd.read_csv(os.path.join(DATA_DIR, "movie_detail.csv"))
    movie_info = {}
    for _, row in movie_df.iterrows():
        movie_info[row["movie_id"]] = {
            "title": row["title"],
            "genres": row["genres"],
        }

    # Ratings (original_user -> {original_movie: (rating, timestamp)})
    ratings = {}
    with open(os.path.join(DATA_DIR, "ratings.dat")) as f:
        for line in f:
            parts = line.strip().split("::")
            uid, mid, rating, ts = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
            if uid not in ratings:
                ratings[uid] = {}
            ratings[uid][mid] = (rating, ts)

    # Train interactions (remapped user -> list of remapped item ids)
    train = {}
    with open(os.path.join(DATA_DIR, "train.txt")) as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            train[uid] = items

    return rev_user, rev_movie, movie_info, ratings, train


def get_random_items_with_labels(user_id, rev_user, rev_movie, movie_info, ratings, train):
    """Get randomly sampled N items with liked/disliked labels for a user."""
    orig_uid = rev_user[user_id]
    items = train[user_id]

    # Get (remapped_item_id, rating) for each item
    item_records = []
    for iid in items:
        orig_mid = rev_movie[iid]
        if orig_uid in ratings and orig_mid in ratings[orig_uid]:
            rating, _ = ratings[orig_uid][orig_mid]
            item_records.append((iid, rating))

    # Randomly sample N items (or all if fewer than N)
    selected = random.sample(item_records, min(len(item_records), NUM_SAMPLE))

    liked = []
    disliked = []
    for iid, rating in selected:
        info = movie_info.get(iid, {})
        title = info.get("title", f"Unknown (id={iid})")
        genres = info.get("genres", "Unknown")
        entry = f"{title} ({genres})"
        if rating >= 3:
            liked.append(entry)
        else:
            disliked.append(entry)

    return liked, disliked


def build_prompt(liked, disliked):
    """Build the prompt for LLM to generate user preference summary."""
    liked_str = "\n".join(f"  - {item}" for item in liked) if liked else "  (none)"
    disliked_str = "\n".join(f"  - {item}" for item in disliked) if disliked else "  (none)"

    prompt = f"""Based on the following user's movie interaction history, generate a short summary (1-2 sentences) of the user's movie preferences.

Liked movies (rated 3 or above):
{liked_str}

Disliked movies (rated below 3):
{disliked_str}

Only include preferences that are clearly supported by the data above. Do not speculate or infer preferences that are not evident from the listed movies.
If a genre appears in both liked and disliked lists, do not simply say the user likes or dislikes that genre. Either explain the specific difference (e.g., "enjoys romantic comedies but not slapstick comedies") or omit that genre entirely if no clear distinction can be made.
Do not mention specific movie titles in the summary. Only describe preferences in terms of genres, themes, and styles."""

    return prompt


def generate_summary(client, prompt, model):
    """Call OpenAI API to generate a summary."""
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "..", "result", "user_preference_summaries.json"))
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    print("Loading data...")
    rev_user, rev_movie, movie_info, ratings, train = load_data()

    user_ids = list(train.keys())[:args.num_users]
    results = {}

    print(f"Generating summaries for {len(user_ids)} users...")
    for i, uid in enumerate(user_ids):
        liked, disliked = get_random_items_with_labels(uid, rev_user, rev_movie, movie_info, ratings, train)
        prompt = build_prompt(liked, disliked)
        summary = generate_summary(client, prompt, args.model)
        results[uid] = {
            "liked": liked,
            "disliked": disliked,
            "summary": summary,
        }
        print(f"  [{i+1}/{len(user_ids)}] User {uid}: {len(liked)} liked, {len(disliked)} disliked -> summary generated")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
