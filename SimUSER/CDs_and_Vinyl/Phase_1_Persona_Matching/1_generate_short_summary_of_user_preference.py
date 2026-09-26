# 논문의 Persona Extraction 부분 참고, short summary 생성하는 코드임 (CDs_and_Vinyl 버전)
import os
import argparse
import json
import random
from openai import OpenAI

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "CDs_and_Vinyl")
NUM_SAMPLE = 20


def load_data():
    """Load all necessary data files for CDs_and_Vinyl."""
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)

    with open(os.path.join(DATA_DIR, "review.json")) as f:
        review = json.load(f)

    with open(os.path.join(DATA_DIR, "train.json")) as f:
        train_raw = json.load(f)
    train = train_raw["History"] if "History" in train_raw else train_raw

    # item_id(str) -> {title, categories, store}
    item_info = {}
    for iid, info in meta.items():
        item_info[iid] = {
            "title": info.get("title", f"Unknown (id={iid})"),
            "categories": ", ".join(info.get("categories", [])) or "Unknown",
            "store": info.get("store", ""),
            "price": info.get("price", ""),
        }

    return item_info, review, train


def get_random_items_with_labels(user_id, item_info, review, train):
    """Get randomly sampled N items with liked/disliked labels for a user."""
    uid_key = str(user_id)
    items = train.get(uid_key, [])
    user_reviews = review.get(uid_key, {})

    item_records = []
    for iid in items:
        iid_key = str(iid)
        if iid_key in user_reviews:
            rating = user_reviews[iid_key].get("rating")
            if rating is not None:
                item_records.append((iid_key, float(rating)))

    selected = random.sample(item_records, min(len(item_records), NUM_SAMPLE))

    liked = []
    disliked = []
    for iid, rating in selected:
        info = item_info.get(iid, {})
        title = info.get("title", f"Unknown (id={iid})")
        categories = info.get("categories", "Unknown")
        store = info.get("store", "")
        price = info.get("price", "")
        parts = [title]
        if store:
            parts.append(f"store: {store}")
        if price:
            parts.append(f"price: ${price}")
        parts.append(f"categories: {categories}")
        entry = " | ".join(parts)
        if rating > 3:
            liked.append(entry)
        elif rating < 3:
            disliked.append(entry)

    return liked, disliked


def build_prompt(liked, disliked):
    """Build the prompt for LLM to generate user preference summary."""
    liked_str = "\n".join(f"  - {item}" for item in liked) if liked else "  (none)"
    disliked_str = "\n".join(f"  - {item}" for item in disliked) if disliked else "  (none)"

    prompt = f"""Based on the following user's music (CDs/Vinyl) interaction history, generate a short summary (1-2 sentences) of the user's music preferences.

Liked albums (rated >= 4):
{liked_str}

Disliked albums (rated <= 2):
{disliked_str}

Only include preferences that are clearly supported by the data above. Do not speculate or infer preferences that are not evident from the listed albums.
If a genre appears in both liked and disliked lists, do not simply say the user likes or dislikes that genre. Either explain the specific difference (e.g., "enjoys classic rock but not modern hard rock") or omit that genre entirely if no clear distinction can be made.
Do not mention specific album titles or artist names in the summary. Only describe preferences in terms of genres, themes, and styles."""

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
    parser.add_argument("--target_users", type=str, default=os.path.join(DATA_DIR, "user_sets.txt"))
    parser.add_argument("--num_users", type=int, default=None)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "..", "result", "user_preference_summaries.json"))
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    print("Loading data...")
    item_info, review, train = load_data()

    with open(args.target_users) as f:
        user_ids = [l.strip() for l in f if l.strip()]
    if args.num_users:
        user_ids = user_ids[:args.num_users]
    results = {}

    print(f"Generating summaries for {len(user_ids)} users...")
    for i, uid in enumerate(user_ids):
        liked, disliked = get_random_items_with_labels(uid, item_info, review, train)
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
