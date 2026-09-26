import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset" / "MovieLens"
DEFAULT_USER_SETS = DATASET_DIR / "user_sets.txt"
TRAIN = DATASET_DIR / "train_sorted.txt"
MOVIE_DETAIL = DATASET_DIR / "movie_detail.csv"


def load_item_genres(path: Path) -> dict[int, list[str]]:
    item_genres: dict[int, list[str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = int(row["movie_id"])
            genres = [g for g in row["genres"].split("|") if g]
            item_genres[iid] = genres
    return item_genres


def load_user_sets(path: Path) -> set[int]:
    with path.open() as f:
        return {int(line.strip()) for line in f if line.strip()}


def load_train_history(path: Path, target_users: set[int]) -> dict[int, list[int]]:
    history: dict[int, list[int]] = {}
    with path.open() as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            uid = int(parts[0])
            if uid not in target_users:
                continue
            history[uid] = [int(x) for x in parts[1:]]
    return history


def top_k_genres(items: list[int], item_genres: dict[int, list[str]], k: int) -> list[str]:
    counter: Counter[str] = Counter()
    last_seen: dict[str, int] = {}
    for idx, iid in enumerate(items):
        for g in item_genres.get(iid, []):
            counter[g] += 1
            last_seen[g] = idx
    # tie-break: more recent (larger last_seen) wins
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], -last_seen[kv[0]]))
    return [g for g, _ in ranked[:k]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top-k preferred genres per user from train history.")
    parser.add_argument("-k", "--k", type=int, default=5, help="Number of top genres to extract (default: 5).")
    parser.add_argument(
        "-u",
        "--user_sets",
        type=Path,
        default=DEFAULT_USER_SETS,
        help="Path to user_sets file (one user_id per line).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "result" / "user_top_k_genres.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    target_users = load_user_sets(args.user_sets)
    item_genres = load_item_genres(MOVIE_DETAIL)
    history = load_train_history(TRAIN, target_users)

    result: dict[str, list[str]] = {}
    for uid in sorted(target_users):
        items = history.get(uid, [])
        result[str(uid)] = top_k_genres(items, item_genres, args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    missing = [u for u in target_users if u not in history]
    print(f"users processed: {len(result)} (missing in train: {len(missing)})")
    print(f"k = {args.k}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
