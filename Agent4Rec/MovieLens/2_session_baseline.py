import argparse
import csv
import json
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset" / "MovieLens"
RESULT_DIR = Path(__file__).parent / "result"
DEFAULT_USER_SETS = DATASET_DIR / "user_sets.txt"
TRAIN_SORTED = DATASET_DIR / "train_sorted.txt"
MOVIE_DETAIL = DATASET_DIR / "movie_detail.csv"
DEFAULT_TOPK = RESULT_DIR / "user_top_k_genres.json"


def load_item_genres(path: Path) -> dict[int, set[str]]:
    item_genres: dict[int, set[str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            iid = int(row["movie_id"])
            item_genres[iid] = {g for g in row["genres"].split("|") if g}
    return item_genres


def load_user_sets(path: Path) -> list[int]:
    with path.open() as f:
        return sorted({int(line.strip()) for line in f if line.strip()})


def load_train_history(path: Path, target_users: set[int]) -> dict[int, list[int]]:
    history: dict[int, list[int]] = {}
    with path.open() as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            uid = int(parts[0])
            if uid in target_users:
                history[uid] = [int(x) for x in parts[1:]]
    return history


def split_sessions(items: list[int], w: int) -> list[list[int]]:
    """Cut from the most recent end. session 0 = newest. Oldest leftover kept as last session.

    Examples (w=10):
      len=25  -> [newest10, mid10, oldest5]   (3 sessions, last size 5)
      len=20  -> [newest10, oldest10]         (2 sessions)
      len=7   -> [all 7]                      (1 session, smaller than w)
    """
    if len(items) <= w:
        return [list(items)]
    sessions: list[list[int]] = []
    end = len(items)
    while end - w >= 0:
        sessions.append(items[end - w : end])
        end -= w
    if end > 0:
        sessions.append(items[0:end])
    return sessions


def session_overlap(session: list[int], top_k: set[str], item_genres: dict[int, set[str]]) -> float:
    """Average number of top-k genre matches per item (range 0..k).

    For each item, count how many of its genres are in top_k; sum over the
    session and divide by the number of items.
    """
    if not session:
        return 0.0
    total_matches = sum(len(item_genres.get(iid, set()) & top_k) for iid in session)
    return total_matches / len(session)


def recency_weights(n: int, gamma: float) -> list[float]:
    """w_i = gamma^i where i=0 is most recent, normalized to sum to 1."""
    raw = [gamma**i for i in range(n)]
    s = sum(raw)
    return [r / s for r in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-session genre-overlap baseline with recency weighting.")
    parser.add_argument("-w", "--window", type=int, default=10, help="Session window size (default: 10).")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help="Exponential decay for recency (default: 0.9).")
    parser.add_argument("-u", "--user_sets", type=Path, default=DEFAULT_USER_SETS, help="Path to user_sets file (one user_id per line).")
    parser.add_argument("--top-k-path", type=Path, default=DEFAULT_TOPK, help="Per-user top-k genres JSON from step 1.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path (default: result/session_baseline_w{W}.json).")
    args = parser.parse_args()

    if args.output is None:
        args.output = RESULT_DIR / f"session_baseline_w{args.window}.json"

    users = load_user_sets(args.user_sets)
    item_genres = load_item_genres(MOVIE_DETAIL)
    history = load_train_history(TRAIN_SORTED, set(users))
    with args.top_k_path.open() as f:
        top_k_map: dict[str, list[str]] = json.load(f)

    result: dict[str, dict] = {}
    empty_users = 0
    for uid in users:
        items = history.get(uid, [])
        top_k = set(top_k_map.get(str(uid), []))
        if not items or not top_k:
            empty_users += 1
            result[str(uid)] = {"baseline": None, "num_sessions": 0, "sessions": []}
            continue
        sessions = split_sessions(items, args.window)
        overlaps = [session_overlap(s, top_k, item_genres) for s in sessions]
        weights = recency_weights(len(sessions), args.gamma)
        baseline = sum(w * o for w, o in zip(weights, overlaps))
        result[str(uid)] = {
            "baseline": round(baseline, 6),
            "num_sessions": len(sessions),
            "sessions": [
                {"index": i, "size": len(s), "overlap": round(o, 6), "weight": round(w, 6)}
                for i, (s, o, w) in enumerate(zip(sessions, overlaps, weights))
            ],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"users processed: {len(result)} (skipped/empty: {empty_users})")
    print(f"window={args.window}, gamma={args.gamma}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
