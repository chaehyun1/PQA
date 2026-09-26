# 대상: user_sets_activity_0.txt (CDs low-activity 그룹, 500명)
# Train history 통계:
#   n=500 | min=3 | mean=4.03 | median=4 | max=5
#   p25=3 | p75=5 | p90=5
# History가 짧아서 w=10이면 모두 1 session으로 처리됨 → recency weighting 비활성화.
# 결과적으로 baseline = 유저의 전체 train history에 대한 평균 top-k category overlap.
# CDs는 데이터 자체가 sparse (전체 14335명 중 49%가 ≤5 items)라 multi-session 의미 없음.

import argparse
import json
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset" / "CDs_and_Vinyl"
RESULT_DIR = Path(__file__).parent / "result"
DEFAULT_USER_SETS = DATASET_DIR / "user_sets.txt"
TRAIN = DATASET_DIR / "train.json"
META = DATASET_DIR / "meta.json"
DEFAULT_TOPK = RESULT_DIR / "user_top_k_categories.json"


def load_item_categories(path: Path) -> dict[int, set[str]]:
    with path.open() as f:
        meta = json.load(f)
    item_categories: dict[int, set[str]] = {}
    for iid_str, info in meta.items():
        cats = info.get("categories", []) or []
        item_categories[int(iid_str)] = {c for c in cats if c}
    return item_categories


def load_user_sets(path: Path) -> list[int]:
    with path.open() as f:
        return sorted({int(line.strip()) for line in f if line.strip()})


def load_train_history(path: Path, target_users: set[int]) -> dict[int, list[int]]:
    with path.open() as f:
        train_raw = json.load(f)
    history_raw = train_raw.get("History", {})
    history: dict[int, list[int]] = {}
    for uid_str, items in history_raw.items():
        uid = int(uid_str)
        if uid in target_users:
            history[uid] = [int(x) for x in items]
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


def session_overlap(session: list[int], top_k: set[str], item_categories: dict[int, set[str]]) -> float:
    """Average number of top-k category matches per item (range 0..k).

    For each item, count how many of its categories are in top_k; sum over the
    session and divide by the number of items.
    """
    if not session:
        return 0.0
    total_matches = sum(len(item_categories.get(iid, set()) & top_k) for iid in session)
    return total_matches / len(session)


def recency_weights(n: int, gamma: float) -> list[float]:
    """w_i = gamma^i where i=0 is most recent, normalized to sum to 1."""
    raw = [gamma**i for i in range(n)]
    s = sum(raw)
    return [r / s for r in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-session category-overlap baseline with recency weighting (CDs).")
    parser.add_argument("-w", "--window", type=int, default=10, help="Session window size (default: 10; activity_0 history는 짧아 모두 1 session으로 떨어짐).")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help="Exponential decay for recency (default: 0.9).")
    parser.add_argument("-u", "--user_sets", type=Path, default=DEFAULT_USER_SETS, help="Path to user_sets file (one user_id per line).")
    parser.add_argument("--top-k-path", type=Path, default=DEFAULT_TOPK, help="Per-user top-k categories JSON from step 1.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path (default: result/session_baseline_w{W}.json).")
    args = parser.parse_args()

    if args.output is None:
        args.output = RESULT_DIR / f"session_baseline_w{args.window}.json"

    users = load_user_sets(args.user_sets)
    item_categories = load_item_categories(META)
    history = load_train_history(TRAIN, set(users))
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
        overlaps = [session_overlap(s, top_k, item_categories) for s in sessions]
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
