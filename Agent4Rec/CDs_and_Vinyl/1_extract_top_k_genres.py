# CDs는 unique category가 ~343개로 ML 장르(~18개)보다 많지만,
# activity_0 유저는 train history가 짧아 유저별 unique cats가 mean 6.8 / median 7로 적음.
# top-10은 대부분 유저에서 "전체 cats"와 같아져 truncation 효과 없음.
# top-3 사용: baseline이 작아져 (~1.20) 단일 아이템과 비교 시 ratio 신호 안정적.
# 매칭 1개만 있어도 ratio≈0.83으로 ABOVE 안정 도달 → annotated 신호 깨끗하게 binary (ABOVE vs no annotation).

import argparse
import json
from collections import Counter
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset" / "CDs_and_Vinyl"
DEFAULT_USER_SETS = DATASET_DIR / "user_sets.txt"
TRAIN = DATASET_DIR / "train.json"
META = DATASET_DIR / "meta.json"


def load_item_categories(path: Path) -> dict[int, list[str]]:
    with path.open() as f:
        meta = json.load(f)
    item_categories: dict[int, list[str]] = {}
    for iid_str, info in meta.items():
        cats = info.get("categories", []) or []
        item_categories[int(iid_str)] = [c for c in cats if c]
    return item_categories


def load_user_sets(path: Path) -> set[int]:
    with path.open() as f:
        return {int(line.strip()) for line in f if line.strip()}


def load_train_history(path: Path, target_users: set[int]) -> dict[int, list[int]]:
    with path.open() as f:
        train_raw = json.load(f)
    history_raw = train_raw.get("History", {})
    history: dict[int, list[int]] = {}
    for uid_str, items in history_raw.items():
        uid = int(uid_str)
        if uid not in target_users:
            continue
        history[uid] = [int(x) for x in items]
    return history


def top_k_categories(items: list[int], item_categories: dict[int, list[str]], k: int) -> list[str]:
    counter: Counter[str] = Counter()
    last_seen: dict[str, int] = {}
    for idx, iid in enumerate(items):
        for c in item_categories.get(iid, []):
            counter[c] += 1
            last_seen[c] = idx
    # tie-break: more recent (larger last_seen) wins
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], -last_seen[kv[0]]))
    return [c for c, _ in ranked[:k]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top-k preferred categories per user from train history (CDs).")
    parser.add_argument("-k", "--k", type=int, default=5, help="Number of top categories to extract (default: 3).")
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
        default=Path(__file__).parent / "result" / "user_top_k_categories.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    target_users = load_user_sets(args.user_sets)
    item_categories = load_item_categories(META)
    history = load_train_history(TRAIN, target_users)

    result: dict[str, list[str]] = {}
    for uid in sorted(target_users):
        items = history.get(uid, [])
        result[str(uid)] = top_k_categories(items, item_categories, args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    missing = [u for u in target_users if u not in history]
    print(f"users processed: {len(result)} (missing in train: {len(missing)})")
    print(f"k = {args.k}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
