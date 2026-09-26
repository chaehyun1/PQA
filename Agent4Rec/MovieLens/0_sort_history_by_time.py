# 실행시킬 필요 없음 

import pickle
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset" / "MovieLens"
RATINGS = DATASET_DIR / "ratings.dat"
USER_MAP = DATASET_DIR / "user_id_map.pkl"
MOVIE_MAP = DATASET_DIR / "movie_id_map.pkl"
SPLITS = ["train.txt", "valid.txt", "test.txt"]


def load_id_maps() -> tuple[dict[int, int], dict[int, int]]:
    with USER_MAP.open("rb") as f:
        user_map_raw = pickle.load(f)  # raw -> mapped
    with MOVIE_MAP.open("rb") as f:
        movie_map_raw = pickle.load(f)
    user_map = {int(k): int(v) for k, v in user_map_raw.items()}
    movie_map = {int(k): int(v) for k, v in movie_map_raw.items()}
    return user_map, movie_map


def load_timestamps(user_map: dict[int, int], movie_map: dict[int, int]) -> dict[tuple[int, int], int]:
    ts: dict[tuple[int, int], int] = {}
    with RATINGS.open() as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            raw_u, raw_m, _, t = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
            if raw_u not in user_map or raw_m not in movie_map:
                continue
            ts[(user_map[raw_u], movie_map[raw_m])] = t
    return ts


def sort_split(src: Path, dst: Path, ts: dict[tuple[int, int], int]) -> tuple[int, int]:
    """Returns (lines_processed, items_missing_ts)."""
    missing = 0
    lines = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            parts = line.split()
            if not parts:
                fout.write(line)
                continue
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            # stable sort by timestamp ascending; items without ts go to the end with their original order
            indexed = []
            for idx, iid in enumerate(items):
                t = ts.get((uid, iid))
                if t is None:
                    missing += 1
                    indexed.append((float("inf"), idx, iid))
                else:
                    indexed.append((t, idx, iid))
            indexed.sort(key=lambda x: (x[0], x[1]))
            sorted_items = [str(iid) for _, _, iid in indexed]
            fout.write(" ".join([str(uid), *sorted_items]) + "\n")
            lines += 1
    return lines, missing


def main() -> None:
    user_map, movie_map = load_id_maps()
    ts = load_timestamps(user_map, movie_map)
    print(f"loaded {len(ts)} (user, item) -> timestamp entries")

    for split in SPLITS:
        src = DATASET_DIR / split
        dst = DATASET_DIR / split.replace(".txt", "_sorted.txt")
        lines, missing = sort_split(src, dst, ts)
        print(f"{split} -> {dst.name}: {lines} lines, {missing} items missing timestamp")


if __name__ == "__main__":
    main()
