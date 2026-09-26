# 필요한 캡션 생성을 위해 아이템의 포스터 이미지를 로컬에 다운로드하는 스크립트

import os
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "MovieLens")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")
CAPTION_CACHE = os.path.join(RESULT_DIR, "captions.json")
POSTER_DIR = os.path.join(RESULT_DIR, "posters")


def get_test_uniform_items() -> set[int]:
    """test.txt에 있는 모든 아이템 ID 집합 반환."""
    items = set()
    test_path = os.path.join(DATA_DIR, "test.txt") # NOTE: 필요 시 여기 수정
    with open(test_path) as f:
        for line in f:
            parts = line.strip().split()
            for mid_str in parts[1:]:  # 첫 번째는 user ID
                items.add(int(mid_str))
    return items


def get_missing_caption_items() -> set[int]:
    """test 아이템 중 captions.json에 캡션이 없는 아이템 집합 반환."""
    test_items = get_test_uniform_items()
    if not os.path.exists(CAPTION_CACHE):
        print("captions.json not found. All test items need captions.")
        return test_items
    with open(CAPTION_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    cached_ids = {int(k) for k in cache.keys()}
    return test_items - cached_ids


def load_movie_info() -> dict[int, dict]:
    df = pd.read_csv(os.path.join(DATA_DIR, "movie_detail.csv"))
    info = {}
    for _, row in df.iterrows():
        mid = int(row["movie_id"])
        url = str(row.get("poster_url", ""))
        if url == "nan":
            url = ""
        info[mid] = {"title": row["title"], "poster_url": url}
    return info


def download_one(mid: int, url: str, out_path: str) -> tuple[int, bool, str]:
    """이미지 하나 다운로드. (movie_id, 성공여부, 메시지) 반환."""
    if os.path.exists(out_path):
        return mid, True, "already exists"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return mid, True, "downloaded"
    except Exception as e:
        return mid, False, str(e)


def main():
    missing_ids = get_missing_caption_items()
    print(f"Missing caption items: {len(missing_ids)}")

    if not missing_ids:
        print("All test_uniform items already have captions.")
        return

    movie_info = load_movie_info()
    os.makedirs(POSTER_DIR, exist_ok=True)

    # 다운로드할 목록 준비
    tasks = []
    no_url = 0
    for mid in sorted(missing_ids):
        info = movie_info.get(mid)
        if info is None or not info["poster_url"]:
            no_url += 1
            continue
        out_path = os.path.join(POSTER_DIR, f"{mid}.jpg")
        tasks.append((mid, info["poster_url"], out_path))

    print(f"To download: {len(tasks)} (no URL: {no_url})")

    # 병렬 다운로드
    success, fail = 0, 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_one, mid, url, path): mid for mid, url, path in tasks}
        for future in as_completed(futures):
            mid, ok, msg = future.result()
            if ok:
                success += 1
            else:
                fail += 1
                print(f"  FAIL movie_id={mid}: {msg}")

    print(f"\nDone. success={success}, fail={fail}, saved to {POSTER_DIR}")


if __name__ == "__main__":
    main()
