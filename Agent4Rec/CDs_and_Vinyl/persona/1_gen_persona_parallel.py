import os
import json
import time
import argparse
import random
from math import ceil
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

# 프롬프트를 기반으로 페르소나를 생성하는 단계 
# 사용자의 히스토리, 평점, 아이템 메타 데이터를 기반으로 페르소나 생성하는 코드 
# 결과: taste, reason, description of rating

def persona_prompts(interaction_his: str) -> str:
    return f"""
## ROLE & GOAL ##
You are an expert specializing in analyzing user interactions within a recommendation system. Your goal is to infer a user's preferences based on their historical data.

## CONTEXT ##
You will be given the user's interaction history, metadata for various items, and the user's ratings for those items.

## User Interaction History ##
{interaction_his}

## INSTRUCTIONS ##
Based on the provided interaction history, item metadata, and user ratings, you must:
1.  Identify 1-3 distinct user tastes. Each taste description must be under 50 words.
2.  For each taste identified, provide a concise reason based on the available data. Each reason must be under 50 words.
3.  Describe the general characteristics of items the user tends to rate highly. This description must be under 50 words. **Do not mention specific item names.**
4.  Describe the general characteristics of items the user tends to rate lowly. This description must be under 50 words. **Do not mention specific item names.**

## REQUIRED OUTPUT FORMAT ##
Your entire response must strictly follow this format, using "|" as a separator for multiple entries.

**Taste:** [Taste 1] | [Taste 2]..
**Reasons:** [Reason for Taste 1] | [Reason for Taste 2]..
**High Rating:** [Description of characteristics for items with high ratings]
**Low Rating:** [Description of characteristics for items with low ratings]
"""


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def chunk_n(seq, n_chunks: int):
    n = len(seq)
    if n == 0 or n_chunks <= 1:
        return [seq]
    step = ceil(n / n_chunks)
    return [seq[i:i + step] for i in range(0, n, step)]


def get_user_ratings(his, review, user_key):
    rating = {}
    try:
        rev = review[user_key]
        for uu in his:
            rating[uu] = rev.get(str(uu), {}).get('rating', 'None')
    except Exception:
        for uu in his:
            rating[uu] = 'None'
    return rating


def build_interaction_his(his, meta, rating):
    interaction_his = "My recent purchased items are "
    for uu in his:
        m = meta.get(str(uu), {})
        title = (m.get('title') or '').strip()
        cats = m.get('categories') or []
        cats_joined = ",".join(cats).strip() if len(cats) >= 2 else ""
        price = str(m.get('price')).strip()
        store = str(m.get('store')).strip()
        interaction_his += (
            f"Title:{title}. "
            f"Category:{cats_joined}. "
            f"Price:{price}. "
            f"Store:{store}. "
            f"User Rating:{rating.get(uu, 'None')}\n"
        )
    return interaction_his


def call_model_with_backoff(client: OpenAI, prompt: str, model_name: str,
                            max_retries: int = 5, base_delay: float = 1.0):
    delay = base_delay
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(model=model_name, input=prompt)
            return resp.output_text
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def worker(chunk_lines, train, review, meta, model_name, out_dir: Path, progress_q: Queue):
    """
    각 항목 처리 후 progress_q.put(1) 호출로 메인 tqdm 업데이트.
    항목 단위 try/except로 실패해도 이후 항목 계속 처리.
    """
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    processed = []
    for raw in chunk_lines:
        key = raw.strip()
        try:
            his = train['History'][key] # 사용자 ID에 해당하는 아이템 목록 가져오기 
            rating = get_user_ratings(his, review, key) # 사용자 ID에 해당하는 아이템들의 평점 가져오기
            interaction_his = build_interaction_his(his, meta, rating) # 텍스트로 사용자와 아이템의 상호작용 이력 구축
            prompt = persona_prompts(interaction_his) # 구축된 상호작용 이력을 기반으로 모델에게 페르소나 생성 요청하는 프롬프트
            text = call_model_with_backoff(client, prompt, model_name)
            out_path = out_dir / f"{key}.txt"
            with out_path.open('w', encoding='utf-8') as f:
                f.write(text)
            processed.append(key)
        except Exception as e:
            # 항목 실패는 경고만 출력하고 계속
            print(f"[WARN] key={key} failed: {repr(e)}")
        finally:
            # 성공/실패와 무관하게 진행률 1칸 업데이트
            progress_q.put(1)
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CDs_and_Vinyl')
    parser.add_argument('--model_name', default='gpt-5-mini')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8, help='Parallel threads (<=4 recommended for rate limits)')
    parser.add_argument('--chunks', type=int, default=4, help='How many chunks to split lines into')
    args = parser.parse_args()

    random.seed(args.seed)

    base = f'./dataset/{args.dataset}'
    train = load_json(f'{base}/train.json')
    meta = load_json(f'{base}/meta.json')
    review = load_json(f'{base}/train_review.json')
    _ = load_json(f'{base}/user2id.json')
    _ = load_json(f'{base}/item2id.json')

    with open(f'{base}/user_sets.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines() # 페르소나를 생성할 사용자 ID 목록

    out_dir = Path(f'./persona_{args.dataset}_{args.model_name}') # 생성된 페르소나 텍스트 파일이 저장될 디렉토리
    out_dir.mkdir(parents=True, exist_ok=True)

    if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY']:
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

    total = len(lines)
    chunks = chunk_n(lines, max(1, args.chunks)) # 병렬 처리를 위해 사용자 ID 목록을 여러 청크로 분할
    max_workers = max(1, min(args.workers, len(chunks))) # 동시에 일할 워커 수 결정 (청크 수보다 많지 않도록)
 
    progress_q = Queue()

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for c in chunks:
            if not c:
                continue
            futures.append(ex.submit(worker, c, train, review, meta, args.model_name, out_dir, progress_q))
            # 메인 스레드는 데이터를 청크로 나눠서 여러 worker에게 처리하게 함 
            # 페르소나 생성 작업 수행 

        # ---- 항목 단위 tqdm 진행 표시 ----
        done = 0
        with tqdm(total=total, desc="Generating personas", unit="file") as pbar:
            while done < total:
                # 항목 하나 끝날 때마다 1이 들어옴
                progress_q.get()  # blocking
                done += 1
                pbar.update(1)

        # 워커 예외 처리(끝난 뒤 수집)
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[WARN] worker failed: {repr(e)}")

    # 요약 로그
    generated = sum(1 for _ in out_dir.glob("*.txt")) if total > 0 else 0
    print(f"[INFO] Requested: {total} | Generated files: {generated} | Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
