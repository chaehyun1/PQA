# Perception Module — Visual Caption Generation
# 논문 절차 (3단계):
#   1. Initial caption: LLM(P_caption, poster_image) → i*
#   2. Atomic claim decomposition: i* → {a_1, ..., a_m}
#   3. Claim scoring via MLLM: a_k → (p_yes, p_no) per claim  ← LLaVA logprobs
#   4. Refined caption: LLM(P_combine, i*, {(a_k, s_a)}) → i_caption
#
# 필요한 아이템만 생성 / 이미 생성된 것은 캐시(result/captions.json)에서 로드
# 지금 설정: OpenAI 최대 3회 + LLaVA 최대 5회

import os
import json
import base64
import argparse
import requests
import torch
import pandas as pd
from io import BytesIO
from typing import Optional
from PIL import Image
from openai import OpenAI
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "MovieLens")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")
CAPTION_CACHE = os.path.join(RESULT_DIR, "captions.json")
POSTER_DIR = os.path.join(RESULT_DIR, "posters")

DEFAULT_LLAVA_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"


# ── 필요한 movie_id 수집 ────────────────────────────────────────────

def collect_needed_movie_ids() -> set[int]:
    """test.txt에 있는 모든 아이템 ID 집합 반환."""
    needed = set()
    rec_path = os.path.join(DATA_DIR, "test.txt") # NOTE: 필요 시 수정 
    with open(rec_path) as f:
        for line in f:
            parts = line.strip().split()
            needed.update(int(x) for x in parts[1:])
    return needed


# ── 캐시 입출력 ────────────────────────────────────────────────────

def load_cache() -> dict[int, str]:
    os.makedirs(RESULT_DIR, exist_ok=True)
    if os.path.exists(CAPTION_CACHE):
        with open(CAPTION_CACHE, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_cache(cache: dict[int, str]) -> None:
    with open(CAPTION_CACHE, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cache.items()}, f, ensure_ascii=False, indent=2)


# ── movie_detail 로드 ───────────────────────────────────────────────

def load_movie_info() -> dict[int, dict]:
    df = pd.read_csv(os.path.join(DATA_DIR, "movie_detail.csv"))
    info = {}
    for _, row in df.iterrows():
        mid = int(row["movie_id"])
        info[mid] = {
            "title": row["title"],
            "genres": row["genres"],
            "poster_url": row.get("poster_url", ""),
        }
    return info


# ── LLaVA 모델 로드 ─────────────────────────────────────────────────

def load_llava(model_name: str = DEFAULT_LLAVA_MODEL, device: str = "auto"):
    """LLaVA 모델과 프로세서를 로드."""
    print(f"Loading LLaVA model: {model_name} (device={device})")
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()
    return processor, model


def _fetch_image(url: str) -> Optional[Image.Image]:
    """URL에서 이미지 다운로드. 실패 시 None 반환."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


# ── OpenAI 호출 헬퍼 ────────────────────────────────────────────────

def _llm(client, model, prompt_text: str, max_output_tokens: int = None) -> str:
    kwargs = {}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    resp = client.responses.create(model=model, input=prompt_text, **kwargs)
    return resp.output_text.strip()


def _read_image_b64(image_or_path) -> str:
    """PIL Image 또는 파일 경로에서 base64 문자열 반환."""
    if isinstance(image_or_path, str):
        with open(image_or_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    buf = BytesIO()
    image_or_path.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _mllm_with_image(client, model, image_url: str, prompt_text: str, max_output_tokens: int = None) -> str:
    """이미지 URL + 텍스트를 함께 전달하는 멀티모달 호출."""
    kwargs = {}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                    {"type": "input_text", "text": prompt_text},
                ],
            }
        ],
        **kwargs,
    )
    return resp.output_text.strip()


def _mllm_with_base64(client, model, image_b64: str, prompt_text: str, max_output_tokens: int = None) -> str:
    """base64 이미지 + 텍스트를 함께 전달하는 멀티모달 호출 (Chat Completions API 사용)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    )
    return resp.choices[0].message.content.strip()


# ── 캡션 생성 4단계 ─────────────────────────────────────────────────

def step1_initial_caption(client, model, title: str, genres: str, poster_url: str) -> str:
    """포스터 이미지로부터 초기 캡션 i* 생성."""
    prompt = (
        f"Movie: {title}\nGenres: {genres}\n\n"
        "Look at the movie poster above. Generate a concise caption (2-3 sentences) that captures:\n"
        "- Emotional tones conveyed by the visual\n"
        "- Key visual details (colors, characters, setting)\n"
        "- Unique selling points that would attract viewers\n"
        "Focus only on factual, observable elements. Avoid subjective opinions."
    )
    if poster_url and str(poster_url) not in ("nan", ""):
        return _mllm_with_image(client, model, poster_url, prompt, max_output_tokens=512)
    
    # poster가 없으면 텍스트만으로 생성
    fallback = (
        f"Movie: {title}\nGenres: {genres}\n\n"
        "Generate a concise caption (2-3 sentences) for this movie that captures "
        "emotional tones, thematic visual details, and unique selling points. "
        "Focus on specific, factual statements."
    )
    return _llm(client, model, fallback, max_output_tokens=512)


def step2_decompose_claims(client, model, caption: str) -> list[str]:
    """초기 캡션 i*를 원자적 사실 주장으로 분해."""
    prompt = (
        f"Caption:\n{caption}\n\n"
        "Decompose the caption above into at most 5 atomic claims. "
        "Each claim must be a single, specific, factual statement (e.g., 'The movie poster shows a dark forest.'). "
        "Do NOT include subjective opinions or uncertain statements. "
        "Only include claims you are confident about. "
        "Output each claim on its own line, no numbering, no bullet points."
    )
    text = _llm(client, model, prompt, max_output_tokens=512)
    claims = [l.strip() for l in text.split("\n") if l.strip()]
    return claims[:5]


def step3_score_claims(
    llava_processor, llava_model,
    claims: list[str], title: str, genres: str, image: Optional[Image.Image]
) -> list[tuple[str, float, float]]:
    """LLaVA logprobs로 각 claim의 (p_yes, p_no) 산출."""

    # Yes / No 토큰 ID 추출
    yes_ids = llava_processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = llava_processor.tokenizer.encode("No", add_special_tokens=False)
    yes_token_id = yes_ids[0]
    no_token_id = no_ids[0]

    device = next(llava_model.parameters()).device

    results = []
    for claim in claims:
        question = (
            f"Movie: {title}\nGenres: {genres}\n\n"
            f"Claim: {claim}\n\n"
            "Does this claim accurately describe this movie? Answer with only 'Yes' or 'No'."
        )

        if image is not None:
            # LLaVA conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            prompt_text = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = llava_processor(images=image, text=prompt_text, return_tensors="pt")
        else:
            # 이미지 없으면 텍스트만
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                }
            ]
            prompt_text = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = llava_processor(text=prompt_text, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = llava_model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # 첫 번째 생성 토큰의 확률 분포
        probs = torch.softmax(output.scores[0][0], dim=-1)
        p_yes = probs[yes_token_id].item()
        p_no = probs[no_token_id].item()

        total = p_yes + p_no
        if total > 0:
            p_yes, p_no = p_yes / total, p_no / total
        else:
            p_yes, p_no = 0.5, 0.5

        results.append((claim, round(p_yes, 3), round(p_no, 3)))

    return results


def step4_refine_caption(client, model, initial_caption: str,
                         scored_claims: list[tuple[str, float, float]]) -> str:
    """scored claim들을 바탕으로 최종 캡션 i_caption 생성."""
    claims_text = "\n".join(
        f"  - \"{claim}\" (p_yes={p_yes}, p_no={p_no})"
        for claim, p_yes, p_no in scored_claims
    )
    prompt = (
        f"Initial Caption:\n{initial_caption}\n\n"
        f"Scored Atomic Claims:\n{claims_text}\n\n"
        "Refine the initial caption using the scored claims above. "
        "Emphasize claims with high p_yes (accurate) and reduce or remove claims with high p_no (inaccurate). "
        "The refined caption should be 1-2 sentences, factual, and highlight emotional tones, "
        "visual details, and unique selling points. Output only the refined caption."
    )
    return _llm(client, model, prompt)


def generate_caption(client, model: str, llava_processor, llava_model, info: dict, mid: int = None) -> str:
    """단일 영화에 대한 캡션 생성 전체 파이프라인."""
    title = info["title"]
    genres = info["genres"]
    poster_url = str(info.get("poster_url", ""))
    if poster_url == "nan":
        poster_url = ""

    # 로컬 포스터 이미지 우선 사용
    image = None
    local_path = os.path.join(POSTER_DIR, f"{mid}.jpg") if mid else None
    if local_path and os.path.exists(local_path):
        image = Image.open(local_path).convert("RGB")
        print(f"    Using local poster: {local_path}")
    elif poster_url:
        print(f"    Fetching poster from URL: {poster_url}")
        image = _fetch_image(poster_url)

    if image is None:
        print(f"    → No poster image, skipping caption generation.")
        return "[No poster image]"

    _invalid_kw = ["[no caption]"] # NOTE: 필요 시 수정 

    prompt = (
        f"Movie: {title}\nGenres: {genres}\n\n"
        "Look at the movie poster above. Generate a concise caption (2-3 sentences) that captures:\n"
        "- Emotional tones conveyed by the visual\n"
        "- Key visual details (colors, characters, setting)\n"
        "- Unique selling points that would attract viewers\n"
        "Focus only on factual, observable elements. Avoid subjective opinions."
    )

    print(f"    [1/4] Initial caption (base64)...")
    b64 = _read_image_b64(local_path)
    i_star = _mllm_with_base64(client, model, b64, prompt, max_output_tokens=512)

    print(f"    [2/4] Decomposing claims...")
    claims = step2_decompose_claims(client, model, i_star)

    print(f"    [3/4] Scoring {len(claims)} claims with LLaVA...")
    scored = step3_score_claims(llava_processor, llava_model, claims, title, genres, image)

    print(f"    [4/4] Refining caption...")
    final = step4_refine_caption(client, model, i_star, scored)

    return final


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="step1/2/4에 사용할 OpenAI 모델 (vision 지원 필요)")
    parser.add_argument("--llava_model", type=str, default=DEFAULT_LLAVA_MODEL, help="step3 claim scoring에 사용할 LLaVA 모델")
    parser.add_argument("--device", type=str, default="cuda:7", help="LLaVA 모델 device 설정")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading movie info...")
    movie_info = load_movie_info()

    print("Collecting needed movie IDs from test.txt...")
    needed_ids = collect_needed_movie_ids()
    print(f"  → {len(needed_ids)} unique movies needed")

    print("Loading caption cache...")
    cache = load_cache()
    missing = needed_ids - set(cache.keys())

    # 캐시에 있지만 실패한 캡션(LLM 메타 응답)을 재생성 대상에 추가
    _invalid_patterns = ["[no caption]"] # NOTE: 필요 시 수정
    _skip_markers = ["[No poster image]"] # url 접속 시 not found인 경우는 포스터 이미지가 없기 때문에 skip
    invalid = {
        mid for mid in (needed_ids & set(cache.keys()))
        if any(p in cache[mid].lower() for p in _invalid_patterns)
        and cache[mid] not in _skip_markers
    }
    
    if invalid:
        print(f"  → {len(invalid)} cached but invalid captions detected, will regenerate")
    missing = missing | invalid

    print(f"  → {len(cache)} cached / {len(missing)} to generate")

    if not missing:
        print("All captions already cached.")
        return

    # 포스터 없는 아이템은 LLaVA 로드 전에 먼저 처리
    missing_with_poster = []
    for mid in sorted(missing):
        local_path = os.path.join(POSTER_DIR, f"{mid}.jpg")
        if not os.path.exists(local_path):
            info = movie_info.get(mid)
            title = info["title"] if info else f"id={mid}"
            print(f"  No poster for '{title}' (id={mid}) → [No poster image]")
            cache[mid] = "[No poster image]"
            save_cache(cache)
        else:
            missing_with_poster.append(mid)

    print(f"  → {len(missing) - len(missing_with_poster)} items marked as [No poster image]")
    print(f"  → {len(missing_with_poster)} items to generate captions for")

    if not missing_with_poster:
        print("No captions to generate.")
        return

    # LLaVA는 생성할 항목이 있을 때만 로드
    llava_processor, llava_model = load_llava(args.llava_model, args.device)

    for i, mid in enumerate(sorted(missing_with_poster)):
        info = movie_info.get(mid)
        if info is None:
            print(f"[{i+1}/{len(missing_with_poster)}] movie_id={mid} not found in movie_detail.csv, skipping.")
            continue

        print(f"[{i+1}/{len(missing_with_poster)}] Generating caption for '{info['title']}' (id={mid})...")
        try:
            caption = generate_caption(client, args.model, llava_processor, llava_model, info, mid=mid)
            if any(p in caption.lower() for p in _invalid_patterns):
                print(f"    WARNING: invalid caption, saving as [No caption] → {caption[:80]}...")
                caption = "[No caption]"
            cache[mid] = caption
            save_cache(cache)
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nDone. {len(cache)} captions saved to {CAPTION_CACHE}")


if __name__ == "__main__":
    main()
