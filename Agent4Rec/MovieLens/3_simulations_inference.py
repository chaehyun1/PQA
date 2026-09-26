# 추천 모델이 제시한 리스트에 대한 것임 not test GT

import json
import os
import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import re


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MovieLens')
parser.add_argument('--rec_domain', default='movie')
parser.add_argument('--model_name', default='gpt-4o-mini')
parser.add_argument('--cf_model_dir', default='sasrec')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_users', type=int, default=None, help='Maximum number of users to simulate. Default: all in user_sets.txt.')
parser.add_argument('--flip_personality', default='none',
                    choices=['none', 'all', 'activity', 'conformity', 'diversity'],
                    help='Flip personality values (1<->3, 2 stays). Specifies which trait(s) to flip.')
parser.add_argument('--rec_ratio', default='1to1',
                    choices=['1to1', '1to3', '1to9'],
                    help='good:bad ratio of recommendation list file to load.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of repeated full runs to account for LLM stochasticity. Each run saved to run_{i}/.')
parser.add_argument('--workers', type=int, default=8,
                    help='Number of parallel worker threads for user-level simulation.')
parser.add_argument('--top_k_path', default='./result/user_top_k_genres.json',
                    help='Per-user top-k preferred genres JSON (from 1_extract_top_k_genres.py).')
parser.add_argument('--baseline_path', default='./result/session_baseline_w10.json',
                    help='Per-user session baseline JSON (from 2_session_baseline.py).')


def call_api(client, model_name, prompt, max_retries=5, base_delay=1.0):
    """OpenAI API 호출 with exponential backoff retry."""
    delay = base_delay
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(model=model_name, input=prompt)
            return resp.output_text
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[API ERROR] all {max_retries} retries failed: {type(e).__name__}: {e}")
                raise
            print(f"[API RETRY {attempt+1}/{max_retries-1}] {type(e).__name__}: {e} — sleeping {delay:.1f}s")
            time.sleep(delay)
            delay *= 2


def compute_page_overlap(items, top_k_set, meta):
    """Average number of top-k genre matches per item across the page (range 0..k).

    Same formula as the session baseline (2_session_baseline.py) so the two
    values are directly comparable.
    """
    if not items or not top_k_set:
        return 0.0
    total = 0
    for iid in items:
        item_genres = {g for g in str(meta.loc[iid]['genres']).split('|') if g}
        total += len(item_genres & top_k_set)
    return total / len(items)


def classify_page_quality(ratio):
    """Deterministic page-vs-baseline ratio classification: ABOVE / NORMAL / BELOW."""
    if ratio is None:
        return "UNKNOWN"
    if ratio < 0.7:
        return "BELOW"
    if ratio < 1.0:
        return "NORMAL"
    return "ABOVE"


def recommendation_page_prompts(items, meta, page):  # 추천된 4개의 아이템(제목, 평균 평점, 장르, 줄거리 등)을 보여줌. 각 아이템 앞에 [ID:xxx] 명시.
    def fmt(item_id):
        row = meta.loc[item_id]
        return f"== [ID:{item_id}] {str(row['title']).strip()} | Avg Rating: {row['rating']} | Genres: {row['genres']} | Summary: {row['summary']}"
    return (
        f"=============    Recommendation Page {page}    =============\n"
        f"{fmt(items[0])}\n"
        f"{fmt(items[1])}\n"
        f"{fmt(items[2])}\n"
        f"{fmt(items[3])}\n"
        f"=============    End Page {page}    =============\n"
    )


def get_recommendation_response(persona, rating_tendency, personality, rec_page, rec_domain):  # 유저에게 성격과 취향 부여하여 추천 리스트에 대해 (구매 여부, 이유, 평점)을 작성하도록 함
    return f"""You excel at role-playing. Picture yourself as a user exploring a {rec_domain} recommendation system. You have the following social traits:\n\
Your activity trait is described as: {personality[0]}\n\
Your conformity trait is described as: {personality[1]}\n\
Your diversity trait is described as: {personality[2]}\n\
Beyond that, your {rec_domain} tastes are: {'; '.join(t.lstrip('I ') for t in persona)}.\n\
And your rating tendency is: {rating_tendency}\n\
The activity characteristic pertains to the frequency of your movie-watching habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste.\n\
\n\
#### Movie List #### \n
{rec_page}\n\
Please respond to all the items in the **Recommendation Page** and provide explanations.\n\
Firstly, determine which items align with your taste and which do not, and provide reasons. You must respond to all the recommended movies using this format (include the ID shown in the page):\n\
ID: [item id]; MOVIE: [movie name]; ALIGN: [yes or no]; REASON: [brief reason]\n\
Secondly, among the movies that align with your tastes, decide the number of movies you want to watch based on your activity and diversity traits. Use this format:\n\
NUM: [number of movie you choose to watch]; WATCH IDs: [comma-separated ids you choose to watch]; WATCH: [all movie name you choose to watch]; REASON: [brief reason];\n\
Thirdly, assume it's your first time watching the movies you've chosen, and rate them on a scale of 1-5 to reflect different degrees of liking, considering your feeling and conformity trait. Use this format:\n\
ID: [item id]; MOVIE: [movie you choose to watch]; RATING: [integer between 1-5]; FEELING: [aftermath sentence];\n\
Do not include any additional information or explanations and stay grounded."""


def next_page_behavior(personality, memory, rec_domain, page, top_k_str, baseline, page_overlap, ratio, page_quality):  # 페이지가 넘어갈수록 유저가 피곤함을 느끼고, 시스템을 나갈지 말지 결정함
    if baseline is None:
        preference_block = (
            f"Your top preferred {rec_domain} categories (derived from your viewing history) are: {top_k_str}.\n"
            f"(No historical baseline could be computed for you, so use this list qualitatively.)\n"
        )
    else:
        ratio_str = f"{ratio:.2f}x your typical level" if ratio is not None else "(undefined — historical baseline is zero)"
        preference_block = (
            f"Your top preferred {rec_domain} categories (derived from your viewing history) are: {top_k_str}.\n"
            f"In your typical past viewing, each {rec_domain} you watched contained on average {baseline:.2f} of these top categories. This is your historical baseline of preference adherence.\n"
            f"On Page {page}, the recommended {rec_domain}s contain on average {page_overlap:.2f} of your top categories — that is {ratio_str}.\n"
            f"Page quality (computed from ratio): {page_quality}  [ABOVE if ratio ≥ 1.0; NORMAL if 0.7 ≤ ratio < 1.0; BELOW if ratio < 0.7]\n"
        )
    return f"""You excel at role-playing. Picture yourself as a user exploring a {rec_domain} recommendation system.

You have the following social trait:
- Activity: {personality}

{preference_block}

Your activity trait shapes how you respond to varying page quality.
Low activity = more sensitive to BELOW pages.
High activity = more patient, may browse a few more pages despite BELOW.

You are now in Page {page}. You may get tired as pages accumulate (above 2 is a little tired, above 4 very tired).

Relevant memory:
{memory}

=== Step 1: Overall Feeling ===
Based primarily on the Page quality stated above, generate your overall feeling.
Your activity trait does not change your feeling about the page quality.

If positive: POSITIVE: [reason]
If negative: NEGATIVE: [reason]

=== Step 2: Exit Decision ===
Decide whether to continue browsing. Both **page quality** and your **activity trait** matter:
- If page quality is ABOVE: you continue regardless of activity.
- If page quality is NORMAL:
    * If this is your first NORMAL page so far: browse at least one more page.
    * Otherwise: your activity determines the decision.
- If page quality is BELOW:
    * If Page 1 is BELOW: browse at least one more page to see if the next is better.
    * If two consecutive pages are BELOW, exit regardless of activity — even tolerant users should leave at that point.
    * Otherwise, judge based on your activity and fatigue.

Check the memory above — if previous pages were also BELOW, your patience is reduced and exit becomes more likely.

To leave: [EXIT]; Reason: [brief reason]
To continue: [NEXT]; Reason: [brief reason]"""


def get_recommender_feedback(persona, personality, rating_tendency, page_memory, rec_domain, top_k_str, baseline, page_overlap_history):  # 시뮬레이션 종료 후 유저가 이용 경험 전체에 대해 만족도 점수를 매기도록 인터뷰 진행
    if baseline is None:
        preference_block = (
            f"Your top preferred {rec_domain} categories (derived from your viewing history) are: {top_k_str}.\n"
            f"(No historical baseline could be computed for you.)\n"
        )
    else:
        overlap_lines = []
        for p, ov, rt in page_overlap_history:
            if rt is None:
                overlap_lines.append(f"  Page {p}: {ov:.2f} matches per item")
            else:
                pq = classify_page_quality(rt)
                overlap_lines.append(f"  Page {p}: {ov:.2f} matches per item ({rt:.2f}x baseline, {pq})")
        overlap_block = "\n".join(overlap_lines) if overlap_lines else "  (no pages browsed)"
        preference_block = (
            f"Your top preferred {rec_domain} categories (derived from your viewing history) are: {top_k_str}.\n"
            f"Your historical baseline of category adherence is {baseline:.2f} (average number of top categories per item you watched in the past).\n"
            f"During this session, the recommended pages reflected your top categories at the following levels:\n"
            f"{overlap_block}\n"
            f"Quality labels: ABOVE if ratio ≥ 1.0, NORMAL if 0.7 ≤ ratio < 1.0, BELOW if ratio < 0.7.\n"
        )
    return f"""You excel at role-playing. Picture yourself as a user who has just finished exploring a {rec_domain} recommendation system.\n\
You have the following social traits:\n\
Your activity trait is described as: {personality[0]}\n\
Your conformity trait is described as: {personality[1]}\n\
Your diversity trait is described as: {personality[2]}\n\
Beyond that, your {rec_domain} tastes are: {'; '.join(t.lstrip('I ') for t in persona)}\n\
And your rating tendency is: {rating_tendency}\n\
The activity characteristic pertains to the frequency of your movie-watching habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste.\n\
\n\
{preference_block}\
\n\
Relevant context from user's memory:\n\
{page_memory}
Act as this user, assume you are having an interview, respond to the following question:

Do you feel satisfied with the recommender system you have just interacted with?
Rate the system from 1 to 10.

When rating, weight the following in order of importance:
1. **Watch satisfaction (most important)** — Of the {rec_domain}s you actually watched, did they fit your taste well? A few well-matched watches can make a session worthwhile.
2. **Quality of exposure** — Across the pages you browsed, did you encounter enough items aligned with your top categories? Use the overlap labels as a guide, not a verdict.
3. **Session fit for your activity level** — A low-activity user who browsed only 1-2 pages is acting normally; do NOT penalize the system just because the session was short. A high-activity user who left early because every page was BELOW is a more meaningful negative signal.

Rating anchors (be consistent with these):
- 9-10: Excellent. Most pages matched your taste, you watched multiple {rec_domain}s you liked.
- 7-8: Good. Some great finds, mostly aligned with your taste, satisfying watches.
- 5-6: Mixed. A few decent finds but also misses; reasonable but not exciting.
- 3-4: Disappointing. Few alignments, most items felt off-target.
- 1-2: Bad. Almost nothing aligned, no satisfying watches.

Important:
- Do NOT anchor your rating only on BELOW/ABOVE labels or page count.
- A short session with a couple of satisfying watches can still rate 7+.
- Many BELOW pages with no good finds rate 2-4.

Please use this respond format: RATING: [integer between 1 and 10]; REASON: [explanation]; In RATING part just give your rating and other reason and explanation should be included in the REASON part."""


if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.seed)

    flip_tag = '' if args.flip_personality == 'none' else f"_flip-{args.flip_personality}"
    sim_root = f"./simulation_{args.dataset}/{args.cf_model_dir}_{args.model_name}_{args.rec_ratio}{flip_tag}"
    os.makedirs(sim_root, exist_ok=True)
    os.makedirs(f"./simulation_{args.dataset}/results", exist_ok=True)

    # ── 데이터 로드 ─────────────────────────────────────────────────
    meta = pd.read_csv(f'./dataset/{args.dataset}/movie_detail.csv').set_index('movie_id')
    personality_df = pd.read_csv(f'./dataset/{args.dataset}/user_statistic.csv').set_index('user_id')
    persona_df = pd.read_csv(f'./dataset/{args.dataset}/all_personas_like_modify.csv')

    with open(f'./dataset/{args.dataset}/user_sets.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # SASRec 추천 결과 (MV_sasrec/recommendation_results_{ratio}.json)
    rec_path = f'./dataset/MV_sasrec/recommendation_results_{args.rec_ratio}.json'
    with open(rec_path, 'r', encoding='utf-8') as f:
        rec_results = json.load(f)

    # Per-user top-k preferred genres & session baseline
    with open(args.top_k_path, 'r', encoding='utf-8') as f:
        top_k_map = json.load(f)  # uid_str -> [genre, ...]
    with open(args.baseline_path, 'r', encoding='utf-8') as f:
        baseline_map = json.load(f)  # uid_str -> {"baseline": float|None, ...}

    # ── Personality flip (MV는 1-3 스케일: 1↔3, 2는 유지) ─────────
    def _flip(v):
        return {1: 3, 2: 2, 3: 1}[int(v)]

    if args.flip_personality != 'none':
        traits_to_flip = ['activity', 'conformity', 'diversity'] if args.flip_personality == 'all' else [args.flip_personality]
        for trait in traits_to_flip:
            personality_df[trait] = personality_df[trait].apply(_flip)
        print(f"[flip_personality={args.flip_personality}] flipped traits: {traits_to_flip}")

    activity_dict = {
        1: "An Incredibly Elusive Occasional Viewer, so seldom attracted by movie recommendations that it's almost a legendary event when you do watch a movie. Your movie-watching habits are extraordinarily infrequent. And you will exit the recommender system immediately even if you just feel little unsatisfied.",
        2: "An Occasional Viewer, seldom attracted by movie recommendations. Only curious about watching movies that strictly align the taste. The movie-watching habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.",
        3: "A Movie Enthusiast with an insatiable appetite for films, willing to watch nearly every movie recommended to you. Movies are a central part of your life, and movie recommendations are integral to your existence. You are tolerant of recommender system, which means you are not easy to exit recommender system even if you have some unsatisfied memory."
    }
    conformity_dict = {
        1: "A Dedicated Follower who gives ratings heavily relies on movie historical ratings, rarely expressing independent opinions. Usually give ratings that are same as historical ratings.",
        2: "A Balanced Evaluator who considers both historical ratings and personal preferences when giving ratings to movies. Sometimes give ratings that are different from historical rating.",
        3: "A Maverick Critic who completely ignores historical ratings and evaluates movies solely based on own taste. Usually give ratings that are a lot different from historical ratings."
    }
    diversity_dict = {
        1: "An Exceedingly Discerning Selective Viewer who watches movies with a level of selectivity that borders on exclusivity. The movie choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
        2: "A Niche Explorer who occasionally explores different genres and mostly sticks to preferred movie types.",
        3: "A Cinematic Trailblazer, a relentless seeker of the unique and the obscure in the world of movies. The movie choices are so diverse and avant-garde that they defy categorization."
    }

    rec_domain = args.rec_domain
    api_key = os.environ['OPENAI_API_KEY']
    model = OpenAI(api_key=api_key)

    run_metrics = []

    # ── 유저 한 명 시뮬레이션 ───────────────────────────────────────
    def simulate_user(key, sim_dir, run_idx):
        f = open(f'{sim_dir}/user_{key}', 'w')
        f.write(f'Simulating User: {key}\n\n')
        exit_ = 'No'
        page = 0
        memory = ''
        page_memory = ''
        uid_int = int(key)
        top_k_list = top_k_map.get(str(uid_int), [])
        top_k_set = set(top_k_list)
        top_k_str = ", ".join(top_k_list) if top_k_list else "(none)"
        baseline_entry = baseline_map.get(str(uid_int), {})
        baseline = baseline_entry.get("baseline")  # float or None
        user_record = {
            "user_id": uid_int,
            "flip_personality": args.flip_personality,
            "run_idx": run_idx,
            "personality_values": {
                "activity": int(personality_df.loc[uid_int, 'activity']),
                "conformity": int(personality_df.loc[uid_int, 'conformity']),
                "diversity": int(personality_df.loc[uid_int, 'diversity']),
            },
            "top_k_genres": top_k_list,
            "baseline": baseline,
            "pages": [],
            "exit_page": None,
            "satisfy": None,
        }
        local_page_view = 0
        local_item_view = 0
        page_overlap_history = []  # list of (page_idx, page_overlap, ratio)

        while exit_ == 'No':
            f.write(f'{"-"*10}\n')
            page += 1
            user_persona = persona_df.iloc[uid_int]
            taste = [t.strip() for t in user_persona['taste'].split('|')]
            high_rating = user_persona['high_rating']
            rating_tendency = f'High Rating Item Properties: {high_rating}'
            personal = [
                activity_dict[int(personality_df.loc[uid_int, 'activity'])],
                conformity_dict[int(personality_df.loc[uid_int, 'conformity'])],
                diversity_dict[int(personality_df.loc[uid_int, 'diversity'])],
            ]
            top_items = rec_results[key][0 + (page-1)*4:page*4]
            page_overlap = compute_page_overlap(top_items, top_k_set, meta)
            ratio = (page_overlap / baseline) if (baseline is not None and baseline > 0) else None
            page_quality = classify_page_quality(ratio)
            page_overlap_history.append((page, page_overlap, ratio))
            rec_page = recommendation_page_prompts(top_items, meta, page)
            prompts = get_recommendation_response(
                taste, rating_tendency, personal, rec_page, rec_domain,
            )

            f.write(f'{prompts}\n\n')

            response_text = call_api(model, args.model_name, prompts)
            f.write(f'{response_text}\n\n')
            response_text = response_text.replace('**', "")

            # ALIGN yes 추출
            align_yes_pattern = r"ID:\s*(\d+)\s*;\s*MOVIE:\s*(.*?)\s*;\s*ALIGN:\s*yes\s*;\s*REASON:\s*(.*?)\s*(?=\n|$)"
            align_yes_matches = re.findall(align_yes_pattern, response_text, re.IGNORECASE)
            # ALIGN no 추출
            align_no_pattern = r"ID:\s*(\d+)\s*;\s*MOVIE:\s*(.*?)\s*;\s*ALIGN:\s*no\s*;\s*REASON:\s*(.*?)\s*(?=\n|$)"
            align_no_matches = re.findall(align_no_pattern, response_text, re.IGNORECASE)

            # WATCH 단계 (페이지 단위 집합 결정)
            watch_pattern = r"NUM:\s*(\d+)\s*;\s*WATCH IDs:\s*(.*?)\s*;\s*WATCH:\s*(.*?)\s*;\s*REASON:\s*(.*?)\s*(?=\n|$)"
            watch_matches = re.findall(watch_pattern, response_text, re.IGNORECASE)
            watched_ids = set()
            watch_reason_agg = ''
            for _num, w_ids, _w_names, w_reason in watch_matches:
                for wid in re.findall(r"\d+", w_ids):
                    watched_ids.add(int(wid))
                watch_reason_agg = w_reason.strip()

            # RATING + FEELING 추출
            rating_feeling_pattern = r"ID:\s*(\d+)\s*;\s*MOVIE:\s*(.*?)\s*;\s*RATING:\s*(\d)\s*;\s*FEELING:\s*(.*?)\s*(?=\n|$)"
            rating_feeling_matches = re.findall(rating_feeling_pattern, response_text, re.IGNORECASE)

            # === 파싱 실패/이상 감지 로그 ===
            total_align = len(align_yes_matches) + len(align_no_matches)
            if total_align != 4:
                print(f"[PARSE WARN] user={key} page={page} ALIGN matches={total_align} (expected 4)")

            like_item = [[], []]
            dis_like_item = []

            recommendation_list = [str(meta.loc[t_i]['title']).strip() for t_i in top_items]
            recommendation_list = ', '.join(recommendation_list)
            for _iid, name, rating, _feeling in rating_feeling_matches:
                like_item[0].append(name.strip())
                like_item[1].append(rating)

            if len(like_item[0]) > 0:
                local_page_view += 1
            local_item_view += len(like_item[0])

            for _nid, nal_item, _nreason in align_no_matches:
                dis_like_item.append(nal_item.strip())

            memory += f"- Page {page} quality was {page_quality}. The recommender recommended the following {rec_domain}s to me on page {page}: {recommendation_list}, among them, I watched {like_item[0]} and rate them {like_item[1]} respectively. I dislike the rest items: {dis_like_item}.\n"
            page_memory += f"- Page {page} quality was {page_quality}. The recommender recommended the following {rec_domain}s to me on page {page}: {recommendation_list}, among them, I watched {like_item[0]} and rate them {like_item[1]} respectively. I dislike the rest items: {dis_like_item}.\n"
            next_page_prompt = next_page_behavior(
                activity_dict[int(personality_df.loc[uid_int, 'activity'])],
                memory, rec_domain, page,
                top_k_str, baseline, page_overlap, ratio, page_quality,
            )
            f.write(f'{next_page_prompt}\n\n')

            response_text = call_api(model, args.model_name, next_page_prompt)
            f.write(f'{response_text}\n\n')
            response_text = response_text.replace('**', "")

            if '[EXIT]' in response_text:
                feeling = response_text.split('[EXIT]')[0][:-1].strip().replace('\n', ' ')
            else:
                feeling = response_text.split('[NEXT]')[0][:-1].strip().replace('\n', ' ')
            page_memory += f"- My feelings on page {page}: {feeling}\n"

            exited_now = '[EXIT]' in response_text or page == 5
            if exited_now:
                exit_ = 'YES'
                page_memory += f"- After browsing {page} pages, I decided to leave the recommendation system.\n"
            else:
                page_memory += f"- Turn to page {page+1} of the recommendation.\n"

            # === per-page 구조화 저장 ===
            item_decisions = {int(iid): {
                "item_id": int(iid),
                "title": str(meta.loc[iid]['title']).strip(),
                "align": None,
                "align_reason": None,
                "watched": False,
                "rating": None,
                "feeling": None,
            } for iid in top_items}
            for iid_str, _name, reason in align_yes_matches:
                iid = int(iid_str)
                if iid in item_decisions:
                    item_decisions[iid]["align"] = "yes"
                    item_decisions[iid]["align_reason"] = reason.strip()
            for iid_str, _name, reason in align_no_matches:
                iid = int(iid_str)
                if iid in item_decisions:
                    item_decisions[iid]["align"] = "no"
                    item_decisions[iid]["align_reason"] = reason.strip()
            for iid_str, _name, rating_v, feeling in rating_feeling_matches:
                iid = int(iid_str)
                if iid in item_decisions:
                    item_decisions[iid]["watched"] = True
                    item_decisions[iid]["rating"] = int(rating_v)
                    item_decisions[iid]["feeling"] = feeling.strip()
            for iid in watched_ids:
                if iid in item_decisions:
                    item_decisions[iid]["watched"] = True

            user_record["pages"].append({
                "page": page,
                "top_item_ids": [int(i) for i in top_items],
                "items": [item_decisions[int(i)] for i in top_items],
                "watch_reason": watch_reason_agg,
                "page_feeling": feeling,
                "page_overlap": round(page_overlap, 6),
                "ratio_vs_baseline": round(ratio, 6) if ratio is not None else None,
                "page_quality": page_quality,
                "exited": exited_now,
            })

        interview_prompt = get_recommender_feedback(
            taste, personal, rating_tendency, page_memory, rec_domain,
            top_k_str, baseline, page_overlap_history,
        )
        f.write(f'{interview_prompt}\n\n')
        response_text = call_api(model, args.model_name, interview_prompt)
        f.write(f'{response_text}\n\n')
        f.close()
        response_text = response_text.replace('**', "")

        rating_pattern = r"RATING:\s*(\d+)"
        rating_match = re.search(rating_pattern, response_text, re.IGNORECASE)
        if rating_match is None:
            print(f"[PARSE WARN] user={key} interview RATING not matched; defaulting to 0")
        rating = rating_match.group(1) if rating_match else 0
        rating = int(rating)
        user_record["exit_page"] = page
        user_record["satisfy"] = rating
        with open(f'{sim_dir}/user_{key}.json', 'w', encoding='utf-8') as jf:
            json.dump(user_record, jf, ensure_ascii=False, indent=2)
        return {
            "key": key,
            "satisfy": rating,
            "page": page,
            "page_view": local_page_view,
            "item_view": local_item_view,
        }

    # ── Multi-run 루프 ──────────────────────────────────────────────
    for run_idx in range(args.num_runs):
        sim_dir = f"{sim_root}/run_{run_idx}" if args.num_runs > 1 else sim_root
        os.makedirs(sim_dir, exist_ok=True)
        print(f"\n===== Run {run_idx+1}/{args.num_runs} — saving to {sim_dir} =====")

        # 처리할 유저 키 선별
        keys_to_run = []
        for line in lines:
            key = line.strip()
            if key not in rec_results:
                continue
            if not len(rec_results[key]) >= 20:
                continue
            keys_to_run.append(key)
            if args.num_users is not None and len(keys_to_run) >= args.num_users:
                break

        page_view = 0
        each_item_view = 0
        satisfy = 0
        total_page = 0
        num_users = 0
        print_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_key = {ex.submit(simulate_user, k, sim_dir, run_idx): k for k in keys_to_run}
            for fut in tqdm(as_completed(future_to_key), total=len(future_to_key)):
                k = future_to_key[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"[ERROR] user={k} failed: {e}")
                    continue
                num_users += 1
                satisfy += res["satisfy"]
                total_page += res["page"]
                page_view += res["page_view"]
                each_item_view += res["item_view"]
                with print_lock:
                    if total_page > 0:
                        print(f"Satisfy: {satisfy/num_users:.4f} | Total Page: {total_page/num_users:.4f} | Page View: {page_view/total_page:.4f} | Each Item View: {each_item_view/(4*total_page):.4f}")

        run_metrics.append({
            "run_idx": run_idx,
            "num_users": num_users,
            "satisfy_avg": satisfy / num_users if num_users else 0,
            "total_page_avg": total_page / num_users if num_users else 0,
            "page_view_ratio": page_view / total_page if total_page else 0,
            "each_item_view_ratio": each_item_view / (4 * total_page) if total_page else 0,
        })

    # ── 전체 run 집계 ──────────────────────────────────────────────
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0
    def _std(xs):
        if len(xs) < 2:
            return 0
        m = _mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    keys = ["satisfy_avg", "total_page_avg", "page_view_ratio", "each_item_view_ratio"]
    summary_path = f'./simulation_{args.dataset}/results/{args.cf_model_dir}_{args.model_name}_{args.rec_ratio}{flip_tag}.txt'
    with open(summary_path, 'w') as fs:
        fs.write(f"num_runs: {args.num_runs} | num_users per run: {run_metrics[0]['num_users'] if run_metrics else 0}\n\n")
        fs.write("Per-run metrics:\n")
        for m in run_metrics:
            fs.write(f"  run {m['run_idx']}: "
                     f"Satisfy={m['satisfy_avg']:.4f} | "
                     f"Total Page={m['total_page_avg']:.4f} | "
                     f"Page View={m['page_view_ratio']:.4f} | "
                     f"Each Item View={m['each_item_view_ratio']:.4f}\n")
        fs.write("\nAggregate (mean ± std):\n")
        for k in keys:
            vals = [m[k] for m in run_metrics]
            fs.write(f"  {k}: {_mean(vals):.4f} ± {_std(vals):.4f}\n")
    print(f"\nSaved summary to {summary_path}")
