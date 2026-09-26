# 원본 CDs Agent4Rec (simulations_inference_personality_parallel.py)에
# top_k_categories + session_baseline 기반 page-quality calibration 메소드를 얹음.
# 주의: 이 파일은 선택 시 preference 관련 정보 제공 X

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
parser.add_argument('--dataset', default='CDs_and_Vinyl')
parser.add_argument('--rec_domain', default='CDs')
parser.add_argument('--model_name', default='gpt-4o-mini')
parser.add_argument('--cf_model_dir', default='sasrec')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_users', type=int, default=None, help='Maximum number of users to simulate. Default: all.')
parser.add_argument('--flip_personality', default='none',
                    choices=['none', 'all', 'activity', 'conformity', 'diversity'],
                    help='Flip personality values (0<->2, 1 stays). Specifies which trait(s) to flip.')
parser.add_argument('--rec_ratio', default='1to1',
                    choices=['1to1', '1to3', '1to9'],
                    help='good:bad ratio of recommendation list file to load.')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of repeated full runs to account for LLM stochasticity. Each run saved to run_{i}/.')
parser.add_argument('--workers', type=int, default=8,
                    help='Number of parallel worker threads for user-level simulation.')
parser.add_argument('--user_set', default='user_sets.txt',
                    help='User set filename under ./dataset/{dataset}/.')
parser.add_argument('--top_k_path', default='./result/user_top_k_categories.json',
                    help='Per-user top-k preferred categories JSON (from 1_extract_top_k_genres.py).')
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
    """Average number of top-k category matches per item across the page (range 0..k).

    Same formula as the session baseline (2_session_baseline.py) so the two
    values are directly comparable.
    """
    if not items or not top_k_set:
        return 0.0
    total = 0
    for iid in items:
        item_cats = {c for c in (meta.get(str(iid), {}).get('categories', []) or []) if c}
        total += len(item_cats & top_k_set)
    return total / len(items)


def classify_page_quality(ratio):
    """Deterministic page-vs-baseline ratio classification: ABOVE / NORMAL / BELOW.
    Threshold (0.3, 0.5): CDs sparse data 특성상 baseline이 높아 ratio가 낮게 나옴.
    원래 (0.7, 1.0)은 83%가 BELOW로 한쪽 쏠림 → (0.3, 0.5)로 balance."""
    if ratio is None:
        return "UNKNOWN"
    if ratio < 0.3:
        return "BELOW"
    if ratio < 0.5:
        return "NORMAL"
    return "ABOVE"


def recommendation_page_prompts(items, meta, page): # 추천된 4개의 아이템(제목, 평균 평점, 카테고리 등)을 보여줌. 각 아이템 앞에 [ID:xxx] 명시.
    return f"""=============    Recommendation Page {page}    =============\n\
== [ID:{items[0]}] {meta[str(items[0])]['title'].strip()} | History ratings: {meta[str(items[0])]['average_rating']} | Summary: categories-{','.join(meta[str(items[0])]['categories'])}/store-{meta[str(items[0])]['store']}/price-{meta[str(items[0])]['price']}\n\
== [ID:{items[1]}] {meta[str(items[1])]['title'].strip()} | History ratings: {meta[str(items[1])]['average_rating']} | Summary: categories-{','.join(meta[str(items[1])]['categories'])}/store-{meta[str(items[1])]['store']}/price-{meta[str(items[1])]['price']}\n\
== [ID:{items[2]}] {meta[str(items[2])]['title'].strip()} | History ratings: {meta[str(items[2])]['average_rating']} | Summary: categories-{','.join(meta[str(items[2])]['categories'])}/store-{meta[str(items[2])]['store']}/price-{meta[str(items[2])]['price']}\n\
== [ID:{items[3]}] {meta[str(items[3])]['title'].strip()} | History ratings: {meta[str(items[3])]['average_rating']} | Summary: categories-{','.join(meta[str(items[3])]['categories'])}/store-{meta[str(items[3])]['store']}/price-{meta[str(items[3])]['price']}\n\
=============    End Page {page}    =============\n"""


def get_recommendation_response(persona, rating_tendency, personality, rec_page, rec_domain): # 유저에게 성격과 취향 부여하여 추천 리스트에 대해 (구매 여부, 이유, 평점)을 작성하도록 함 — 원래 Agent4Rec 버전 (preference_block 없음)
    return f"""You excel at role-playing. Picture yourself as a user exploring a {rec_domain} recommendation system. You have the following social traits:\n\
Your activity trait is described as:{personality[0]}\n\
Your conformity trait is described as: {personality[1]}\n\
Your diversity trait is described as: {personality[2]}\n\
Beyond that, your {rec_domain} tastes are: {persona}\n\
And your rating tendency is: {rating_tendency}\n\
The activity characteristic pertains to the frequency of your item-purchase habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of purchasing items that may not align with your usual taste.\n\
\n\
{rec_page}\n\
Please respond to all the items in the **Recommendation Page** and provide explanations.
Firstly, determine which items align with your taste and which do not, and provide reasons. You must respond to all the recommended items using this format (include the ID shown in the page):
ID: [item id]; Item: [item name]; ALIGN: [yes or no]; REASON: [brief reason]
Secondly, among the items that align with your tastes, decide the number of items you want to purchase based on your activity and diversity traits. Use this format:
NUM: [number of {rec_domain} you choose to purchase]; Purchase IDs: [comma-separated ids you choose to purchase]; Purchase: [all item name you choose to purchase]; REASON: [brief reason];
Thirdly, assume it's your first time purchasing the {rec_domain} you've chosen, and rate them on a scale of 1-5 to reflect different degrees of liking, considering your feeling and conformity trait. Use this format:
ID: [item id]; Item: [item you choose to purchase]; RATING: [integer between 1-5]; FEELING: [aftermath sentence];
Do not include any additional information or explanations and stay grounded."""


def next_page_behavior(personality, memory, rec_domain, page, top_k_str, baseline, page_overlap, ratio, page_quality): # 페이지가 넘어갈수록 유저가 피곤함을 느끼고, 시스템을 나갈지 말지 결정함
    if baseline is None:
        preference_block = (
            f"Your top preferred {rec_domain} categories (derived from your purchase history) are: {top_k_str}.\n"
            f"(No historical baseline could be computed for you, so use this list qualitatively.)\n"
        )
    else:
        ratio_str = f"{ratio:.2f}x your typical level" if ratio is not None else "(undefined — historical baseline is zero)"
        preference_block = (
            f"Your top preferred {rec_domain} categories (derived from your purchase history) are: {top_k_str}.\n"
            f"In your typical past purchases, each {rec_domain} you bought contained on average {baseline:.2f} of these top categories. This is your historical baseline of preference adherence.\n"
            f"On Page {page}, the recommended {rec_domain} contain on average {page_overlap:.2f} of your top categories — that is {ratio_str}.\n"
            f"Page quality (computed from ratio): {page_quality}  [ABOVE if ratio ≥ 0.5; NORMAL if 0.3 ≤ ratio < 0.5; BELOW if ratio < 0.3]\n"
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


def get_recommender_feedback(persona, personality, rating_tendency, page_memory, rec_domain, top_k_str, baseline, page_overlap_history): # 시뮬레이션 종료 후 유저가 이용 경험 전체에 대해 만족도 점수를 매기도록 인터뷰 진행
    if baseline is None:
        preference_block = (
            f"Your top preferred {rec_domain} categories (derived from your purchase history) are: {top_k_str}.\n"
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
            f"Your top preferred {rec_domain} categories (derived from your purchase history) are: {top_k_str}.\n"
            f"Your historical baseline of category adherence is {baseline:.2f} (average number of top categories per item you bought in the past).\n"
            f"During this session, the recommended pages reflected your top categories at the following levels:\n"
            f"{overlap_block}\n"
            f"Quality labels: ABOVE if ratio ≥ 0.5, NORMAL if 0.3 ≤ ratio < 0.5, BELOW if ratio < 0.3.\n"
        )
    return f"""You excel at role-playing. Picture yourself as a user who has just finished exploring a {rec_domain} recommendation system.\n\
You have the following social traits:\n\
Your activity trait is described as: {personality[0]}\n\
Your conformity trait is described as: {personality[1]}\n\
Your diversity trait is described as: {personality[2]}\n\
Beyond that, your {rec_domain} tastes are: {persona}\n\
And your rating tendency is: {rating_tendency}\n\
The activity characteristic pertains to the frequency of your item-purchase habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of purchasing items that may not align with your usual taste.\n\
\n\
{preference_block}\
\n\
Relevant context from user's memory:\n\
{page_memory}
Act as this user, assume you are having an interview, respond to the following question:

Do you feel satisfied with the recommender system you have just interacted with?
Rate the system from 1 to 10.

When rating, weight the following in order of importance:
1. **Purchase satisfaction (most important)** — Of the items you actually purchased, did they fit your taste well? A few well-matched purchases can make a session worthwhile.
2. **Quality of exposure** — Across the pages you browsed, did you encounter enough items aligned with your top categories? Use the overlap labels as a guide, not a verdict.
3. **Session fit for your activity level** — A low-activity user who browsed only 1-2 pages is acting normally; do NOT penalize the system just because the session was short. A high-activity user who left early because every page was BELOW is a more meaningful negative signal.

Rating anchors (be consistent with these):
- 9-10: Excellent. Most pages matched your taste, you purchased multiple items you liked.
- 7-8: Good. Some great finds, mostly aligned with your taste, satisfying purchases.
- 5-6: Mixed. A few decent finds but also misses; reasonable but not exciting.
- 3-4: Disappointing. Few alignments, most items felt off-target.
- 1-2: Bad. Almost nothing aligned, no satisfying purchases.

Important:
- Do NOT anchor your rating only on BELOW/ABOVE labels or page count.
- A short session with a couple of satisfying purchases can still rate 7+.
- Many BELOW pages with no good finds rate 2-4.

Please use this respond format: RATING: [integer between 1 and 10]; REASON: [explanation]; In RATING part just give your rating and other reason and explanation should be included in the REASON part."""


if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.seed)

    flip_tag = '' if args.flip_personality == 'none' else f"_flip-{args.flip_personality}"
    user_set_tag = '' if args.user_set == 'user_sets.txt' else f"_{os.path.splitext(args.user_set)[0].replace('user_sets_', '')}"
    sim_root = f"./simulation_{args.dataset}/{args.cf_model_dir}_{args.model_name}_{args.rec_ratio}{user_set_tag}{flip_tag}"
    if not os.path.isdir(sim_root):
        os.makedirs(sim_root)
    if not os.path.isdir(f"./simulation_{args.dataset}/results"):
        os.makedirs(f"./simulation_{args.dataset}/results")

    with open(f'./dataset/{args.dataset}/train.json', 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(f'./dataset/{args.dataset}/meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(f'./dataset/{args.dataset}/review.json', 'r', encoding='utf-8') as f:
        review = json.load(f)
    with open(f'./dataset/{args.dataset}/user2id.json', 'r', encoding='utf-8') as f:
        user2id = json.load(f)
    with open(f'./dataset/{args.dataset}/item2id.json', 'r', encoding='utf-8') as f:
        item2id = json.load(f)
    with open(f'./dataset/{args.dataset}/{args.user_set}', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(f'./persona_{args.dataset}_{args.model_name}/personality.json', 'r', encoding='utf-8') as f:
        personality = json.load(f)

    # --- personality flip (0 <-> 2, 1 stays) ---
    def _flip(v):
        return {0: 2, 1: 1, 2: 0}[int(v)]

    if args.flip_personality != 'none':
        traits_to_flip = ['activity', 'conformity', 'diversity'] if args.flip_personality == 'all' else [args.flip_personality]
        for trait in traits_to_flip:
            personality[trait] = {u: _flip(v) for u, v in personality[trait].items()}
        print(f"[flip_personality={args.flip_personality}] flipped traits: {traits_to_flip}")
    persona = pd.read_csv(f'./persona_{args.dataset}_{args.model_name}/all_persona.csv', sep=',')

    with open(f'./dataset/{args.dataset}_{args.cf_model_dir}/recommendation_results_{args.rec_ratio}.json', 'r', encoding='utf-8') as f:
        rec_results = json.load(f)

    # Per-user top-k preferred categories & session baseline
    with open(args.top_k_path, 'r', encoding='utf-8') as f:
        top_k_map = json.load(f)  # uid_str -> [category, ...]
    with open(args.baseline_path, 'r', encoding='utf-8') as f:
        baseline_map = json.load(f)  # uid_str -> {"baseline": float|None, ...}

    activity_dict = {
        0: "An Incredibly Elusive Occasional Listener, so seldom attracted by CD and vinyl recommendations that it's a legendary event when you actually listen to a record. Your listening habits are extraordinarily infrequent, and you will exit the recommender system immediately if you feel even slightly unsatisfied.",
        1: "An Occasional Listener, seldom attracted by CD and vinyl recommendations. You are only curious about albums that strictly align with your taste. Your listening habits are not very frequent, and you tend to exit the recommender system if you have a few unsatisfying listening experiences.",
        2: "A Music Enthusiast with an insatiable appetite for CDs and vinyl, willing to listen to nearly every album recommended to you. Music is a central part of your life, and album recommendations are integral to your experience. You are tolerant of the recommender system and are not likely to leave even with some unsatisfying recommendations."
    }

    conformity_dict = {
        0: "A Dedicated Follower who heavily relies on an album's historical ratings and popular reviews, rarely expressing independent opinions. You usually give ratings that are the same as the general consensus.",
        1: "A Balanced Evaluator who considers both popular reviews and personal preferences when rating albums. You sometimes give ratings that are different from the historical average.",
        2: "A Maverick Critic who completely ignores popular opinion and evaluates albums solely based on your own taste. You usually give ratings that are significantly different from an album's historical ratings."
    }

    diversity_dict = {
        0: "An Exceedingly Discerning Selective Listener who chooses CDs and vinyl with a level of selectivity that borders on exclusivity. Your album choices are meticulously curated to match your personal taste, leaving almost no room for genre diversity.",
        1: "A Niche Explorer who occasionally explores different music genres but mostly sticks to your preferred styles and artists.",
        2: "A Sonic Trailblazer, a relentless seeker of the unique and the obscure in the world of music. Your CD and vinyl collection is so diverse and avant-garde that it defies easy categorization."
    }

    rec_domain = args.rec_domain
    api_key = os.environ['OPENAI_API_KEY']
    model = OpenAI(api_key=api_key)

    run_metrics = []  # [(satisfy_avg, total_page_avg, page_view_ratio, each_item_view_ratio), ...]

    # 유저 한 명 시뮬레이션
    def simulate_user(key, sim_dir, run_idx):
        f = open(f'{sim_dir}/user_{key}', 'w')
        f.write(f'Simulating User: {key}\n\n')
        exit_ = 'No'
        page = 0
        memory = ''
        page_memory = ''
        # Per-user method state
        top_k_list = top_k_map.get(str(key), [])
        top_k_set = set(top_k_list)
        top_k_str = ", ".join(top_k_list) if top_k_list else "(none)"
        baseline_entry = baseline_map.get(str(key), {})
        baseline = baseline_entry.get("baseline")  # float or None
        user_record = {
            "user_id": key,
            "flip_personality": args.flip_personality,
            "run_idx": run_idx,
            "personality_values": {
                "activity": personality['activity'][key],
                "conformity": personality['conformity'][key],
                "diversity": personality['diversity'][key],
            },
            "top_k_categories": top_k_list,
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
            page +=1
            user_persona = persona[persona['user_id'] == int(key)]
            taste = user_persona['taste'].iloc[0]
            high_rating = user_persona['high_rating'].iloc[0]
            rating_tendency = f'High Rating Item Properties: {high_rating}'
            personal = [activity_dict[personality['activity'][key]], conformity_dict[personality['conformity'][key]], diversity_dict[personality['diversity'][key]]]
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

            # 'ALIGN: yes' 항목에서 ID, Item, REASON 추출
            align_yes_pattern = r"ID:\s*(\d+)\s*;\s*Item:\s*(.*?)\s*;\s*ALIGN:\s*yes\s*;\s*REASON:\s*(.*?)\s*(?=\n|$)"
            align_yes_matches = re.findall(align_yes_pattern, response_text, re.IGNORECASE)

            # 'ALIGN: no' 항목에서 ID, Item, REASON 추출
            align_no_pattern = r"ID:\s*(\d+)\s*;\s*Item:\s*(.*?)\s*;\s*ALIGN:\s*no\s*;\s*REASON:\s*(.*?)\s*(?=\n|$)"
            align_no_matches = re.findall(align_no_pattern, response_text, re.IGNORECASE)

            # PURCHASE 단계 (페이지 단위 집합 결정)
            purchase_pattern = r"NUM:\s*(\d+)\s*;\s*Purchase IDs:\s*(.*?)\s*;\s*Purchase:\s*(.*?)\s*;\s*REASON:\s*(.*?)\s*(?=\n|$)"
            purchase_matches = re.findall(purchase_pattern, response_text, re.IGNORECASE)
            purchased_ids = set()
            purchase_reason_agg = ''
            for _num, pur_ids, _pur_names, pur_reason in purchase_matches:
                for pid in re.findall(r"\d+", pur_ids):
                    purchased_ids.add(int(pid))
                purchase_reason_agg = pur_reason.strip()

            # 유저가 실제로 구매/선택해서 남긴 RATING과 FEELING 추출 (ID 포함)
            rating_feeling_pattern = r"ID:\s*(\d+)\s*;\s*Item:\s*(.*?)\s*;\s*RATING:\s*(\d)\s*;\s*FEELING:\s*(.*?)\s*(?=\n|$)"
            rating_feeling_matches = re.findall(rating_feeling_pattern, response_text, re.IGNORECASE)

            # === 파싱 실패/이상 감지 로그 ===
            total_align = len(align_yes_matches) + len(align_no_matches)
            if total_align != 4:
                print(f"[PARSE WARN] user={key} page={page} ALIGN matches={total_align} (expected 4)")

            like_item = [[],[]] # 유저가 구매/선택한 아이템 이름과 평점
            dis_like_item = [] # 유저가 ALIGN: no라고 응답한 아이템 이름

            recommendation_list = [meta[str(t_i)]['title'].strip() for t_i in top_items]
            recommendation_list = ', '.join(recommendation_list)
            for _iid, name, rating, _feeling in rating_feeling_matches:
                like_item[0].append(name.strip())
                like_item[1].append(rating)

            if len(like_item[0]) >0:
                local_page_view +=1
            local_item_view += len(like_item[0])

            for _nid, nal_item, _nreason in align_no_matches:
                dis_like_item.append(nal_item.strip())

            memory += f"- Page {page} quality was {page_quality}. The recommender recommended the following {rec_domain} to me on page {page}: {recommendation_list}, among them, I purchased {like_item[0]} and rate them {like_item[1]} respectively. I dislike the rest items: {dis_like_item}.\n"
            page_memory += f"- Page {page} quality was {page_quality}. The recommender recommended the following {rec_domain} to me on page {page}: {recommendation_list}, among them, I purchased {like_item[0]} and rate them {like_item[1]} respectively. I dislike the rest items: {dis_like_item}.\n"
            next_page_prompt = next_page_behavior(
                activity_dict[personality['activity'][key]], memory, rec_domain, page,
                top_k_str, baseline, page_overlap, ratio, page_quality,
            )
            f.write(f'{next_page_prompt}\n\n')

            response_text = call_api(model, args.model_name, next_page_prompt)
            f.write(f'{response_text}\n\n')
            response_text = response_text.replace('**', "")

            if '[EXIT]' in response_text:
                fealing = response_text.split('[EXIT]')[0][:-1].strip().replace('\n',' ')
            else:
                fealing = response_text.split('[NEXT]')[0][:-1].strip().replace('\n',' ')
            page_memory += f"- My feelings on page {page}: {fealing}\n"

            exited_now = '[EXIT]' in response_text or page == 5
            if exited_now:
                exit_ = 'YES'
                page_memory += f"- After browsing {page} pages, I decided to leave the recommendation system.\n"
            else:
                page_memory += f"- Turn to page {page+1} of the recommendation.\n"

            # === per-page 구조화 저장 ===
            item_decisions = {int(iid): {
                "item_id": int(iid),
                "title": meta[str(iid)]['title'].strip(),
                "align": None,
                "align_reason": None,
                "purchased": False,
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
                    item_decisions[iid]["purchased"] = True
                    item_decisions[iid]["rating"] = int(rating_v)
                    item_decisions[iid]["feeling"] = feeling.strip()
            for iid in purchased_ids:
                if iid in item_decisions:
                    item_decisions[iid]["purchased"] = True

            user_record["pages"].append({
                "page": page,
                "top_item_ids": [int(i) for i in top_items],
                "items": [item_decisions[int(i)] for i in top_items],
                "purchase_reason": purchase_reason_agg,
                "page_feeling": fealing,
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

    for run_idx in range(args.num_runs): # 같은 task를 반복적으로 돌려서 실험함
        sim_dir = f"{sim_root}/run_{run_idx}" if args.num_runs > 1 else sim_root
        if not os.path.isdir(sim_dir):
            os.makedirs(sim_dir)
        print(f"\n===== Run {run_idx+1}/{args.num_runs} — saving to {sim_dir} =====")

        # 처리할 유저 키 미리 선별 (rec_results 길이 조건 + num_users 상한)
        keys_to_run = []
        for line in lines:
            key = line.strip()
            if not len(rec_results[key]) > 16:
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

        # run 종료: run별 지표 기록
        run_metrics.append({
            "run_idx": run_idx,
            "num_users": num_users,
            "satisfy_avg": satisfy / num_users if num_users else 0,
            "total_page_avg": total_page / num_users if num_users else 0,
            "page_view_ratio": page_view / total_page if total_page else 0,
            "each_item_view_ratio": each_item_view / (4 * total_page) if total_page else 0,
        })

    # 전체 run 집계 (mean / std)
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0
    def _std(xs):
        if len(xs) < 2:
            return 0
        m = _mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    keys = ["satisfy_avg", "total_page_avg", "page_view_ratio", "each_item_view_ratio"]
    summary_path = f'./simulation_{args.dataset}/results/{args.cf_model_dir}_{args.model_name}_{args.rec_ratio}{user_set_tag}{flip_tag}.txt'
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
