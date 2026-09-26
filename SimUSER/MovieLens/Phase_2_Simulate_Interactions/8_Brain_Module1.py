# Brain Module (MovieLens 버전, baseline: flip 없음)
# Chain-of-Thought 5단계:
# 1. Multi-round Preference Elicitation: [WATCH] or [SKIP]
#    - persona, pickiness, episodic/KG 메모리 기반 초기 결정
#    - 충돌/증거 부족 시 k1, k2 확장하며 재평가 (max 3 rounds)
# 2. Item Evaluation: explicit rating (1-5) + subjective feelings
#    - KG 경로(u → ... → item) 참조
# 3. Action Selection: [EXIT] / [NEXT] / [CLICK]
#    - 추천에 대한 만족도, 피로도, 감정 상태 고려
#    - [EXIT]인 경우 interview 진행
# 4. Causal Action Refinement: 반사실적 추론으로 최종 액션 확정
# 5. Post-interaction Reflection: → episodic memory에 추가

import os
import re
import json
import pickle
import argparse
import threading
import importlib.util
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

ITEMS_PER_PAGE = 4
MAX_PAGES = 20
K1_INIT = 5       # episodic memory 초기 retrieval 수
K2_INIT = 3       # KG memory 초기 retrieval 수
DELTA_K = 2       # 라운드당 증가량
MAX_ROUNDS = 3    # multi-round elicitation 최대 반복 수

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "MovieLens")
SASREC_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "MV_sasrec")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")
EXP_TAG = "baseline"   # flip 없음 (activity 그대로)


def compute_page_overlap(items, top_k_set, movie_info):
    """Average number of top-k genre matches per item across the page (range 0..k).

    Same formula as Agent4Rec's session baseline so the two are directly comparable.
    """
    if not items or not top_k_set:
        return 0.0
    total = 0
    for iid in items:
        genres = movie_info.get(int(iid), {}).get("genres", "")
        item_genres = {g for g in str(genres).split("|") if g}
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


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BrainModule:
    def __init__(self, client, model, profile, episodic_memory, kg_memory, movie_info, captions=None, trace_file=None,
                 top_k_genres=None, baseline=None):
        self.client = client
        self.model = model
        self.uid = profile["uid"]
        self.persona = profile["persona"]
        self.pickiness = profile["pickiness"]["level"]
        self.pickiness_desc = profile["pickiness"]["description"]
        self.habits = profile["habits"]
        self.unique_tastes = profile["unique_tastes"]
        self.episodic_memory = episodic_memory
        self.kg_memory = kg_memory
        self.movie_info = movie_info
        self.captions = captions or {}
        self.interaction_log = []
        self.trace_file = trace_file
        self.top_k_genres = list(top_k_genres) if top_k_genres else []
        self.top_k_set = set(self.top_k_genres)
        self.top_k_str = ", ".join(self.top_k_genres) if self.top_k_genres else "(none)"
        self.baseline = baseline  # float or None
        self.page_overlap_history = []  # list of (page, page_overlap, ratio, page_quality)

    def _trace(self, text):
        if self.trace_file:
            self.trace_file.write(text + "\n")

    def _safe_call(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                resp = self.client.responses.create(model=self.model, input=prompt)
                return resp.output_text
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                return ""

    # ── Helpers ────────────────────────────────────────────────────

    def _movie_description(self, mid):
        info = self.movie_info.get(mid, {})
        title = info.get("title", f"Unknown (id={mid})")
        genres = info.get("genres", "Unknown")
        avg_rating = info.get("rating")
        summary = info.get("summary", "")
        caption = self.captions.get(mid, "")

        desc = f"{title} | Genres: {genres}"
        if avg_rating is not None:
            desc += f" | Avg Rating: {round(float(avg_rating), 2)}"
        if summary:
            desc += f" | Summary: {summary}"
        if caption:
            desc += f" | Visual captions from poster: {caption}"
        return desc

    def _format_profile(self):
        return (f"Persona: {self.persona}\n"
                f"Pickiness: {self.pickiness_desc}\n"
                f"Unique tastes: {self.unique_tastes}\n"
                f"Activity: {self.habits['activity']}\n"
                f"Diversity: {self.habits['diversity']}\n"
                f"Conformity: {self.habits['conformity']}")

    def _get_kg_paths(self, mid):
        """유저 u → liked movie → shared genre → target movie 경로 텍스트 반환."""
        liked_movies = self.kg_memory.graph.get(f"user_{self.uid}", {}).get("liked", set())
        target_genres = self.kg_memory._genres_of(mid)
        target_title = self.movie_info.get(mid, {}).get("title", f"movie_{mid}")

        paths = []
        for movie_node in liked_movies:
            try:
                remapped_mid = int(movie_node.split("_")[1])
            except (IndexError, ValueError):
                continue
            movie_genres = self.kg_memory._genres_of(remapped_mid)
            shared = target_genres & movie_genres
            if shared:
                liked_title = self.movie_info.get(remapped_mid, {}).get("title", movie_node)
                genre_str = ", ".join(g.replace("genre_", "") for g in shared)
                paths.append(
                    f"You liked '{liked_title}' --[{genre_str}]--> '{target_title}'"
                )
        return paths

    def _extract_page_emotion(self, action_text):
        """action_selection 응답에서 Satisfaction/Fatigue/Emotion/Reason 추출."""
        if not action_text:
            return ""
        parts = []
        for label in ['Satisfaction', 'Fatigue', 'Emotion', 'Reason']:
            m = re.search(rf"{label}:\s*([^\n]+)", action_text)
            if m:
                parts.append(f"{label}={m.group(1).strip()}")
        return " | ".join(parts)

    def _preference_block_facts(self):
        """Page-vs-baseline factual block (top-k, baseline, overlap, ratio, page_quality).

        Used by action_selection and the three causal-refinement sub-prompts so
        they share an identical view of the deterministic signal.
        """
        last = self.page_overlap_history[-1] if self.page_overlap_history else None
        if last is None or self.baseline is None or self.baseline <= 0:
            return (
                f"Your top preferred movie categories (derived from your viewing history) are: {self.top_k_str}.\n"
                "(No historical baseline could be computed for you, so use this list qualitatively.)\n"
            )
        p, ov, rt, pq = last
        ratio_str = f"{rt:.2f}x your typical level" if rt is not None else "(undefined — historical baseline is zero)"
        return (
            f"Your top preferred movie categories (derived from your viewing history) are: {self.top_k_str}.\n"
            f"In your typical past viewing, each movie you watched contained on average {self.baseline:.2f} of these top categories. This is your historical baseline of preference adherence.\n"
            f"On Page {p}, the recommended movies contain on average {ov:.2f} of your top categories — that is {ratio_str}.\n"
            f"Page quality (computed from ratio): {pq}  [ABOVE if ratio ≥ 1.0; NORMAL if 0.7 ≤ ratio < 1.0; BELOW if ratio < 0.7]\n"
        )

    _DISPOSITION_LINE = (
        "Your activity trait shapes how you respond to varying page quality:\n"
        "- Low activity = more sensitive to BELOW pages.\n"
        "- High activity = more patient, willing to browse a few more pages despite BELOW."
    )

    _ANCHOR_RULES = (
        "Decide based on page quality and activity:\n"
        "- If page quality is ABOVE: you continue regardless of activity.\n"
        "- If page quality is NORMAL:\n"
        "    * If this is your first NORMAL page so far: browse at least one more page.\n"
        "    * Otherwise: your activity determines the decision.\n"
        "- If page quality is BELOW:\n"
        "    * If Page 1 is BELOW: browse at least one more page to see if the next is better.\n"
        "    * If two consecutive pages are BELOW, exit regardless of activity — even tolerant users should leave at that point.\n"
        "    * Otherwise, judge based on your activity and fatigue.\n"
        "\n"
        "Check the interaction history above — if previous pages were also BELOW, your patience is reduced and exit becomes more likely."
    )

    def _format_interaction_history(self):
        if not self.interaction_log:
            return "  (no interactions yet)"
        lines = []
        for entry in self.interaction_log:
            if "page" not in entry:
                continue
            items_str = ", ".join(
                f"[{i+1}] {self.movie_info.get(mid, {}).get('title', str(mid))}"
                for i, mid in enumerate(entry.get("items", []))
            )
            pq = entry.get("page_quality")
            quality_str = f" [Quality: {pq}]" if pq else ""
            lines.append(f"  Page {entry['page']}{quality_str} ({items_str}): watched {entry['watched_count']}, skipped {entry['skipped_count']}")
            for mid, rating, feelings in entry.get("ratings", []):
                title = self.movie_info.get(mid, {}).get("title", str(mid))
                lines.append(f"    - '{title}': {rating}/5 — {feelings[:80]}")
            if entry.get("page_emotion"):
                lines.append(f"    Feelings on this page: {entry['page_emotion']}")
        return "\n".join(lines) if lines else "  (no interactions yet)"


    # ── Step 1: Multi-round Preference Elicitation ────────────────

    def _build_initial_watch_skip_prompt(self, page, items, episodic_results, kg_results, k1, k2):
        items_text = "\n".join(
            f"  [{i+1}] {self._movie_description(mid)}"
            for i, mid in enumerate(items)
        )
        episodic_text = (
            "\n".join(f"  - {r['text']}" for r in episodic_results)
            if episodic_results else "  (none)"
        )
        kg_text = (
            "\n".join(
                f"  - {self.movie_info.get(r['movie_id'], {}).get('title', r['movie_id'])} [{self.movie_info.get(r['movie_id'], {}).get('genres', '')}]"
                for r in kg_results
            )
            if kg_results else "  (none)"
        )
        return f"""You are simulating a movie user browsing a recommendation system.

## Your Profile
{self._format_profile()}

## Movies on Page {page}
{items_text}

## Relevant Past Memories (top-{k1})
{episodic_text}

## Similar Movies from Knowledge Graph (top-{k2})
The following movies share genres with the page items and match your viewing history:
{kg_text}

## Instructions
For each movie [1]-[{len(items)}], make your initial WATCH/SKIP decision based on your profile, the movie's features, and retrieved evidences.
SKIP only if the movie clearly conflicts with your preferences or has no appeal to you.
For each movie, cite the supporting evidence if any.
Your diversity trait gauges your likelihood of watching movies that may not align with your usual taste:
selective viewers SKIP movies outside their preferred genres,
trailblazers WATCH diverse or unfamiliar styles.

Respond strictly in this format:
[1] [WATCH/SKIP]: <reason> | Evidence: <cited evidence>
[2] [WATCH/SKIP]: <reason> | Evidence: <cited evidence>
[3] [WATCH/SKIP]: <reason> | Evidence: <cited evidence>
[4] [WATCH/SKIP]: <reason> | Evidence: <cited evidence>"""

    def _build_refine_watch_skip_prompt(self, page, items, episodic_results, kg_results, k1, k2, prev_decision_text):
        items_text = "\n".join(
            f"  [{i+1}] {self._movie_description(mid)}"
            for i, mid in enumerate(items)
        )
        episodic_text = (
            "\n".join(f"  - {r['text']}" for r in episodic_results)
            if episodic_results else "  (none)"
        )
        kg_text = (
            "\n".join(
                f"  - {self.movie_info.get(r['movie_id'], {}).get('title', r['movie_id'])} [{self.movie_info.get(r['movie_id'], {}).get('genres', '')}]"
                for r in kg_results
            )
            if kg_results else "  (none)"
        )
        return f"""You are simulating a movie user refining your WATCH/SKIP decisions.

## Your Profile
{self._format_profile()}

## Movies on Page {page}
{items_text}

## Relevant Past Memories (top-{k1}, expanded)
{episodic_text}

## Similar Movies from Knowledge Graph (top-{k2}, expanded)
The following movies share genres with the page items and match your viewing history:
{kg_text}

## Your Previous Decision
{prev_decision_text}

## Instructions
Check each previous decision for contradictions with your persona (e.g., deciding WATCH for a genre you dislike).
If a conflict exists or supporting evidence is insufficient, modify the decision. Otherwise, confirm it.
Remember: your diversity trait should influence how willing you are to WATCH movies outside your usual taste.

Respond strictly in this format:
[1] [WATCH/SKIP]: <reason> | Contradiction: <yes/no>
[2] [WATCH/SKIP]: <reason> | Contradiction: <yes/no>
[3] [WATCH/SKIP]: <reason> | Contradiction: <yes/no>
[4] [WATCH/SKIP]: <reason> | Contradiction: <yes/no>"""

    def _parse_watch_skip(self, text, n):
        decisions = []
        for i in range(1, n + 1):
            match = re.search(rf"\[{i}\].*?\[?(WATCH|SKIP)\]?", text, re.IGNORECASE)
            decisions.append(match.group(1).upper() if match else "SKIP")
        return decisions

    def _has_contradiction(self, text):
        has = bool(re.search(r"Contradiction:\s*yes", text, re.IGNORECASE))
        print(f"-------------- Contradiction: {'yes' if has else 'no'} --------------")
        return has

    def preference_elicitation(self, page, items):
        """Multi-round WATCH/SKIP 결정. 반환: list of (mid, "WATCH"/"SKIP")"""
        k1, k2 = K1_INIT, K2_INIT
        prev_decision_text = None
        parsed = None
        cached_follow_ups = None

        for round_idx in range(MAX_ROUNDS):
            query = "Page {} movies: ".format(page) + "; ".join(
                "{} [genres: {}] (summary: {})".format(
                    self.movie_info.get(mid, {}).get("title", str(mid)),
                    self.movie_info.get(mid, {}).get("genres", ""),
                    self.movie_info.get(mid, {}).get("summary", "")
                ) for mid in items
            )
            episodic_results, cached_follow_ups = self.episodic_memory.retrieve(query, self.model, top_k=k1, follow_ups=cached_follow_ups)

            # query item의 2-hop neighbor를 candidate로 사용 (Graph-Aware Dynamic Item Retrieval)
            # movie_x → has_genre → genre → has_genre⁻¹ → movie_y
            kg_results = []
            for mid in items:
                query_genres = self.kg_memory._genres_of(mid)
                candidate_pairs = [
                    (int(node.split("_")[1]), len(r_dict.get("has_genre", set()) & query_genres))
                    for node, r_dict in self.kg_memory.graph.items()
                    if node.startswith("movie_") and node != f"movie_{mid}"
                    and r_dict.get("has_genre", set()) & query_genres
                ]
                candidates = [m for m, _ in sorted(candidate_pairs, key=lambda x: x[1], reverse=True)[:50]]
                if candidates:
                    kg_results.extend(self.kg_memory.retrieve(self.uid, mid, candidates))

            seen, unique_kg = set(), []
            for r in sorted(kg_results, key=lambda x: x["score"], reverse=True):
                if r["movie_id"] not in seen:
                    seen.add(r["movie_id"])
                    unique_kg.append(r)
            kg_results = unique_kg[:k2]

            # trace: retrieved memory
            episodic_trace = "\n".join(f"    - {r['text']}" for r in episodic_results) if episodic_results else "    (none)"
            kg_trace = "\n".join(
                f"    - {self.movie_info.get(r['movie_id'], {}).get('title', r['movie_id'])} [{self.movie_info.get(r['movie_id'], {}).get('genres', '')}]"
                for r in kg_results
            ) if kg_results else "    (none)"
            self._trace(f"[Step 1] Retrieved Memory (round {round_idx}, k1={k1}, k2={k2}):")
            self._trace(f"  Episodic:\n{episodic_trace}")
            self._trace(f"  KG:\n{kg_trace}\n")

            if round_idx == 0:
                prompt = self._build_initial_watch_skip_prompt(page, items, episodic_results, kg_results, k1, k2)
            else:
                prompt = self._build_refine_watch_skip_prompt(page, items, episodic_results, kg_results, k1, k2, prev_decision_text)

            resp_text = self._safe_call(prompt).strip()
            self._trace(f"[Step 1] Raw LLM Response (round {round_idx}):\n{resp_text}\n")
            parsed = self._parse_watch_skip(resp_text, len(items))

            if round_idx > 0 and not self._has_contradiction(resp_text):
                break

            prev_decision_text = resp_text
            k1 += DELTA_K
            k2 += DELTA_K

        return [(items[i], parsed[i]) for i in range(len(items))]

    # ── Step 2: Item Evaluation ────────────────────────────────────

    def item_evaluation(self, mid):
        """시청한 영화 rating(1-5) + 감상 생성. KG 경로를 참조하여 rating 근거 설명."""
        paths = self._get_kg_paths(mid)
        paths_text = (
            "\n".join(f"  - {p}" for p in paths) if paths else "  (none)"
        )
        prompt = f"""You are simulating a movie user who just watched a movie.

## Your Profile
{self._format_profile()}

## Movie
{self._movie_description(mid)}

## Evidence Paths from Knowledge Graph
{paths_text}

## Instructions
Rate this movie from 1 to 5, considering your conformity trait
(which measures how much your ratings follow the movie's historical average vs your personal taste)
and your pickiness level (which influences your overall rating tendency).
Express your subjective feelings.
Explain how your persona and the evidence paths above influence your rating.

Respond in this format:
Rating: <integer 1-5>
Feelings: <your subjective reaction in 3-4 sentences, referencing how persona and evidence paths influenced your rating>"""

        text = self._safe_call(prompt).strip()
        title = self.movie_info.get(mid, {}).get("title", str(mid))
        self._trace(f"[Step 2] Raw LLM Response ({title}):\n{text}\n")

        rating_match = re.search(r"Rating:\s*([1-5])", text)
        rating = int(rating_match.group(1)) if rating_match else 3

        feelings_match = re.search(r"Feelings:\s*(.+)", text, re.DOTALL)
        feelings = feelings_match.group(1).strip() if feelings_match else text

        return rating, feelings

    # ── Step 3: Action Selection ───────────────────────────────────

    def action_selection(self, page, total_pages, items):
        """감정 기반 액션 결정: 만족도 → 피로도 → 감정 → 액션 (4단계 순차 추론)."""
        watched_mids = {mid for mid, r, f in self.interaction_log[-1].get("ratings", [])}
        items_text = "\n".join(
            f"  [{i+1}] {self.movie_info.get(mid, {}).get('title', str(mid))} — {'WATCHED' if mid in watched_mids else 'SKIPPED'}"
            for i, mid in enumerate(items)
        )

        preference_block = self._preference_block_facts()

        prompt = f"""You are simulating a movie user deciding what to do next in a recommendation system.

## Your Profile
{self._format_profile()}

{preference_block}
{self._DISPOSITION_LINE}

## Interaction History
{self._format_interaction_history()}

## Current Page: {page} / {total_pages}

## Items on this page:
{items_text}

## Instructions
Reason through the following steps sequentially:
Step 1 - Satisfaction: Based primarily on the Page quality stated above, estimate your satisfaction level (HIGH/MEDIUM/LOW). Your activity trait does not change your feeling about the page quality.
Step 2 - Fatigue: Generate your current fatigue level (LOW/MEDIUM/HIGH). Fatigue accumulates with page count; on early pages (1-2) fatigue should typically be LOW unless activity is extremely low.
Step 3 - Emotion: Infer your current emotion (e.g., EXCITED, NEUTRAL, DISAPPOINTED, BORED).
Step 4 - Action: Select the most suitable action:
  - [NEXT]: You are still curious and want to see more recommendations.
  - [EXIT]: You are clearly disappointed AND ready to stop (not just tired or skeptical after one page).
  - [CLICK <number>]: Click on a SKIPPED item (e.g., [CLICK 2]) caught your attention and you want more details before deciding.

{self._ANCHOR_RULES}

Respond in this format:
Satisfaction: <HIGH/MEDIUM/LOW>
Fatigue: <LOW/MEDIUM/HIGH>
Emotion: <emotion>
Action: [NEXT/EXIT/CLICK <number>]
Reason: <one sentence explanation>"""

        text = self._safe_call(prompt)

        click_match = re.search(r"Action:\s*\[CLICK\s+(\d+)\]", text, re.IGNORECASE)
        if click_match:
            return f"CLICK_{click_match.group(1)}", text

        match = re.search(r"Action:\s*\[(NEXT|EXIT)\]", text, re.IGNORECASE)
        action = match.group(1).upper() if match else "NEXT"

        return action, text

    def _handle_click(self, item_idx, items):
        """CLICK 액션 처리: 확장된 설명 제공 후 WATCH/SKIP 결정."""
        mid = items[item_idx - 1]
        extended_desc = self._movie_description(mid)

        prompt = f"""You are simulating a movie user who clicked on a movie for more details.

## Your Profile
{self._format_profile()}

## Extended Movie Details
{extended_desc}

## Instructions
Now that you have seen the full details, decide whether to WATCH or SKIP this movie.
Respond in this format:
Decision: [WATCH/SKIP]
Reason: <brief explanation>"""

        text = self._safe_call(prompt)
        match = re.search(r"Decision:\s*\[(WATCH|SKIP)\]", text, re.IGNORECASE)
        return mid, match.group(1).upper() if match else "SKIP"

    def satisfaction_interview(self):
        """EXIT 시 추천 시스템 만족도 인터뷰 (1-10점)."""
        if self.baseline is None or self.baseline <= 0 or not self.page_overlap_history:
            preference_block = (
                f"Your top preferred movie categories (derived from your viewing history) are: {self.top_k_str}.\n"
                "(No historical baseline could be computed for you.)\n"
            )
        else:
            overlap_lines = []
            for p, ov, rt, pq in self.page_overlap_history:
                if rt is None:
                    overlap_lines.append(f"  Page {p}: {ov:.2f} matches per item ({pq})")
                else:
                    overlap_lines.append(f"  Page {p}: {ov:.2f} matches per item ({rt:.2f}x baseline, {pq})")
            overlap_block = "\n".join(overlap_lines)
            preference_block = (
                f"Your top preferred movie categories (derived from your viewing history) are: {self.top_k_str}.\n"
                f"Your historical baseline of category adherence is {self.baseline:.2f} (average number of top categories per item you watched in the past).\n"
                f"During this session, the recommended pages reflected your top categories at the following levels:\n"
                f"{overlap_block}\n"
                f"Quality labels: ABOVE if ratio ≥ 1.0, NORMAL if 0.7 ≤ ratio < 1.0, BELOW if ratio < 0.7.\n"
            )

        prompt = f"""You are simulating a movie user who has just finished exploring a recommendation system.

## Your Profile
{self._format_profile()}

{preference_block}

## Interaction Summary
{self._format_interaction_history()}

You are having an interview. Answer the following question:
Do you feel satisfied with the recommender system you just interacted with?
Rate the system from 1 to 10.

When rating, weight the following in order of importance:
1. **Watch satisfaction (most important)** — Of the movies you actually watched, did they fit your taste well? A few well-matched watches can make a session worthwhile.
2. **Quality of exposure** — Across the pages you browsed, did you encounter enough items aligned with your top categories? Use the overlap labels as a guide, not a verdict.
3. **Session fit for your activity level** — A low-activity user who browsed only 1-2 pages is acting normally; do NOT penalize the system just because the session was short. A high-activity user who left early because every page was BELOW is a more meaningful negative signal.

Rating anchors (be consistent with these):
- 9-10: Excellent. Most pages matched your taste, you watched multiple movies you liked.
- 7-8: Good. Some great finds, mostly aligned with your taste, satisfying watches.
- 5-6: Mixed. A few decent finds but also misses; reasonable but not exciting.
- 3-4: Disappointing. Few alignments, most items felt off-target.
- 1-2: Bad. Almost nothing aligned, no satisfying watches.

Important:
- Do NOT anchor your rating only on BELOW/ABOVE labels or page count.
- A short session with a couple of satisfying watches can still rate 7+.
- Many BELOW pages with no good finds rate 2-4.

Respond in this format:
RATING: <integer 1-10>
REASON: <explanation>"""
        text = self._safe_call(prompt).strip()
        match = re.search(r"RATING:\s*(\d+)", text, re.IGNORECASE)
        rating = int(match.group(1)) if match else 5
        rating = max(1, min(10, rating))
        return rating, text

    # ── Step 4: Causal Action Refinement ──────────────────────────

    CONSISTENCY_THRESHOLD = 0.5

    def _generate_causal_questions(self, tentative_action, action_context):
        prompt = f"""You are simulating a movie user reflecting on a tentative action.

## Your Profile
{self._format_profile()}

## Page Quality Signal
{self._preference_block_facts()}

## Interaction History
{self._format_interaction_history()}

## Action Selection Context
{action_context}

## Tentative Action: [{tentative_action}]

Generate 2 counterfactual questions that challenge whether [{tentative_action}] is the right action.
Each question should ask "what if" you took a different action, considering the action selection context above.
If this is Page 1 and you have only seen one page so far, lean toward giving the system another chance before exiting.
Example 1: "Does this action match your activity level and curiosity?", "Are there reasons to give the system another page before deciding?"
Example 2: "Does the recent BELOW history actually justify exiting now?", "Would continuing one more page reveal better options?"

Respond with exactly 2 questions, one per line, no numbering."""

        text = self._safe_call(prompt)
        self._trace(f"[Step 4] Causal Questions Raw LLM Response:\n{text.strip()}\n")
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

        return lines[:2]

    def _evaluate_causal_question(self, question, tentative_action, action_context):
        prompt = f"""You are simulating a movie user evaluating a causal question about your next action.

## Your Profile
{self._format_profile()}

## Page Quality Signal
{self._preference_block_facts()}
{self._DISPOSITION_LINE}

## Interaction History
{self._format_interaction_history()}

## Action Selection Context
{action_context}

## Tentative Action: [{tentative_action}]
## Causal Question: {question}

Estimate the outcome based on the page quality and your activity-aware response (see disposition above).
Other factors (satisfaction, persona alignment, fatigue, BELOW page history) all contribute to the verdict.
Does the cause-effect relationship support or contradict the tentative action?

Respond in this format:
Score: <float 0.0-1.0>  (1.0 = fully supports action, 0.0 = fully contradicts)
Verdict: <brief explanation of cause-effect relationship>"""

        text = self._safe_call(prompt).strip()
        self._trace(f"[Step 4] Causal Evaluation Raw LLM Response:\n  Q: {question}\n{text}\n")

        score_match = re.search(r"Score:\s*(\d+\.?\d*|\.\d+)", text)
        try:
            s_q = float(score_match.group(1)) if score_match else 0.5
        except (ValueError, AttributeError):
            s_q = 0.5
        s_q = max(0.0, min(1.0, s_q))

        verdict_match = re.search(r"Verdict:\s*(.+)", text, re.DOTALL)
        v_q = verdict_match.group(1).strip() if verdict_match else text

        return s_q, v_q

    def causal_refinement(self, tentative_action, page, total_pages, action_context):
        """인과 추론으로 최종 액션 확정."""
        questions = self._generate_causal_questions(tentative_action, action_context)

        evaluations = []
        for q in questions:
            s_q, v_q = self._evaluate_causal_question(q, tentative_action, action_context)
            evaluations.append((q, s_q, v_q))

        consistency = sum(s for _, s, _ in evaluations) / len(evaluations) if evaluations else 1.0

        if consistency >= self.CONSISTENCY_THRESHOLD:
            return tentative_action

        qa_text = "\n".join(
            f"  Q: {q}\n  Score: {round(s, 2)} | Verdict: {v}"
            for q, s, v in evaluations
        )
        current_items = self.interaction_log[-1].get("items", [])
        watched_mids = {mid for mid, r, f in self.interaction_log[-1].get("ratings", [])}
        items_text = "\n".join(
            f"  [{i+1}] {self.movie_info.get(mid, {}).get('title', str(mid))} — {'WATCHED' if mid in watched_mids else 'SKIPPED'}"
            for i, mid in enumerate(current_items)
        )
        prompt = f"""You are simulating a movie user adjusting your action based on causal reasoning.

## Your Profile
{self._format_profile()}

## Page Quality Signal
{self._preference_block_facts()}
{self._DISPOSITION_LINE}

{self._ANCHOR_RULES}

## Interaction History
{self._format_interaction_history()}

## Current Page: {page} / {total_pages}
## Items on this page:
{items_text}

## Tentative Action: [{tentative_action}]
## Consistency Score: {round(consistency, 2)} (below threshold {self.CONSISTENCY_THRESHOLD} — action may need adjustment)

## Causal Q&A
{qa_text}

Based on the causal analysis above and the page quality anchor rules, adjust or confirm your action.
Respond in this format:
Final Action: [NEXT/EXIT/CLICK <number>]
Reasoning: <brief explanation>"""

        text = self._safe_call(prompt)
        self._trace(f"[Step 4] Causal Refinement Raw LLM Response:\n{text.strip()}\n")
        click_match = re.search(r"Final Action:\s*\[CLICK\s+(\d+)\]", text, re.IGNORECASE)
        if click_match:
            return f"CLICK_{click_match.group(1)}"
        match = re.search(r"Final Action:\s*\[(NEXT|EXIT)\]", text, re.IGNORECASE)
        return match.group(1).upper() if match else tentative_action

    # ── Step 5: Post-interaction Reflection ───────────────────────

    def post_interaction_reflection(self):
        prompt = f"""You are simulating a movie user reflecting on your recent browsing session.

## Your Profile
{self._format_profile()}

## Interaction Summary
{self._format_interaction_history()}

## Instructions
Reflect on what you learned about your preferences from this session.
Write a brief reflection (2-3 sentences) covering:
- What you enjoyed or disliked
- Insights about your tastes
- What you would do differently next time"""

        reflection = self._safe_call(prompt).strip()
        self.episodic_memory.add(reflection)
        return reflection

    # ── Main simulation loop ───────────────────────────────────────

    def simulate(self, recommendation_pages, max_pages=MAX_PAGES):
        """
        recommendation_pages: list of lists (각 페이지 = 4개 remapped movie ID)
        반환: interaction_log
        """
        self.interaction_log = []
        total_pages = min(len(recommendation_pages), max_pages)
        page = 1

        self._trace(f"Simulating User: {self.uid}\n")
        self._trace(f"Persona: {self.persona}")
        self._trace(f"Pickiness: {self.pickiness} — {self.pickiness_desc}")
        self._trace(f"Habits: {self.habits}")
        self._trace(f"Unique Tastes: {self.unique_tastes}\n")

        while 1 <= page <= total_pages:
            items = recommendation_pages[page - 1]
            print(f"  Page {page}/{total_pages}")

            # Page-vs-baseline metrics (computed before LLM steps so it can be
            # injected into action_selection / interview prompts).
            page_overlap = compute_page_overlap(items, self.top_k_set, self.movie_info)
            if self.baseline and self.baseline > 0:
                ratio = page_overlap / self.baseline
            else:
                ratio = None
            page_quality = classify_page_quality(ratio)
            self.page_overlap_history.append((page, page_overlap, ratio, page_quality))

            items_desc = ", ".join(
                f"{self.movie_info.get(mid, {}).get('title', str(mid))} [{self.movie_info.get(mid, {}).get('genres', '')}]"
                for mid in items
            )
            self._trace(f"{'='*60}")
            self._trace(f"Page {page}/{total_pages}: {items_desc}")
            self._trace(f"  page_overlap={page_overlap:.3f} | ratio={'NA' if ratio is None else f'{ratio:.3f}'} | quality={page_quality}")
            self._trace(f"{'='*60}\n")

            # Step 1: WATCH/SKIP
            watch_skip = self.preference_elicitation(page, items)
            watched = [(mid, d) for mid, d in watch_skip if d == "WATCH"]
            skipped = [(mid, d) for mid, d in watch_skip if d == "SKIP"]

            self._trace(f"[Step 1] Preference Elicitation:")
            for mid, decision in watch_skip:
                title = self.movie_info.get(mid, {}).get("title", str(mid))
                self._trace(f"  {title}: [{decision}]")
            self._trace("")

            # Step 2: Item evaluation
            ratings_log = []
            for mid, _ in watched:
                rating, feelings = self.item_evaluation(mid)
                ratings_log.append((mid, rating, feelings))
                # KG memory 업데이트
                self.kg_memory.add_user_movie(self.uid, mid, rating)
                # 개별 liked/disliked 항목을 episodic memory에 추가
                title = self.movie_info.get(mid, {}).get("title", str(mid))
                self.episodic_memory.add(
                    self.episodic_memory._format_history_entry(title, rating)
                )

                self._trace("")

            self.interaction_log.append({
                "page": page,
                "items": items,
                "watch_skip": [(mid, d) for mid, d in watch_skip],
                "watched_count": len(watched),
                "skipped_count": len(skipped),
                "ratings": ratings_log,
                "page_overlap": round(page_overlap, 6),
                "ratio_vs_baseline": round(ratio, 6) if ratio is not None else None,
                "page_quality": page_quality,
            })

            # ------------------------------------------------

            # Step 3: Action selection
            tentative_action, action_context = self.action_selection(page, total_pages, items)
            self.interaction_log[-1]["page_emotion"] = self._extract_page_emotion(action_context)

            self._trace(f"[Step 3] Raw LLM Response:\n{action_context}\n")

            # Step 4: Causal refinement
            final_action = self.causal_refinement(tentative_action, page, total_pages, action_context)
            print(f"    Action: {final_action}")

            self._trace(f"[Step 4] Final Action: [{final_action}]\n")

            # CLICK 처리
            if final_action.startswith("CLICK_"):
                idx = int(final_action.split("_")[1])
                if 1 <= idx <= len(items):
                    clicked_mid, click_decision = self._handle_click(idx, items)
                    click_title = self.movie_info.get(clicked_mid, {}).get("title", str(clicked_mid))
                    self._trace(f"[CLICK] {click_title} → [{click_decision}]")

                    if click_decision == "WATCH" and clicked_mid not in [m for m, _ in watched]:
                        rating, feelings = self.item_evaluation(clicked_mid)
                        ratings_log.append((clicked_mid, rating, feelings))
                        self.kg_memory.add_user_movie(self.uid, clicked_mid, rating)
                        title = self.movie_info.get(clicked_mid, {}).get("title", str(clicked_mid))
                        self.episodic_memory.add(
                            self.episodic_memory._format_history_entry(title, rating)
                        )
                        # click→WATCH 반영
                        self.interaction_log[-1]["watched_count"] += 1
                        self.interaction_log[-1]["skipped_count"] -= 1
                        self.interaction_log[-1]["ratings"] = ratings_log
                        self.interaction_log[-1]["watch_skip"] = [
                            (m, "WATCH" if m == clicked_mid else d)
                            for m, d in self.interaction_log[-1]["watch_skip"]
                        ]

                        self._trace(f"  Rating: {rating}/5")
                        self._trace(f"  Feelings: {feelings}")
                    self._trace("")
                final_action = "NEXT"

            # Episodic memory 페이지 요약
            final_watched_mids = [mid for mid, _, _ in ratings_log]
            all_titles = [self.movie_info.get(mid, {}).get("title", str(mid)) for mid in items]
            final_watched_titles = [self.movie_info.get(mid, {}).get("title", str(mid)) for mid in final_watched_mids]
            final_rating_vals = [r for _, r, _ in ratings_log]
            final_skipped_titles = [
                self.movie_info.get(mid, {}).get("title", str(mid))
                for mid in items if mid not in final_watched_mids
            ]
            self.episodic_memory.add_rs_interaction(
                page_number=page,
                item_type="movies",
                name_all_items=all_titles,
                watched_items=final_watched_titles,
                ratings_list=final_rating_vals,
                dislike_items=final_skipped_titles,
            )

            if final_action == "EXIT" or page >= total_pages:
                satisfy_rating, satisfy_text = self.satisfaction_interview()
                print(f"    Satisfaction: {satisfy_rating}/10")

                self._trace(f"[Interview] Raw LLM Response:\n{satisfy_text}\n")
                self._trace(f"[Interview] Satisfaction: {satisfy_rating}/10")
                self._trace(f"  Browsed {page} pages, then EXIT.\n")

                # Step 5: Post-interaction reflection (원본에서도 주석 처리됨)
                # reflection = self.post_interaction_reflection()
                # print(f"    Reflection: {reflection[:100]}...")
                self.interaction_log.append({"satisfy": satisfy_rating, "interview_text": satisfy_text})
                break
            else:  # NEXT (or CLICK which already set final_action = "NEXT")
                page += 1

        return self.interaction_log


# ── Data loading ───────────────────────────────────────────────────

def load_data(rec_ratio="1to1"):
    # movie_detail.csv: movie_id (0-indexed) -> {title, genres, rating, summary}
    movie_df = pd.read_csv(os.path.join(DATA_DIR, "movie_detail.csv"))
    movie_info = {
        int(row["movie_id"]): {
            "title": row["title"],
            "genres": row["genres"],
            "rating": row["rating"],
            "summary": row.get("summary", ""),
        }
        for _, row in movie_df.iterrows()
    }

    # SASRec recommendation_results_{rec_ratio}.json: {uid(str): [20 items (top-K + bottom-K, sorted)]}
    rec_path = os.path.join(SASREC_DIR, f"recommendation_results_{rec_ratio}.json")
    recommendations = {}
    with open(rec_path) as f:
        rec_raw = json.load(f)
    for uid_str, items in rec_raw.items():
        uid = int(uid_str)
        if isinstance(items, list) and len(items) >= 20:
            recommendations[uid] = [int(x) for x in items[:20]]

    # GT 평점 dict 구성: gt_ratings[remapped_uid][remapped_mid] = rating
    with open(os.path.join(DATA_DIR, "user_id_map.pkl"), "rb") as f:
        user_id_map = pickle.load(f)
    with open(os.path.join(DATA_DIR, "movie_id_map.pkl"), "rb") as f:
        movie_id_map = pickle.load(f)
    gt_ratings = {}
    with open(os.path.join(DATA_DIR, "ratings.dat"), "r") as f:
        for line in f:
            uid, mid, rat, _ = line.strip().split("::")
            uid, mid, rat = int(uid), int(mid), float(rat)
            if uid in user_id_map and mid in movie_id_map:
                r_uid = user_id_map[uid]
                r_mid = movie_id_map[mid]
                if r_uid not in gt_ratings:
                    gt_ratings[r_uid] = {}
                gt_ratings[r_uid][r_mid] = rat

    return movie_info, recommendations, gt_ratings


def build_recommendation_pages(rec_list, items_per_page=ITEMS_PER_PAGE):
    """추천 리스트를 페이지 단위(4개씩)로 분할."""
    pages = []
    for i in range(0, len(rec_list), items_per_page):
        page = rec_list[i:i + items_per_page]
        if page:
            pages.append(page)
    return pages


TRACE_DIR = None


def _simulate_one_user(uid, client, model, profiles, episodic_data, kg_data, kg_state,
                       movie_info, recommendations, gt_ratings, captions,
                       EpisodicMemory, KGMemory, top_k_map=None, baseline_map=None):
    """단일 유저 시뮬레이션 (병렬 실행 가능)."""
    profile = profiles[str(uid)]
    rec_list = recommendations.get(uid, [])
    if not rec_list:
        print(f"  User {uid}: No recommendations, skipping.")
        return uid, None, [], None, None

    em = EpisodicMemory(client)
    em.from_dict(episodic_data[str(uid)])

    kg = KGMemory(client)
    kg.from_dict(kg_data)
    user_key = f"user_{uid}"
    if str(uid) in kg_state:
        for r, t_list in kg_state[str(uid)].items():
            kg.graph[user_key][r] |= set(t_list)

    pages = build_recommendation_pages(rec_list)

    os.makedirs(TRACE_DIR, exist_ok=True)
    trace_path = os.path.join(TRACE_DIR, f"user_{uid}")
    trace_file = open(trace_path, "w", encoding="utf-8")

    uid_key = str(uid)
    top_k = (top_k_map or {}).get(uid_key, [])
    baseline = ((baseline_map or {}).get(uid_key) or {}).get("baseline")

    brain = BrainModule(client, model, profile, em, kg, movie_info, captions, trace_file=trace_file,
                         top_k_genres=top_k, baseline=baseline)
    log = brain.simulate(pages)

    trace_file.close()

    # rating comparison 수집
    comparisons = []
    for entry in log:
        if "page" not in entry:
            continue
        for mid, rating, feelings in entry.get("ratings", []):
            gt_rat = gt_ratings.get(uid, {}).get(mid, None)
            comparisons.append({
                "user_id": uid,
                "item_id": mid,
                "title": movie_info.get(mid, {}).get("title", str(mid)),
                "predicted_rating": rating,
                "gt_rating": gt_rat
            })

    kg_user_edges = {r: list(t_set) for r, t_set in kg.graph.get(user_key, {}).items()}
    return uid, log, comparisons, em.to_dict(), kg_user_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_users", type=str,
                        default=os.path.join(DATA_DIR, "user_sets.txt"),
                        help="Path to user_sets.txt")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--rec_ratio", type=str, default="1to1", choices=["1to1", "1to3", "1to9"],
                        help="SASRec recommendation list good:bad ratio (Agent4Rec과 동일)")
    parser.add_argument("--profiles", type=str, default=os.path.join(RESULT_DIR, "user_profiles.json"))
    parser.add_argument("--episodic", type=str, default=os.path.join(RESULT_DIR, "episodic_memory_init.json"))
    parser.add_argument("--episodic_state", type=str, default=None,
                        help="Accumulated post-simulation episodic memory. Default: <exp_dir>/episodic_memory_state.json")
    parser.add_argument("--kg", type=str, default=os.path.join(RESULT_DIR, "kg_memory.json"))
    parser.add_argument("--kg_state", type=str, default=None,
                        help="Accumulated post-simulation KG edges per user. Default: <exp_dir>/kg_memory_state.json")
    parser.add_argument("--top_k_path", type=str, default=os.path.join(RESULT_DIR, "user_top_k_genres.json"),
                        help="Per-user top-k preferred genres JSON.")
    parser.add_argument("--baseline_path", type=str, default=os.path.join(RESULT_DIR, "session_baseline_w10.json"),
                        help="Per-user session baseline JSON.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8, help="병렬 처리 수")
    parser.add_argument("--num_users", type=int, default=None, help="처리할 유저 수")
    args = parser.parse_args()

    global TRACE_DIR
    model_tag = args.model.replace("/", "_")
    exp_dir = os.path.join(RESULT_DIR, f"{EXP_TAG}_{args.rec_ratio}_{model_tag}")
    os.makedirs(exp_dir, exist_ok=True)
    if args.output is None:
        args.output = os.path.join(exp_dir, "simulation_results.json")
    if args.episodic_state is None:
        args.episodic_state = os.path.join(exp_dir, "episodic_memory_state.json")
    if args.kg_state is None:
        args.kg_state = os.path.join(exp_dir, "kg_memory_state.json")
    TRACE_DIR = os.path.join(exp_dir, "traces")

    base_dir = os.path.dirname(__file__)
    episodic_mod = _load_module("episodic_memory", os.path.join(base_dir, "6_Episodic_Memory.py"))
    kg_mod = _load_module("kg_memory", os.path.join(base_dir, "7_KG_Memory.py"))
    EpisodicMemory = episodic_mod.EpisodicMemory
    KGMemory = kg_mod.KGMemory

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading data...")
    movie_info, recommendations, gt_ratings = load_data(args.rec_ratio)

    with open(args.profiles, encoding="utf-8") as f:
        profiles = json.load(f)
    with open(args.episodic, encoding="utf-8") as f:
        episodic_data = json.load(f)
    episodic_state = {}
    if os.path.exists(args.episodic_state):
        with open(args.episodic_state, encoding="utf-8") as f:
            episodic_state = json.load(f)
        for uid_str, mems in episodic_state.items():
            episodic_data[uid_str] = mems
        print(f"  Loaded accumulated episodic state for {len(episodic_state)} users from {args.episodic_state}")
    with open(args.kg, encoding="utf-8") as f:
        kg_data = json.load(f)
    kg_state = {}
    if os.path.exists(args.kg_state):
        with open(args.kg_state, encoding="utf-8") as f:
            kg_state = json.load(f)
        print(f"  Loaded accumulated KG state for {len(kg_state)} users from {args.kg_state}")

    captions = {}
    caption_path = os.path.join(RESULT_DIR, "captions.json")
    if os.path.exists(caption_path):
        with open(caption_path, encoding="utf-8") as f:
            captions = {int(k): v for k, v in json.load(f).items()}

    with open(args.top_k_path, encoding="utf-8") as f:
        top_k_map = json.load(f)
    with open(args.baseline_path, encoding="utf-8") as f:
        baseline_map = json.load(f)
    print(f"  Loaded top-k for {len(top_k_map)} users, baseline for {len(baseline_map)} users")

    with open(args.target_users, 'r') as f:
        target = set(int(line.strip()) for line in f if line.strip())
    user_ids = [int(k) for k in profiles.keys() if int(k) in target]
    user_ids = user_ids[:args.num_users]

    # --------------------------------------------------------------------------------

    results = {}
    rating_comparison = []
    save_lock = threading.Lock()

    total_satisfy = 0
    total_pages_browsed = 0
    total_page_view = 0
    total_item_view = 0
    num_users_done = 0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Simulating {len(user_ids)} users with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _simulate_one_user, uid, client, args.model, profiles,
                episodic_data, kg_data, kg_state, movie_info, recommendations,
                gt_ratings, captions, EpisodicMemory, KGMemory,
                top_k_map, baseline_map,
            ): uid
            for uid in user_ids
        }

        for future in as_completed(futures):
            uid, log, comparisons, em_dump, kg_user_edges = future.result()
            if log is None:
                continue

            num_users_done += 1
            results[uid] = log
            rating_comparison.extend(comparisons)
            if em_dump is not None:
                episodic_state[str(uid)] = em_dump
            if kg_user_edges is not None:
                kg_state[str(uid)] = kg_user_edges

            for entry in log:
                if "page" not in entry:
                    total_satisfy += entry.get("satisfy", 0)
                    continue
                total_pages_browsed += 1
                watched = entry.get("watched_count", 0)
                total_item_view += watched
                if watched > 0:
                    total_page_view += 1

            if total_pages_browsed > 0:
                print(f"  [{num_users_done}/{len(user_ids)}] User {uid} done | "
                      f"Satisfy: {total_satisfy/num_users_done:.2f} | "
                      f"Avg Pages: {total_pages_browsed/num_users_done:.2f} | "
                      f"Page View: {total_page_view/total_pages_browsed:.4f} | "
                      f"Item View: {total_item_view/(ITEMS_PER_PAGE*total_pages_browsed):.4f}")

            with save_lock:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                with open(args.episodic_state, "w", encoding="utf-8") as f:
                    json.dump(episodic_state, f, ensure_ascii=False, indent=2)
                with open(args.kg_state, "w", encoding="utf-8") as f:
                    json.dump(kg_state, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {args.output}")

    if num_users_done > 0 and total_pages_browsed > 0:
        stats = (f"Satisfy: {total_satisfy/num_users_done:.2f} | "
                 f"Avg Pages: {total_pages_browsed/num_users_done:.2f} | "
                 f"Page View: {total_page_view/total_pages_browsed:.4f} | "
                 f"Item View: {total_item_view/(ITEMS_PER_PAGE*total_pages_browsed):.4f}")
        print(f"\n[Final] {stats}")
        stats_path = os.path.join(exp_dir, "simulation_stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(stats)
        print(f"Stats saved to {stats_path}")

    comparison_path = os.path.join(exp_dir, "rating_comparison.jsonl")
    with open(comparison_path, "w", encoding="utf-8") as rf:
        for r in rating_comparison:
            rf.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Rating comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
