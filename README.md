# A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators

Official code for **"A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators"** (CIKM 2026 short paper).

LLM user simulators inject a behavioral *activity* trait that is meant to control only how long a user browses. In practice, amplifying activity makes the simulator accept preference-mismatched items to keep browsing (**trait interference**), and satisfaction scores inflate with page count (**evaluation invalidity**).
PQA fixes this by giving the simulator a personalized page-level quality anchor `μ_u`, computed from the user's own history. Each page is labeled ABOVE / NORMAL / BELOW against this anchor before the continue-or-exit decision, so activity modulates browsing depth only within preference-conforming pages.

This repository contains PQA applied to two simulators, [Agent4Rec](https://github.com/LehengTHU/Agent4Rec) and [SimUSER](https://aclanthology.org/2025.acl-industry.5/), on MovieLens and Amazon CDs & Vinyl.

## Repository structure

```
Agent4Rec/
├── MovieLens/
│   ├── 0_sort_history_by_time.py     # sort train/valid/test histories by timestamp
│   ├── 1_extract_top_k_genres.py     # top-K categories per user
│   ├── 2_session_baseline.py         # anchor μ_u (recency-weighted session overlap)
│   └── 3_simulations_inference.py    # Agent4Rec + PQA simulation
└── CDs_and_Vinyl/
    ├── persona/                      # Agent4Rec persona generation (1 → 2 → 3)
    ├── 1_extract_top_k_genres.py
    ├── 2_session_baseline.py
    └── 3_simulations_inference.py
SimUSER/
├── MovieLens/
│   ├── Phase_1_Persona_Matching/     # 1 summary → 2 persona candidates → 3 self-consistent selection
│   └── Phase_2_Simulate_Interactions/
│       ├── 4_Persona_Module.py
│       ├── 5_1_download_posters.py   # optional: poster captions (perception)
│       ├── 5_2_Perception_Module.py  # optional: poster captions (perception)
│       ├── 6_Episodic_Memory.py
│       ├── 7_KG_Memory.py
│       └── 8_Brain_Module1.py        # SimUSER + PQA simulation
└── CDs_and_Vinyl/                    # same layout (no perception step)
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # all LLM calls use GPT-4o-mini by default
```

## Data

Datasets, personas and recommendation lists are not included. Each script reads from paths relative to its own folder:

- **Agent4Rec / MovieLens** — `dataset/MovieLens/` with the Agent4Rec MovieLens files (`ratings.dat`, `movie_detail.csv`, `user_statistic.csv`, `all_personas_like_modify.csv`, `user_id_map.pkl`, `movie_id_map.pkl`, `train.txt`, `valid.txt`, `test.txt`, `user_sets.txt`).
- **Agent4Rec / CDs** — `dataset/CDs_and_Vinyl/` (`train.json`, `valid.json`, `test.json`, `meta.json`, `review.json`, `user2id.json`, `item2id.json`, `user_sets.txt`).
- **SimUSER** — the same files under `SimUSER/<dataset>/dataset/<dataset>/`. On CDs, SimUSER also reads `personality.json` and `all_persona.csv` produced by `Agent4Rec/CDs_and_Vinyl/persona/`.
- **Recommendation lists** — `dataset/MV_sasrec/` (MovieLens) or `dataset/CDs_and_Vinyl_sasrec/` (CDs) containing `recommendation_results_{1to1,1to3,1to9}.json`: `{user_id: [20 item ids]}`, ordered by the backbone recommender's (SASRec) predicted logits, with a 1:k ratio of top-ranked to bottom-ranked items.

`user_sets.txt` lists the target user ids, one per line.

## Running

### Agent4Rec + PQA

```bash
cd Agent4Rec/MovieLens
python 0_sort_history_by_time.py          # MovieLens only
python 1_extract_top_k_genres.py          # → result/user_top_k_genres.json
python 2_session_baseline.py              # → result/session_baseline_w10.json
python 3_simulations_inference.py --rec_ratio 1to1
```

For CDs, first generate personas from `Agent4Rec/CDs_and_Vinyl`:

```bash
cd Agent4Rec/CDs_and_Vinyl
python persona/1_gen_persona_parallel.py
python persona/2_gen_personality.py
python persona/3_process_persona.py
python 1_extract_top_k_genres.py          # → result/user_top_k_categories.json
python 2_session_baseline.py
python 3_simulations_inference.py --rec_ratio 1to1
```

### SimUSER + PQA

```bash
cd SimUSER/MovieLens
python Phase_1_Persona_Matching/1_generate_short_summary_of_user_preference.py
python Phase_1_Persona_Matching/2_generate_persona_candidate.py
python Phase_1_Persona_Matching/3_self_consistent_persona_evalution.py
python Phase_2_Simulate_Interactions/4_Persona_Module.py
python Phase_2_Simulate_Interactions/5_1_download_posters.py      # optional (MovieLens)
python Phase_2_Simulate_Interactions/5_2_Perception_Module.py     # optional (MovieLens)
python Phase_2_Simulate_Interactions/6_Episodic_Memory.py
python Phase_2_Simulate_Interactions/7_KG_Memory.py
python Phase_2_Simulate_Interactions/8_Brain_Module1.py --rec_ratio 1to1 \
    --top_k_path ../../Agent4Rec/MovieLens/result/user_top_k_genres.json \
    --baseline_path ../../Agent4Rec/MovieLens/result/session_baseline_w10.json
```

Simulation logs and summary statistics (P_view, N_exit, S_sat) are written under `simulation_<dataset>/` (Agent4Rec) or `result/` (SimUSER).

## Citation

```bibtex
@article{kim2026behavioral,
  title={A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators},
  author={Kim, Chaehyun and Kim, Sein and Kang, Hongseok and Park, Chanyoung},
  journal={arXiv preprint arXiv:2609.25572},
  year={2026}
}

```
