#!/usr/bin/env bash
# mip_paper_cells.sh — the MIP paper's experiments (arXiv:2607.26148) as copy-pasteable runner lines.
# A reference, not a program: the first line below exits, so the file never runs as a whole.
# Part 1 = commands (by paper table). Part 2 = the paper's numbers and where each run dir sits.
# Every seat here runs the same loop on EmbodiedScore-envs (exp_workspace/bareES) — a new line, not a
# re-run of the paper's number. Blocks headed "not seated yet" name the seats this repo does not have.
echo "reference only — copy one line, do not run the file"; exit 0

# ══════════════════════════ Part 1 · commands ══════════════════════════

# probe / fake / board / fill — the four ways to launch any seat
python runner.py std_r2r_es_bareES harness=cc model=fable-5 run.episodes=0-9
python runner.py std_r2r_es_bareES harness=cc model=fable-5 +run.fake=true
python runner.py std_r2r_es_bareES harness=cc model=fable-5
python runner.py std_r2r_es_bareES harness=cc model=fable-5 run.episodes=3,7 run.resume=true

# Table 2 · bare board — R2R-CE rand100, default effort, Claude SDK
python runner.py std_r2r_es_bareES harness=cc model=sonnet-5
python runner.py std_r2r_es_bareES harness=cc model=opus-4.8
python runner.py std_r2r_es_bareES harness=cc model=fable-5
python runner.py std_r2r_es_bareES harness=cc model=opus-5

# Table 2 · bare board — mini-swe-agent (qwen3.5-4b/9b: local ollama; qwen*-plus: DashScope key; gpt: OpenAI key; claude: Anthropic key)
python runner.py std_r2r_es_bareES harness=mini model=qwen3.5-4b
python runner.py std_r2r_es_bareES harness=mini model=qwen3.5-9b
python runner.py std_r2r_es_bareES harness=mini model=qwen3.5-plus
python runner.py std_r2r_es_bareES harness=mini model=qwen3.6-plus
python runner.py std_r2r_es_bareES harness=mini model=qwen3.7-plus
python runner.py std_r2r_es_bareES harness=mini model=gpt-5.5
python runner.py std_r2r_es_bareES harness=mini model=gpt-5.6
python runner.py std_r2r_es_bareES harness=mini model=sonnet-5
python runner.py std_r2r_es_bareES harness=mini model=opus-4.8
python runner.py std_r2r_es_bareES harness=mini model=fable-5
python runner.py std_r2r_es_bareES harness=mini model=opus-5

# Table 2 · bare board — Codex CLI (auth = codex login; gpt-5.6 is codex's gpt-5.6-sol; the paper's cell ran at low, codex's default here is medium)
python runner.py std_r2r_es_bareES harness=codex model=gpt-5.5
python runner.py std_r2r_es_bareES harness=codex model=gpt-5.6 effort=low

# Table 2 (last row) + Table 3 · effort — Claude SDK at max
python runner.py std_r2r_es_bareES harness=cc model=sonnet-5 effort=max
python runner.py std_r2r_es_bareES harness=cc model=opus-4.8 effort=max
python runner.py std_r2r_es_bareES harness=cc model=fable-5 effort=max
python runner.py std_r2r_es_bareES harness=cc model=opus-5 effort=max

# Table 3 · effort — Codex CLI / mini-swe-agent at xhigh
python runner.py std_r2r_es_bareES harness=codex model=gpt-5.5 effort=xhigh
python runner.py std_r2r_es_bareES harness=codex model=gpt-5.6 effort=xhigh
python runner.py std_r2r_es_bareES harness=mini model=gpt-5.5 effort=xhigh
python runner.py std_r2r_es_bareES harness=mini model=gpt-5.6 effort=xhigh

# Table 4 · interface — VLNVerse primitive
python runner.py std_vlnverse_es_bareES harness=cc model=sonnet-5
python runner.py std_vlnverse_es_bareES harness=cc model=fable-5

# Table 4 · interface — not seated yet (the waypoint arm: observe / goto / stop)
python runner.py std_r2r_es_wp harness=mini model=qwen3.5-4b
python runner.py std_r2r_es_wp harness=mini model=qwen3.5-9b
python runner.py std_r2r_es_wp harness=mini model=qwen3.5-plus
python runner.py std_r2r_es_wp harness=codex model=gpt-5.5
python runner.py std_r2r_es_wp harness=codex model=gpt-5.6
python runner.py std_r2r_es_wp harness=cc model=sonnet-5
python runner.py std_r2r_es_wp harness=cc model=opus-4.8
python runner.py std_r2r_es_wp harness=cc model=fable-5
python runner.py std_r2r_es_wp harness=cc model=opus-5
python runner.py std_vlnverse_es_wp harness=cc model=sonnet-5
python runner.py std_vlnverse_es_wp harness=cc model=fable-5

# Table 5 · hybrid — not seated yet (the hybrid arm)
python runner.py std_r2r_es_hybrid harness=cc model=fable-5

# Table 6 · long horizon — R2R-CE / RxR-CE primitives
python runner.py std_r2r_es_bareES harness=cc model=fable-5
python runner.py std_rxr_es_bareES harness=cc model=fable-5

# Table 6 · long horizon — not seated yet
python runner.py std_rxr_es_wp harness=cc model=fable-5

# Appendix B · beyond R2R-CE — HM-EQA (mip100)
python runner.py std_hmeqa_es_bareES harness=cc model=fable-5

# ══════════════════════════ Part 2 · the paper's numbers ══════════════════════════
# SR / SPL / OSR in percent, NE in metres. The last column is the paper-era run directory
# (the std-v2 board's cell name under <harness dir>/); those archives are lab-internal and not
# distributed — the numbers are the paper's, the names say which cell each number came from.

cat <<'PERF'
Table 2 · bare board (R2R-CE rand100, default effort)
harness   model         SR     SPL    NE    OSR   paper-era run dir
mini      qwen3.5-4b     5     4.58  8.93   11   mini-swe-agent/std_r2r_mini_qwen3.5-4b_default_bare
mini      qwen3.5-9b     7     5.36  8.63   15   mini-swe-agent/std_r2r_mini_qwen3.5-9b_default_bare
mini      qwen3.5-plus  34    26.74  6.32   48   mini-swe-agent/std_r2r_mini_qwen3.5-plus_default_bare
mini      qwen3.7-plus  42    32.85  6.81   55   mini-swe-agent/std_r2r_mini_qwen3.7-plus_default_bare
mini      qwen3.6-plus  45    33.27  6.25   57   mini-swe-agent/std_r2r_mini_qwen3.6-plus_default_bare
mini      gpt-5.5       52    44.24  7.29   57   mini-swe-agent/std_r2r_mini_gpt-5.5_default_bare
mini      gpt-5.6       60    42.04  4.99   68   mini-swe-agent/std_r2r_mini_gpt-5.6_default_bare        (medium; low twin …_low_bare = 54)
mini      sonnet-5      53    38.14  5.52   61   mini-swe-agent/std_r2r_mini_sonnet-5_default_bare
mini      opus-4.8      63    52.77  4.21   65   mini-swe-agent/std_r2r_mini_opus-4.8_default_bare
mini      fable-5       72    59.08  4.48   77   mini-swe-agent/std_r2r_mini_fable-5_default_bare
mini      opus-5        69    50.24  5.15   78   mini-swe-agent/std_r2r_mini_opus-5_default_bare
sdk       sonnet-5    51.3±1.2 37.84 5.80  61.3  claudecode/std_r2r_cc_sonnet-5_default_bare{,_rep2,_rep3,_rep4}   mean of three
sdk       opus-4.8    55.7±2.3 47.31 5.24  59.3  claudecode/std_r2r_cc_opus-4.8_default_bare{,_rep2,_rep3,_rep4}   mean of three
sdk       fable-5     68.3±1.5 58.02 5.13  73.3  claudecode/std_r2r_cc_fable-5_default_bare{,_rep2,_rep3,_rep4}    mean of rep2/3/4; single run 75
sdk       opus-5      70.7±3.5 55.21 4.79  78.3  claudecode/std_r2r_cc_opus-5_default_bare{,_rep2,_rep3,_rep4}     mean of three
codex     gpt-5.5       45    35.74  5.66   51   codex/std_r2r_codex_gpt-5.5_default_bare                 (medium; low twin …_low_bare = 45)
codex     gpt-5.6       56    41.57  6.15   64   codex/std_r2r_codex_gpt-5.6_default_bare                 (sol, low)
sdk       fable-5 max   78    65.27  3.84   83   claudecode/std_r2r_cc_fable-5_max_bare

Table 3 · effort (default → max / xhigh; SR, mean t/ep s, median t/ep s)
sdk       sonnet-5    51.3 → 56   446 → 698   323 → 532   claudecode/std_r2r_cc_sonnet-5_max_bare_rep2    (first max run 49)
sdk       opus-4.8    55.7 → 56   437 → 899   254 → 652   claudecode/std_r2r_cc_opus-4.8_max_bare_rep2    (first max run 55, 9 timeouts)
sdk       fable-5     68.3 → 78   379 → 633   210 → 484   claudecode/std_r2r_cc_fable-5_max_bare
sdk       opus-5      70.7 → 74   332 → 553   228 → 429   claudecode/std_r2r_cc_opus-5_max_bare_rep2
codex     gpt-5.5       45 → 50                           codex/std_r2r_codex_gpt-5.5_max_bare
codex     gpt-5.6       56 → 62                           codex/std_r2r_codex_gpt-5.6_max_bare
mini      gpt-5.5       52 → 50                           mini-swe-agent/std_r2r_mini_gpt-5.5_max_bare
mini      gpt-5.6       60 → 55                           mini-swe-agent/std_r2r_mini_gpt-5.6_max_bare

Table 4 · interface (SR primitive → waypoint; SPL primitive → waypoint)
R2R-CE    qwen3.5-4b     5 → 43    4.58 → 34.18   mini-swe-agent/std_r2r_mini_qwen3.5-4b_default_wp
R2R-CE    qwen3.5-9b     7 → 44    5.36 → 32.81   mini-swe-agent/std_r2r_mini_qwen3.5-9b_default_wp
R2R-CE    qwen3.5-plus  34 → 53   26.74 → 44.05   mini-swe-agent/std_r2r_mini_qwen3.5-plus_default_wp
R2R-CE    gpt-5.5       45 → 67   35.74 → 58.85   codex/std_r2r_codex_gpt-5.5_default_wp
R2R-CE    gpt-5.6-sol   56 → 73   41.57 → 63.98   codex/std_r2r_codex_gpt-5.6_default_wp                 (medium, one step above its bare cell)
R2R-CE    sonnet-5    51.3 → 60   37.84 → 45.83   claudecode/std_r2r_cc_sonnet-5_default_wp
R2R-CE    opus-4.8    55.7 → 65   47.31 → 54.23   claudecode/std_r2r_cc_opus-4.8_default_wp
R2R-CE    fable-5     68.3 → 69   58.02 → 59.15   claudecode/std_r2r_cc_fable-5_default_wp
R2R-CE    opus-5      70.7 → 72   55.21 → 60.52   claudecode/std_r2r_cc_opus-5_default_wp
VLNVerse  sonnet-5      78 → 72   52.06 → 22.86   (collision 6 → 35)   run dirs never landed here (collaborator side)
VLNVerse  fable-5       84 → 80   62.47 → 42.57   (collision 7 → 27)   run dirs never landed here (collaborator side)

Table 5 · hybrid (fable-5, Claude SDK, R2R-CE; SR, SPL, NE, median steps / s / calls)
primitives default   68.3±1.5  58.02  5.13   87  210  39   (the bare-board reps)
waypoint   default   69        59.15  4.30   38   91  15   claudecode/std_r2r_cc_fable-5_default_wp
hybrid     default   76.7±0.6  63.63  3.49   48  112  20   claudecode/std_r2r_cc_fable-5_default_hybrid{,_rep2,_rep3}
primitives max       78        65.27  3.84   97  484  48   claudecode/std_r2r_cc_fable-5_max_bare

Table 6 · long horizon (fable-5, Claude SDK; SR, time s, context tokens)
R2R-CE    primitives  70  187  22.4k   claudecode/std_r2r_cc_fable-5_default_bare_rep2
R2R-CE    waypoint    69   91  14.6k   claudecode/std_r2r_cc_fable-5_default_wp
RxR-CE    primitives  26  527  49.2k   claudecode/rxr_bare_fable_rand100_turn30       (turn30; the later max-turn70 run = 49, not in the paper)
RxR-CE    waypoint    39  336  44.2k   claudecode/rxr_wp_fable_rand100_turn30

Appendix B · beyond R2R-CE (SR, SPL)
VLNVerse  fable-5   84   62.47   run dirs never landed here
HM-EQA    fable-5   76.0   —     claudecode/std_hmeqa_cc_fable-5_default_bare        (mip100, was teaps100)

Table 1 · Human row, Appendix D · real robot
Human     one tester, same rand100 board + minimal interface     human/rand100
Go2       Unitree Go2, 23 episodes                               claudecode/std_go2_cc_fable-5_default_bare   (no summary.json)
PERF
