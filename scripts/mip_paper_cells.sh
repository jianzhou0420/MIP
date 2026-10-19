#!/usr/bin/env bash
# mip_paper_cells.sh — the runs behind the MIP paper's tables (arXiv:2607.26148), as runner lines.
# A reference, not a program: the first line below exits, so the file never runs as a whole; copy a
# line or a loop. The numbers are in the paper and on the project page.
#
# Seats: harness=cc (Claude Agent SDK), harness=codex (Codex CLI), harness=mini (mini-SWE-agent).
# R2R-CE Minimal rows in the paper are the mean of three runs: repeat a run under its own name with
# run.name=<run name>_rep2, _rep3. The Waypoint / Hybrid cells need the waypoint predictor at
# run.wp_server; the SLAM cells need docker and the ORB-SLAM3 image (INSTALL.md § 6).
echo "reference only — copy one line, do not run the file"; exit 0

CELLS="std_mip_r2r_minimal std_mip_rxr_minimal std_mip_hm3d_minimal std_mip_hm3d_v2_minimal std_mip_mp3d_minimal std_mip_hmeqa_minimal"

# ── Six-benchmark results: the Minimal interface on all six benchmarks, default effort ──
for model in gpt-6-astra gpt-6.1-sol gpt-6-sol gpt-6-luna gpt-5.6 gpt-5.6-luna gpt-5.5; do
  for cell in $CELLS; do python runner.py $cell harness=codex model=$model; done
done
for model in fable-5.1 opus-5.5 opus-5 fable-5 sonnet-5 haiku-5.5; do
  for cell in $CELLS; do python runner.py $cell harness=cc model=$model; done
done
for cell in $CELLS; do python runner.py $cell harness=mini model=qwen3.8-max; done

# ── Interfaces on R2R-CE: each interface against the model's Minimal run ──
for model in opus-5.5 fable-5.1 opus-5 haiku-5.5; do
  for cell in std_mip_r2r_waypoint std_mip_r2r_hybrid std_mip_r2r_slam std_mip_r2r_hybrid_slam; do
    python runner.py $cell harness=cc model=$model
  done
done
for cell in std_mip_r2r_waypoint std_mip_r2r_hybrid std_mip_r2r_hybrid_slam; do
  python runner.py $cell harness=cc model=fable-5
done

# ── Waypoint-only control on R2R-CE: matched models and harnesses ──
python runner.py std_mip_r2r_minimal  harness=mini model=qwen3.5-4b        # qwen3.5-4b / 9b run locally (OLLAMA_URL)
python runner.py std_mip_r2r_waypoint harness=mini model=qwen3.5-4b
python runner.py std_mip_r2r_minimal  harness=mini model=qwen3.5-9b
python runner.py std_mip_r2r_waypoint harness=mini model=qwen3.5-9b
python runner.py std_mip_r2r_minimal  harness=mini model=qwen3.5-plus
python runner.py std_mip_r2r_waypoint harness=mini model=qwen3.5-plus
python runner.py std_mip_r2r_waypoint harness=codex model=gpt-5.5
python runner.py std_mip_r2r_waypoint harness=codex model=gpt-5.6 effort=medium
python runner.py std_mip_r2r_minimal  harness=cc model=opus-4.8
python runner.py std_mip_r2r_waypoint harness=cc model=opus-4.8
python runner.py std_mip_r2r_waypoint harness=cc model=sonnet-5
# fable-5, opus-5, opus-5.5, fable-5.1 and haiku-5.5: the Waypoint runs of "Interfaces on R2R-CE" above

# ── Harnesses on R2R-CE: the same models under mini-SWE-agent (the Native column is the six-benchmark run) ──
for model in gpt-5.5 gpt-5.6 sonnet-5 opus-4.8 fable-5 opus-5; do
  python runner.py std_mip_r2r_minimal harness=mini model=$model
done

# ── Reasoning effort on R2R-CE (appendix): each against its default-effort run ──
for model in sonnet-5 opus-4.8 fable-5 opus-5; do
  python runner.py std_mip_r2r_minimal harness=cc model=$model effort=max
done
for model in gpt-5.5 gpt-5.6; do
  python runner.py std_mip_r2r_minimal harness=codex model=$model effort=xhigh
  python runner.py std_mip_r2r_minimal harness=mini  model=$model effort=xhigh
done

# ── Reasoning cost on R2R-CE (appendix): the Minimal runs above, plus ──
for model in qwen3.6-plus qwen3.7-plus qwen3.8-27b; do
  python runner.py std_mip_r2r_minimal harness=mini model=$model
done
