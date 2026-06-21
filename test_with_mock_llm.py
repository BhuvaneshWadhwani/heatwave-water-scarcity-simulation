"""Validate water_scarcity_sim.py's control flow with a deterministic mock
LLM layer — no real API key or network access required. Catches parsing
bugs, conservation-law violations, and crashes before any real (billed) API
calls are made. Also exercises run_batch() across multiple conditions and
seeds, since that's the actual point of the decisions_df/outcomes_df
contract.

NOT a substitute for running the real simulation — the mock responses are
simplistic and don't produce meaningful research content. It only proves
the engine doesn't crash and that the deterministic invariants hold.
"""
import random
import re

import numpy as np
import pandas as pd

import water_scarcity_sim as sim

_call_log = {"chat": 0, "embed": 0}


def mock_embed(text, model="text-embedding-3-small"):
    """Deterministic pseudo-embedding from a hash of the text — enough for
    retrieve()'s cosine math to run; not semantically meaningful."""
    _call_log["embed"] += 1
    h = abs(hash(text)) % (2**32)
    rng = np.random.RandomState(h)
    return rng.normal(size=16)


def mock_llm(prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=400, seed=None):
    _call_log["chat"] += 1
    # Combine seed with prompt content: same (prompt, seed) -> same answer
    # (faithful to a real seeded API call), but different prompts under the
    # same run seed still diverge — unlike a naive Random(seed) per call,
    # which would make every same-shaped prompt in a run return identically.
    combined = hash((seed, prompt)) % (2**32)
    rng = random.Random(combined)

    if "On a scale of 1 to 10" in prompt:
        return f"{rng.randint(2, 9)}. Mock importance rating."

    if "Decide what to formally REQUEST today" in prompt:
        m = re.search(r"baseline operational need today.*?is\s*([\d.]+)\s*units", prompt)
        base = float(m.group(1)) if m else 100.0
        shaded = base * rng.choice([0.95, 1.0, 1.1, 1.25])
        return f"REQUEST: {shaded:.0f}\nARGUMENT: Mock priority argument for this stakeholder."

    if "Choose ONE move:" in prompt:
        has_trade_option = "PROPOSE_TRADE" in prompt
        options = ["ACCEPT", "CONCEDE", "OBJECT"] + (["PROPOSE_TRADE"] if has_trade_option else [])
        choice = rng.choice(options)
        if choice == "CONCEDE":
            m = re.search(r"minimum acceptable:\s*([\d.]+)", prompt)
            cur_min = float(m.group(1)) if m else 50.0
            revised = cur_min * 0.85
            return (f"MOVE: CONCEDE\nDETAIL: Mock concession.\nREVISED_MIN: {revised:.0f}\n"
                    f"TRADE_TARGET: NONE\nTRADE_UNITS: NONE")
        if choice == "PROPOSE_TRADE":
            peer_match = re.search(r"propose a direct trade with: ([\w, ]+)\.", prompt)
            peer = peer_match.group(1).split(",")[0].strip() if peer_match else "industry"
            return (f"MOVE: PROPOSE_TRADE\nDETAIL: Mock trade offer.\nREVISED_MIN: NONE\n"
                    f"TRADE_TARGET: {peer}\nTRADE_UNITS: 10")
        return (f"MOVE: {choice}\nDETAIL: Mock {choice.lower()} response.\nREVISED_MIN: NONE\n"
                f"TRADE_TARGET: NONE\nTRADE_UNITS: NONE")

    if "Write a short (2-3 sentence) public justification" in prompt:
        return "Mock ruling: allocation followed priority order, balancing fairness and stability."

    if "most salient strategic questions" in prompt:
        return "1. Mock question one?\n2. Mock question two?"

    if "high-level strategic insights" in prompt:
        return ("Mock insight number one about strategy. [1, 2]\n"
                "Mock insight number two about cooperation. [2]\n"
                "Mock insight number three about fairness. [1]")

    return "Mock generic response."


sim.llm = mock_llm
sim.embed = mock_embed

print("=== Test 1: single run_simulation() with default RunConfig ===")
decisions_df, outcomes_df, agents = sim.run_simulation(verbose=False)

assert set(["run_id", "condition", "seed", "day", "stakeholder_id"]).issubset(decisions_df.columns)
assert set(["run_id", "condition", "seed", "day", "stakeholder_id", "satisfaction"]).issubset(outcomes_df.columns)
print(f"OK  decisions_df: {decisions_df.shape}, outcomes_df: {outcomes_df.shape}")

# Conservation: total allocated must never exceed supply, per (run, day)
totals = outcomes_df.groupby(["run_id", "day"]).agg(
    supply=("supply", "first"), total_allocated=("total_allocated", "first"))
violations = totals[totals["total_allocated"] > totals["supply"] + 1e-3]
assert len(violations) == 0, f"CONSERVATION VIOLATION:\n{violations}"
print("OK  allocation never exceeds supply")

assert (outcomes_df["allocated"] >= -1e-6).all(), "Negative allocation found"
print("OK  no negative allocations")

mismatched = outcomes_df[(outcomes_df["rounds_today"] > 0) & (~outcomes_df["severity_today"])]
assert len(mismatched) == 0, f"Severity/rounds mismatch:\n{mismatched}"
print("OK  negotiation rounds only occur on days flagged severe")

for sid, a in agents.items():
    assert len(a["stream"]) > 0, f"{sid} has empty memory stream"
print(f"OK  all {len(agents)} agents have non-empty memory streams "
      f"(sizes: {sorted(len(a['stream']) for a in agents.values())})")

per_run_day, per_run_summary = sim.compute_metrics(outcomes_df, decisions_df)
assert (per_run_day["fairness_gini"] >= -1e-6).all() and (per_run_day["fairness_gini"] <= 1 + 1e-6).all()
assert len(per_run_summary) == 1, "Expected exactly one summary row for one run"
print("OK  compute_metrics produced one summary row for the single run:")
for k, v in per_run_summary.iloc[0].items():
    print(f"      {k}: {v}")


print("\n=== Test 2: run_batch() across two conditions x three seeds ===")
configs = [
    sim.RunConfig(condition_label="mild_shortage", demand_multiplier=0.8),
    sim.RunConfig(condition_label="severe_shortage", demand_multiplier=1.3),
]
batch_decisions, batch_outcomes = sim.run_batch(configs, n_seeds=3, base_seed=100)

expected_runs = len(configs) * 3
actual_runs = batch_outcomes["run_id"].nunique()
assert actual_runs == expected_runs, f"Expected {expected_runs} distinct runs, got {actual_runs}"
print(f"OK  run_batch produced {actual_runs} distinct runs as expected")

batch_totals = batch_outcomes.groupby(["run_id", "day"]).agg(
    supply=("supply", "first"), total_allocated=("total_allocated", "first"))
batch_violations = batch_totals[batch_totals["total_allocated"] > batch_totals["supply"] + 1e-3]
assert len(batch_violations) == 0, f"CONSERVATION VIOLATION IN BATCH:\n{batch_violations}"
print("OK  conservation holds across every run in the batch")

batch_per_day, batch_summary = sim.compute_metrics(batch_outcomes, batch_decisions)
assert len(batch_summary) == expected_runs
print(f"OK  compute_metrics produced {len(batch_summary)} summary rows (one per run)")

agg = batch_summary.groupby("condition")[
    ["mean_fairness_gini", "mean_collective_welfare_utilitarian",
     "mean_collective_welfare_rawlsian", "n_critical_failures"]
].agg(["mean", "std"])
print("\nOK  cross-seed aggregation by condition works:")
print(agg)

# severe_shortage should show *at least as many* critical failures on average
# as mild_shortage under this mock (sanity check on the demand_multiplier knob
# actually doing something, not a claim about real LLM behaviour)
mild_failures = batch_summary[batch_summary["condition"] == "mild_shortage"]["n_critical_failures"].mean()
severe_failures = batch_summary[batch_summary["condition"] == "severe_shortage"]["n_critical_failures"].mean()
print(f"\nmean critical failures — mild: {mild_failures}, severe: {severe_failures}")
assert severe_failures >= mild_failures, (
    "severe_shortage condition produced fewer critical failures than mild_shortage — "
    "the demand_multiplier knob in RunConfig may not be wired correctly")
print("OK  demand_multiplier condition knob produces the expected direction of effect")

# Reproducibility: same condition + same seed should be exactly re-runnable
# (within the mock — real LLM determinism depends on the API honouring `seed`,
# which is why every llm() call in the real engine forwards it explicitly)
repeat_decisions, repeat_outcomes = sim.run_batch([configs[0]], n_seeds=1, base_seed=100)
first_run = batch_outcomes[batch_outcomes["run_id"] == "mild_shortage_seed100"].reset_index(drop=True)
repeat_run = repeat_outcomes[repeat_outcomes["run_id"] == "mild_shortage_seed100"].reset_index(drop=True)
compare_cols = ["day", "stakeholder_id", "requested", "allocated", "satisfaction", "critical_failure"]
assert first_run[compare_cols].equals(repeat_run[compare_cols]), (
    "Re-running the same condition+seed produced different outcomes — seed is not "
    "actually threading through to determinism.")
print(f"\nOK  re-running condition={configs[0].condition_label} seed=100 reproduced "
      f"identical outcomes ({len(repeat_run)} rows compared)")

print(f"\nMock LLM calls: {_call_log['chat']} chat, {_call_log['embed']} embed")
print("\nALL CHECKS PASSED")
