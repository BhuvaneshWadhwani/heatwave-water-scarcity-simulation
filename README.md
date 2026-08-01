# Coordination Under Scarcity: A Multi-Agent LLM Simulation of Water Allocation During a Heatwave

**Bhuvanesh Dinesh Wadhwani, Anastasiia Khitrova, Artur Zavistovskyi**
University of Konstanz, 2026

---

## Overview

This repository contains the simulation code and analysis notebooks for a multi-agent LLM study of water allocation under heatwave-induced scarcity. Six autonomous LLM-based agents representing real-world stakeholders (hospital, households, agriculture, industry, energy utility, and an environmental regulator) negotiate daily access to a shared, declining water supply managed by a municipal Water Authority across a simulated six-day heatwave in Germany.

The study examines whether mechanically real cascade consequences, where one stakeholder's critical failure raises the operational threshold of dependent stakeholders the following day, change coordination behavior compared to scarcity alone.

---

## Repository Structure

```
.gitignore
LICENSE
README.md
water_scarcity_sim.py                   Core simulation engine
water_scarcity_simulation.ipynb         Batch run and analysis notebook
decisions_20260704_205922.csv           Negotiation decisions log
outcomes_20260704_205922.csv            Allocation outcomes per stakeholder per day
weights_20260704_205922.csv             Authority weight trajectory
zones_20260704_205922.csv              Zone classification per stakeholder per day
water_scarcity_sim_cache/               Pre-cached LLM responses (enables free reproduction)
```


---

## Dataset

The full dataset (decisions, outcomes, weights, zones CSVs) is also publicly available on Kaggle:

https://www.kaggle.com/datasets/bhuvaneshwadhwani/llm-agents-water-allocation-under-scarcity

---

## Experimental Conditions

| Condition | Supply schedule | Cascade consequences |
|---|---|---|
| Baseline | Moderate (1000 to 600 units/day) | Disabled |
| Baseline cascade | Moderate | Enabled |
| Deeper scarcity | Deeper (900 to 460 units/day) | Enabled |

Each condition was run across three random seeds for a total of nine simulation runs.

---

## Requirements
pip install openai numpy pandas scipy matplotlib vaderSentiment

You will also need an OpenAI API key set as the environment variable `OPENAI_API_KEY`.

---

## Reproducing the Results

1. Clone the repository (includes the cache folder `water_scarcity_sim_cache/`)
2. Install dependencies
3. Open `water_scarcity_simulation.ipynb` and run the batch cell

No API key is required to reproduce the published results. All LLM responses
are pre-cached in `water_scarcity_sim_cache/` by (model, prompt, temperature,
seed). The notebook will replay cached responses at zero API cost.

To generate new runs with different seeds, set `OPENAI_API_KEY` as an
environment variable and use seeds not present in the cache.

---

## Key Findings

- Cooperation rates were low but non-zero (0.46 to 0.56) across all conditions
- Zero concessions occurred in any condition: no agent voluntarily lowered its comfortable threshold during negotiation
- Cascade consequences alone did not significantly change cooperation rates or critical failures at equivalent supply levels
- Deeper scarcity produced significantly more critical failures (p = .047) and greater inequality
- A gap between articulated reasoning and behavioral output was observed: agents produced sophisticated interdependence-aware reflections yet still defaulted to objection when their own allocations fell short

---

## Citation

If you use this code or dataset, please cite:

Wadhwani, B. D., Khitrova, A., & Zavistovskyi, A. (2026). Coordination Under Scarcity: A Multi-Agent LLM Simulation of Water Allocation During a Heatwave. University of Konstanz.