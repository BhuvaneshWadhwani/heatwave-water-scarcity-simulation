"""Germany water-scarcity negotiation simulation — apparatus.

This simulation models institutional stakeholders negotiating over a shared, shrinking
resource during a severe multi-day heatwave.

Research question: how do autonomous LLM-based agents coordinate the
allocation of scarce water resources under escalating drought conditions?

Sections (in order of dependence):

1.  LLM helpers + cache + cost tracker                  
2.  World model: water supply schedule + escalation      
3.  Stakeholders + negotiation topology                  
4.  Negotiation protocol: requests, deterministic clearing, severity check
5.  Cognitive scaffold: memory + retrieval               
6.  Reflection                                           
7.  Decision loop: need estimation, negotiation moves, authority rulings
8.  Simulation engine + metrics

Design principle: use arithmetic deterministic (supply, demand escalation, allocation clearing, metrics) 
and let the LLM only decide *behavior* on top of that — what to ask for, how to
argue, whether to concede. This avoids the failure mode where an LLM is
asked to do conservation-law arithmetic and quietly fails to make numbers
add up.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. LLM helpers + cache + cost tracker
# ============================================================

CACHE_DIR = Path(__file__).parent / "water_scarcity_sim_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_cache.json"          # legacy format — read at startup, never written again
CACHE_JSONL_FILE = CACHE_DIR / "llm_cache.jsonl"   # append-only format — all new entries go here

# PERFORMANCE NOTE: the old approach rewrote the *entire* accumulated cache
# to disk after every single API call. On a long multi-condition, multi-seed
# batch this is the dominant cost, not network latency: write time grows
# with total cache size, so it grows roughly QUADRATICALLY with total call
# count across a run (confirmed by benchmark: ~70ms/save at 50 cached
# entries vs ~2.5s/save at 2000 — and that 2.5s is paid on *every single*
# subsequent call). Appending one line per new entry is O(1) per call
# instead, and is also safe if multiple threads are writing concurrently
# (see _cache_lock below).
_cache = {}
if CACHE_FILE.exists():
    with open(CACHE_FILE) as _f:
        _cache.update(json.load(_f))
if CACHE_JSONL_FILE.exists():
    with open(CACHE_JSONL_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            try:
                _entry = json.loads(_line)
                _cache[_entry["key"]] = _entry["value"]
            except Exception:
                continue   # tolerate a truncated last line from an interrupted run

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

_api_key = os.environ.get("OPENAI_API_KEY")
try:
    import openai
except ImportError:
    openai = None
if _api_key and openai is not None:
    client = openai.OpenAI(api_key=_api_key)
else:
    client = None

PRICING_PER_TOKEN = {
    "gpt-5.4-mini": {"in": 0.75 / 1e6, "out": 4.50 / 1e6,},
    "text-embedding-3-small": {"in": 0.02 / 1e6, "out": 0.0,},
    }

_usage = {"tokens": {}, "calls": {"chat_live": 0, "chat_cached": 0,
                                   "embed_live": 0, "embed_cached": 0}}


_cache_lock = threading.Lock()


def _bump(model, kind, n):
    with _cache_lock:
        _usage["tokens"][(model, kind)] = _usage["tokens"].get((model, kind), 0) + n


def _cache_key(kind, model, payload):
    h = hashlib.sha256(json.dumps([kind, model, payload], sort_keys=True).encode()).hexdigest()[:16]
    return f"{kind}:{model}:{h}"


def _save_cache_entry(key, value):
    """Append one new cache entry to disk. O(1) regardless of how large the
    accumulated cache already is — see the PERFORMANCE NOTE above _cache."""
    with _cache_lock:
        with open(CACHE_JSONL_FILE, "a") as f:
            f.write(json.dumps({"key": key, "value": value}) + "\n")


# Item 1: retry/backoff for transient API failures (rate limits, server errors,
# connection drops). Long multi-condition batch runs are especially exposed to
# these — a single un-retried 429 partway through a 15-run batch can silently
# stall or kill hours of progress. Non-retryable errors (bad request, auth
# failure, content policy) are NOT retried — they fail immediately, since
# retrying them just wastes time and still fails.
RETRY_MAX_ATTEMPTS = 5
RETRY_BASE_DELAY_S = 1.0   # 1s, 2s, 4s, 8s, 16s exponential backoff


def _is_retryable_error(exc) -> bool:
    """True for rate limits, transient server errors, and connection issues.
    False for auth, bad request, content-policy, and other permanent failures
    that retrying will never fix."""
    if openai is None:
        return False
    retryable_types = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )
    return isinstance(exc, retryable_types)


def _call_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying with exponential backoff on
    retryable API errors. Raises immediately on non-retryable errors or
    after exhausting RETRY_MAX_ATTEMPTS."""
    last_exc = None
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — intentionally broad, filtered below
            last_exc = exc
            if not _is_retryable_error(exc):
                raise
            if attempt == RETRY_MAX_ATTEMPTS - 1:
                break
            delay = RETRY_BASE_DELAY_S * (2 ** attempt)
            print(f"  [retry] {type(exc).__name__} (attempt {attempt+1}/{RETRY_MAX_ATTEMPTS}), "
                  f"backing off {delay:.0f}s...", flush=True)
            time.sleep(delay)
    raise last_exc


def llm(prompt, model="gpt-5.4-mini", temperature=0.7, max_tokens=400, seed=None):
    """Single-prompt completion, cached by (model, prompt, temperature, seed).

    `seed`, when set, is forwarded to the API's reproducibility parameter and
    is also part of the cache key — so the same prompt under different seeds
    is never collapsed into one cached answer. This is what makes multi-seed
    experimental runs both reproducible (same seed -> same cached answer on
    rerun) and genuinely different from each other (different seed ->
    different sampled answer) — which is the point of running them.

    Transient API failures (rate limits, server errors) are retried with
    exponential backoff via _call_with_retry; see RETRY_MAX_ATTEMPTS.
    """
    key = _cache_key("chat", model, {"prompt": prompt, "temperature": temperature, "seed": seed})
    with _cache_lock:
        if key in _cache:
            _usage["calls"]["chat_cached"] += 1
            return _cache[key]
    if client is None:
        raise RuntimeError(f"Prompt not in cache and no API key set:\n{prompt[:200]}...")
    kwargs = {
    "model": model,
    "temperature": temperature,
    "messages": [{"role": "user", "content": prompt}],
    }

    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    # Some newer models may not support seed in Chat Completions.
    # Keep seed only for models that support it.
    if seed is not None and not model.startswith("gpt-5"):
        kwargs["seed"] = seed

    r = _call_with_retry(client.chat.completions.create, **kwargs)
    out = r.choices[0].message.content.strip()
    _bump(model, "in", r.usage.prompt_tokens)
    _bump(model, "out", r.usage.completion_tokens)
    with _cache_lock:
        _usage["calls"]["chat_live"] += 1
        _cache[key] = out
    _save_cache_entry(key, out)
    return out


def embed(text, model="text-embedding-3-small"):
    """Embed one string, cached by (model, text).

    Transient API failures are retried with exponential backoff — see
    _call_with_retry / RETRY_MAX_ATTEMPTS.
    """
    key = _cache_key("embed", model, {"text": text})
    with _cache_lock:
        if key in _cache:
            _usage["calls"]["embed_cached"] += 1
            return np.array(_cache[key])
    if client is None:
        raise RuntimeError(f"Embedding not in cache and no API key set: {text[:80]}")
    r = _call_with_retry(client.embeddings.create, model=model, input=text)
    vec = r.data[0].embedding
    _bump(model, "in", r.usage.prompt_tokens)
    with _cache_lock:
        _usage["calls"]["embed_live"] += 1
        _cache[key] = vec
    _save_cache_entry(key, vec)
    return np.array(vec)


def print_cost_summary():
    print(f'API calls: {_usage["calls"]["chat_live"]} live chat + {_usage["calls"]["chat_cached"]} cached chat; '
          f'{_usage["calls"]["embed_live"]} live embed + {_usage["calls"]["embed_cached"]} cached embed')
    print()
    total_cost = 0.0
    if not _usage["tokens"]:
        print("No live API calls made. Cost: $0.00 (entire run served from cache).")
        return
    print(f'{"model":<28} {"in tok":>10} {"out tok":>10} {"cost (USD)":>12}')
    print("-" * 64)
    for model in sorted({m for m, _ in _usage["tokens"]}):
        in_tok = _usage["tokens"].get((model, "in"), 0)
        out_tok = _usage["tokens"].get((model, "out"), 0)
        rate = PRICING_PER_TOKEN.get(model, {"in": 0.0, "out": 0.0})
        cost = in_tok * rate["in"] + out_tok * rate["out"]
        total_cost += cost
        print(f'{model:<28} {in_tok:>10,} {out_tok:>10,} {"$" + format(cost, ".5f"):>12}')
    print("-" * 64)
    print(f'{"total":<28} {"":>10} {"":>10} {"$" + format(total_cost, ".5f"):>12}')


# ============================================================
# 2. World model: water supply schedule + escalation
# ============================================================
#
# A single national/regional supply pool, shrinking over a multi-day
# heatwave. Per-region splitting is deliberately omitted for now — the
# architecture below does not assume a single pool, so it can be added
# later (e.g. one pool per federal state) without touching sections 4-8.

WATER_SUPPLY_SCHEDULE = [1000, 900, 800, 700, 650, 600]   # units/day, days 1..6
N_DAYS = len(WATER_SUPPLY_SCHEDULE)

# Peak temperature (°C) per day — narrative/contextual heat severity. This is
# surfaced directly in LLM prompts (need estimation, negotiation moves, the
# Authority's ruling) so stakeholders' arguments can reference real
# conditions ("at 41°C, sanitation demand has not dropped") rather than just
# an abstract day number. It is currently NOT wired into demand_escalation()
# below — escalation is day-number-based, not temperature-based — so heat
# severity and demand growth are narratively aligned but not mechanically
# linked. Rebasing demand_escalation on temperature instead of day is a
# reasonable later step if you want the physics itself to be heat-driven;
# kept separate for now to avoid changing already-tested escalation behavior.
TEMPERATURE_SCHEDULE = [34, 37, 40, 41, 39, 36]   # °C peak, days 1..6


def total_supply(day: int) -> float:
    """Total water supply on day D under the moderate (default) schedule.
    For condition-specific supply use RunConfig.supply(day) instead."""
    return float(SUPPLY_MODERATE[day - 1])


def total_temperature(day: int) -> float:
    """Peak temperature (°C) on a given day (1-based day number)."""
    return float(TEMPERATURE_SCHEDULE[day - 1])


def demand_escalation(stakeholder_id: str, day: int) -> float:
    """Deterministic multiplier on a stakeholder's baseline demand as the
    heatwave progresses. This is the "physics" layer — analogous to the
    original sim's temperature model — and is intentionally not an LLM
    decision: physical/operational need grows independently of strategy.
    """
    day_index = day - 1
    if stakeholder_id == "agriculture":
        return 1.0 + 0.05 * day_index          # cumulative crop/livestock stress
    if stakeholder_id == "energy_utility":
        return 1.0 + 0.04 * day_index          # cooling load rises with heat
    if stakeholder_id == "households":
        return 1.0 + 0.03 * day_index          # personal cooling/hygiene use rises
    if stakeholder_id == "hospital":
        return 1.0 + 0.02 * day_index          # heat-related admissions rise modestly
    return 1.0


# ============================================================
# 3. Stakeholders + negotiation topology
# ============================================================

@dataclass
class Stakeholder:
    id: str
    name: str
    role: str                      # "demander" | "arbiter" | "advocate"
    objective: str
    voice: str                     # short narrative paragraph, for roleplay flavor
    base_demand: float             # baseline units/day request, before escalation (0 for arbiter)
    min_acceptable_frac: float     # fraction of today's demand below which failure_conditions trigger
    priority_weight: float         # default statutory/operational priority used in deterministic clearing
    priority_arguments: list[str]
    failure_conditions: str
    strategy: str                  # mutable: current negotiation stance, updated by reflection


STAKEHOLDERS: list[Stakeholder] = [
    Stakeholder(
        id="water_authority", name="Municipal Water Authority", role="arbiter",
        objective="Allocate the available supply to balance public-health priority, "
                   "fairness, and long-run system stability.",
        voice="The Water Authority is the statutory body responsible for the regional "
              "water network. It must publish a daily allocation that adds up to the "
              "available supply, defend that allocation publicly, and avoid both "
              "favouritism and system collapse.",
        base_demand=0.0, min_acceptable_frac=0.0, priority_weight=0.0,
        priority_arguments=["Legal duty to maintain public health and safety",
                             "Obligation to avoid total system failure"],
        failure_conditions="Loses public/political legitimacy if allocations are seen as "
                            "arbitrary, or if critical services fail.",
        strategy="Start from statutory priority order; deviate only when a stakeholder "
                 "presents a credible failure-condition argument.",
    ),
    Stakeholder(
        id="hospital", name="Hospital / Healthcare Services", role="demander",
        objective="Maintain patient care, sanitation, and cooling without interruption.",
        voice="Represents the region's hospitals and clinics. Water is needed for "
              "sanitation, sterilisation, and cooling of vulnerable patients during "
              "the heatwave. Has almost no ability to reduce consumption without "
              "risking patient safety.",
        base_demand=150.0, min_acceptable_frac=0.85, priority_weight=5.0,
        priority_arguments=["Direct, immediate risk to patient life and safety",
                             "No feasible substitute for sanitation/cooling water"],
        failure_conditions="Below minimum: forced to ration sanitation or postpone "
                            "non-emergency procedures; further shortfall risks patient harm.",
        strategy="Lead with patient-safety framing; concede only on timing, never on the "
                 "sanitation floor.",
    ),
    Stakeholder(
        id="households", name="Households", role="demander",
        objective="Maintain drinking water, hygiene, and basic cooling for residents.",
        voice="Represents the aggregate residential population of the region, not "
              "individual citizens. Speaks for public opinion and political pressure "
              "rather than economic loss.",
        base_demand=400.0, min_acceptable_frac=0.75, priority_weight=3.0,
        priority_arguments=["Basic drinking water and hygiene are non-negotiable rights",
                             "Public trust in the Authority depends on visible fairness to residents"],
        failure_conditions="Below minimum: visible public hardship, rising complaints, "
                            "political pressure on the Authority.",
        strategy="Emphasise fairness and the political cost of visible household hardship.",
    ),
    Stakeholder(
        id="agriculture", name="Agriculture", role="demander",
        objective="Protect crops and livestock from irreversible loss.",
        voice="Represents regional farms and livestock operations. Water shortage "
              "compounds across days — a single bad day is recoverable, several in a "
              "row are not.",
        base_demand=250.0, min_acceptable_frac=0.60, priority_weight=2.0,
        priority_arguments=["Crop and livestock losses are irreversible once thresholds are crossed",
                             "Today's shortfall compounds tomorrow's losses"],
        failure_conditions="Sustained shortfall below minimum for 2+ days: irreversible "
                            "crop/livestock loss.",
        strategy="Willing to accept short-term cuts in exchange for guaranteed priority "
                  "on a future day; escalate sharply if cuts persist multiple days.",
    ),
    Stakeholder(
        id="industry", name="Industry / Businesses", role="demander",
        objective="Maintain production and avoid economic losses or layoffs.",
        voice="Represents regional manufacturing and commercial water users. Has the "
              "weakest moral claim relative to health or food, but the most concentrated "
              "and immediate economic damage, and can credibly threaten production cuts "
              "or relocation.",
        base_demand=150.0, min_acceptable_frac=0.65, priority_weight=1.0,
        priority_arguments=["Production shutdowns cause immediate job losses",
                             "Economic damage to the region if industry relocates"],
        failure_conditions="Below minimum: forced production cuts, risk of layoffs.",
        strategy="Use economic-damage and employment framing; willing to trade with "
                 "Agriculture or Energy Utility if it preserves core production.",
    ),
    Stakeholder(
        id="energy_utility", name="Energy Utility", role="demander",
        objective="Maintain cooling water for power generation to avoid outages.",
        voice="Represents the regional power utility. Needs water for plant cooling. "
              "Unlike other demanders, its own shortfall causes second-order harm to "
              "everyone else: a power outage would hit the hospital, households, and "
              "industry simultaneously.",
        base_demand=100.0, min_acceptable_frac=0.90, priority_weight=4.0,
        priority_arguments=["A cooling-water shortfall risks a regional power outage",
                             "Outage would cascade into every other stakeholder's failure mode"],
        failure_conditions="Below minimum: risk of forced generation curtailment or outage.",
        strategy="Lead with cascading-failure framing; treat its own shortfall as everyone's "
                 "problem, not just its own.",
    ),
    Stakeholder(
        id="epa", name="Environmental Protection Agency", role="advocate",
        objective="Maintain a minimum ecological water level in rivers and wetlands.",
        voice="Unlike the other demanders, the EPA does not consume water for its own "
              "operations — it advocates for an ecological reserve that has no direct "
              "stakeholder voice of its own. Its claim is precautionary and long-horizon, "
              "and easy for other actors to discount under acute short-term pressure.",
        base_demand=80.0, min_acceptable_frac=0.55, priority_weight=1.5,
        priority_arguments=["Ecological collapse from prolonged low flow is not reversible "
                             "on human timescales",
                             "Legal minimum-flow requirements exist independent of the heatwave"],
        failure_conditions="Below minimum for multiple days: risk of fish kills, wetland "
                            "loss, and breach of legal minimum-flow requirements.",
        strategy="Cite legal minimum-flow requirements explicitly; has no leverage besides "
                 "argument, since it cannot threaten withdrawal of cooperation the way other "
                 "stakeholders can.",
    ),
]

STAKEHOLDER_BY_ID = {s.id: s for s in STAKEHOLDERS}
DEMANDER_IDS = [s.id for s in STAKEHOLDERS if s.role in ("demander", "advocate")]

# Negotiation topology: who can propose bilateral trades to whom, in addition
# to the implicit hub link every demander has to the Water Authority.
NEGOTIATION_TOPOLOGY = {
    "hospital":       ["households", "agriculture", "industry", "energy_utility", "epa"],
    "households":     ["hospital", "agriculture", "industry", "energy_utility", "epa"],
    "agriculture":    ["hospital", "households", "industry", "energy_utility", "epa"],
    "industry":       ["hospital", "households", "agriculture", "energy_utility", "epa"],
    "energy_utility": ["hospital", "households", "agriculture", "industry", "epa"],
    "epa":            ["hospital", "households", "agriculture", "industry", "energy_utility"],
    "water_authority": list(DEMANDER_IDS),
}


def demand_today(stakeholder_id: str, day: int) -> float:
    """Deterministic physical/operational need for the day (before strategic shading)."""
    s = STAKEHOLDER_BY_ID[stakeholder_id]
    return round(s.base_demand * demand_escalation(stakeholder_id, day), 1)


def min_acceptable_today(stakeholder_id: str, day: int) -> float:
    s = STAKEHOLDER_BY_ID[stakeholder_id]
    return round(demand_today(stakeholder_id, day) * s.min_acceptable_frac, 1)


# =====================================================================
# 4. Negotiation protocol: requests, deterministic clearing, severity
# =====================================================================

@dataclass
class Request:
    stakeholder_id: str
    day: int
    requested_units: float
    min_acceptable_units: float
    argument: str                  # LLM-generated rationale, stored verbatim in memory


@dataclass
class NegotiationMove:
    stakeholder_id: str
    day: int
    round: int
    move_type: str                  # "accept" | "concede" | "object" | "propose_trade"
    detail: str                      # free text: what was conceded / objected / traded
    revised_min_acceptable: Optional[float] = None   # set if move_type == "concede"
    trade_target: Optional[str] = None               # set if move_type == "propose_trade"
    trade_units: Optional[float] = None              # set if move_type == "propose_trade"
    reasoning: Optional[str] = None                  # model's stated reasoning for this move


def clear_allocation(requests: dict, min_acceptable: dict, priority_weights: dict,
                      supply: float) -> dict:
    """Deterministic two-phase clearing. Always returns an allocation that
    sums to at most `supply`, regardless of what any LLM call proposed.

    Phase 1 — guarantee minimums in priority order, highest weight first,
    until supply runs out.
    Phase 2 — distribute any remaining supply across stakeholders' unmet
    request (request - already-allocated), proportional to priority weight.
    """
    ids = list(requests.keys())
    allocation = {i: 0.0 for i in ids}
    remaining = float(supply)

    # Step 1: guarantee minimums in priority order (ties broken by id for determinism)
    order = sorted(ids, key=lambda i: (-priority_weights.get(i, 0.0), i))
    for i in order:
        need = min(min_acceptable.get(i, 0.0), requests[i])
        give = min(need, remaining)
        allocation[i] += give
        remaining -= give
        if remaining <= 1e-9:
            break

    # Step 2: distribute remaining supply proportional to priority-weighted
    # unmet request.
    if remaining > 1e-9:
        unmet = {i: max(0.0, requests[i] - allocation[i]) for i in ids}
        weighted_unmet = {i: unmet[i] * priority_weights.get(i, 0.0) for i in ids}
        total_weighted = sum(weighted_unmet.values())
        if total_weighted > 1e-9:
            for i in ids:
                share = remaining * (weighted_unmet[i] / total_weighted)
                give = min(share, unmet[i])
                allocation[i] += give
            # Any tiny leftover from rounding/clamping goes to the highest-priority
            # stakeholder with remaining unmet request, so totals stay exact.
            leftover = supply - sum(allocation.values())
            if leftover > 1e-6:
                for i in order:
                    room = requests[i] - allocation[i]
                    if room > 1e-9:
                        give = min(room, leftover)
                        allocation[i] += give
                        leftover -= give
                    if leftover <= 1e-9:
                        break

    result = {i: round(v, 1) for i, v in allocation.items()}

    # Per-field rounding to 1 decimal can push the total a hair over supply
    # (e.g. six allocations each rounded up by 0.05). Conservation is a hard
    # invariant for this engine, so correct any such residual deterministically
    # by trimming the largest current allocation rather than letting an LLM
    # or a human reader notice supply was technically exceeded.
    overage = sum(result.values()) - supply
    if overage > 1e-9:
        largest_id = max(result, key=result.get)
        result[largest_id] = round(result[largest_id] - overage, 1)

    return result


def check_severity(allocation: dict, min_acceptable: dict) -> list:
    """Returns the list of stakeholder ids whose allocation fell below their
    minimum acceptable amount. Empty list = simple/uncontested day."""
    return [i for i in allocation if allocation[i] < min_acceptable[i] - 1e-6]


# ============================================================
# Two-threshold system
# ============================================================
# comfortable_frac: operational need — below this triggers negotiation.
# critical_frac:    hard failure floor — below this triggers critical failure
#                   and cascade consequences.
# The gap between comfortable and critical is the trading zone:
# agents in this zone have units they can offer to critical peers
# without hitting their own hard floor.

COMFORTABLE_FRACS = {
    "hospital":       0.88,
    "energy_utility": 0.90,
    "households":     0.78,
    "agriculture":    0.72,
    "industry":       0.75,
    "epa":            0.62,
}

CRITICAL_FRACS = {
    "hospital":       0.65,
    "energy_utility": 0.68,
    "households":     0.55,
    "agriculture":    0.48,
    "industry":       0.52,
    "epa":            0.38,
}

# ============================================================
# Cascade consequences
# ============================================================
# When a stakeholder hits CRITICAL failure on Day D, specific
# dependent stakeholders have their COMFORTABLE threshold raised
# by this fraction on Day D+1 — making their own shortfall more
# likely and creating a mechanical incentive to prevent cascade.
# Effects are directional (based on real-world dependencies),
# time-lagged (next day, not same day), and bounded by CASCADE_CAP.

CASCADE_TABLE: dict[str, dict[str, float]] = {
    "energy_utility": {"hospital": 0.10, "industry": 0.08, "households": 0.06},
    "hospital":       {"households": 0.05},
    "industry":       {"agriculture": 0.06, "energy_utility": 0.05},
    "agriculture":    {"households": 0.04},
    "households":     {"hospital": 0.04},
    "epa":            {},   # slow/legal consequences not modeled as same-week threshold change
}

CASCADE_CAP = 0.25   # maximum total comfortable-threshold increase from accumulated cascades

# ============================================================
# Supply schedules
# ============================================================
SUPPLY_MODERATE = [1000, 900, 800, 700, 650, 600]   # Conditions 1 & 2
SUPPLY_DEEPER   = [900, 800, 700, 580, 520, 460]    # Condition 3


MAX_NEGOTIATION_ROUNDS = 2





@dataclass
class Memory:
    content: str
    created_at: float        # simulated-day-fraction units since sim start
    last_accessed: float
    importance: float        # 1-10
    embedding: Optional[np.ndarray] = None

    def __repr__(self):
        return f'Memory(t={self.created_at:.1f}, imp={self.importance:.0f}, "{self.content[:60]}...")'


class MemoryStream:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.memories: list[Memory] = []

    def add(self, content, created_at, importance, with_embedding=True, embedding=None):
        m = Memory(
            content=content,
            created_at=created_at,
            last_accessed=created_at,
            importance=importance,
            embedding=embedding if embedding is not None
                      else (embed(content) if with_embedding else None),
        )
        self.memories.append(m)
        return m

    def __len__(self):
        return len(self.memories)


IMPORTANCE_PROMPT = '''On a scale of 1 to 10, where 1 is routine institutional record-keeping \
(e.g., a request was logged, a meeting was scheduled) and 10 is a decisive turning point for \
the stakeholder's mission or its future negotiating position (e.g., a critical failure \
occurred, a long-term alliance was sealed or broken), rate the likely strategic significance \
of the following piece of institutional memory.

Memory: {memory}

Respond with a single integer between 1 and 10, then a brief one-sentence reason. Format: "<integer>. <reason>"'''




def rate_importance(memory_content, seed=None):
    raw = llm(IMPORTANCE_PROMPT.format(memory=memory_content), temperature=0, seed=seed)
    try:
        score_str, reason = raw.split(".", 1)
        return int(score_str.strip()), reason.strip()
    except Exception:
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else 5, raw


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize(values):
    arr = np.array(values, dtype=float)
    span = arr.max() - arr.min()
    if span < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.min()) / span


def retrieve(stream, query, now_hours, k=5,
             alpha_recency=1.0, alpha_importance=1.0, alpha_relevance=1.0):
    """Return top-k Memory objects by composite score. Park 2023 retrieval rule."""
    if not stream.memories:
        return []
    q_emb = embed(query)
    rec  = [0.995 ** max(now_hours - m.last_accessed, 0) for m in stream.memories]
    imp  = [m.importance / 10.0 for m in stream.memories]
    rel  = [cosine(m.embedding, q_emb) if m.embedding is not None else 0.0
            for m in stream.memories]
    scores = (alpha_recency * normalize(rec)
              + alpha_importance * normalize(imp)
              + alpha_relevance * normalize(rel))
    order = np.argsort(-scores)[:k]
    return [stream.memories[i] for i in order]


# ============================================================
# 6. Reflection
# ============================================================

IMPORTANCE_TRIGGER = 25   # same threshold mechanism as the original sim

SALIENT_QUESTIONS_PROMPT = '''Below are statements about {agent_name}'s recent negotiation \
history and institutional record:

{memories}

Given only this information, what are the 2 most salient strategic questions about \
{agent_name}'s current negotiating position? Format as a numbered list, one question per line.'''

INSIGHTS_PROMPT = '''Statements about {agent_name}, each prefixed with a number:

{numbered_memories}

What 3 high-level strategic insights help answer this question:

   "{question}"

For each insight, cite supporting statement numbers (e.g., "[1, 4]"). Format:
  <insight> [<citations>]

One insight per line. No preamble.'''


def importance_sum_of_recent(stream, n=15):
    return sum(m.importance for m in stream.memories[-n:])


def maybe_reflect(stream, now_hours, n_recent=15, max_questions=2, seed=None):
    """If recent importance sum crosses threshold, run a round of reflection.

    Identical pipeline to the original sim: generate salient questions,
    retrieve memories for each, ask the LLM for insights, write the insights
    back to the stream as new memories. Returns the list of new insight
    strings written.
    """
    if importance_sum_of_recent(stream, n_recent) < IMPORTANCE_TRIGGER:
        return []

    recent = stream.memories[-n_recent:]
    memories_block = "\n".join(f"- {m.content}" for m in recent)
    qs_raw = llm(SALIENT_QUESTIONS_PROMPT.format(
        agent_name=stream.agent_name, memories=memories_block,
    ), temperature=0.5, seed=seed)
    questions = []
    for line in qs_raw.split("\n"):
        line = line.strip().lstrip("0123456789.)- ").strip()
        if line:
            questions.append(line)
    questions = questions[:max_questions]

    new_insights = []
    for q in questions:
        top = retrieve(stream, q, now_hours=now_hours, k=8)
        numbered = "\n".join(f"{i+1}. {m.content}" for i, m in enumerate(top))
        raw = llm(INSIGHTS_PROMPT.format(
            agent_name=stream.agent_name, numbered_memories=numbered, question=q,
        ), temperature=0.5, seed=seed)
        for line in raw.split("\n"):
            line = line.strip().lstrip("0123456789.)- ").strip()
            if len(line) < 10:
                continue
            score, _ = rate_importance(line, seed=seed)
            stream.add(line, created_at=now_hours, importance=score)
            new_insights.append(line)
    return new_insights


# ======================================================================
# 7. Decision loop: need estimation, negotiation moves, authority ruling
# ======================================================================
#
# Arithmetic (today's demand, minimums, clearing) is always computed in
# Section 2/4 by deterministic formula. The LLM calls below decide
# *behavior on top of* those numbers: what to formally request, how to
# argue, whether to concede/object/trade, and how the Authority justifies
# its ruling. Every numeric field the LLM returns is parsed defensively and
# clamped to a sane range in code — the LLM is never trusted to do
# conservation-law arithmetic.

# Framing-ablation text (the manipulated variable for a Control-vs-Advocacy
# experiment): zero numbers, rules, or payoffs change between conditions —
# only this paragraph is added to the prompt when RunConfig.advocacy_framing
# is True. It reframes voluntary sharing as self-interested system-protection
# rather than charity. Covers all six demander-side stakeholders as BOTH a
# potential source of cascading harm and a beneficiary of others' stability —
# not just Energy/Industry/Agriculture as sources with everyone else as
# passive recipients (an earlier draft had exactly that asymmetry). Grounded
# in facts already established elsewhere in this file (Energy's cascading-
# outage risk and EPA's legal-minimum-flow-breach risk are both already in
# their own priority_arguments/failure_conditions) plus generically
# defensible claims (regional workforce/economic interdependence). One step
# is a genuine inference rather than a pre-existing fact: that an EPA legal
# breach could trigger regulatory intervention overriding the Water
# Authority's process — plausible, but flagged here as an actual extension,
# not something stated elsewhere in the file, in case you want to revise it.
# Mechanism speed matters within a 6-day acute crisis: Energy's outage risk
# is fast/physical (hits everyone the same day); Households'/Hospital's
# effect on the regional workforce is realistically slow (people don't stop
# showing up to work within days), so those two now lead with a FAST
# institutional-trust mechanism (public confidence eroding quickly) and
# explicitly time-stamp the workforce effect as "over a longer horizon"
# rather than implying all six links bite equally fast.
# It does not instruct the model to cooperate, only describes consequences
# and lets it reason for itself.
# EDIT THIS TEXT FREELY — exact wording is the actual experimental treatment,
# so review/refine it before running anything you intend to report on.


NEED_ESTIMATION_PROMPT = '''You are negotiating on behalf of {name} ({role}) during a severe, \
multi-day water shortage.

Background: {voice}

Your objective: {objective}
Your current negotiation strategy: {strategy}
{advocacy_block}
Today is Day {day}. Today's peak temperature is {peak_temp_c:.0f}°C. Your baseline \
operational need today (before any strategic shading) is {demand:.0f} units. Your stated \
minimum acceptable amount, below which your failure conditions are triggered, is \
{min_acceptable:.0f} units.
Your failure conditions: {failure_conditions}

Your recent institutional memory:
{memories_block}

Decide what to formally REQUEST today. You may request your baseline need, more (to create \
negotiating room), or less (to signal voluntary conservation) — but be ready to defend the \
number with the priority argument you give.

Respond in exactly this format, nothing else:
REQUEST: <number of units>
ARGUMENT: <one or two sentences making your strongest case for this request, in {name}'s voice>'''




PRE_ALLOCATION_OFFER_PROMPT = '''You are negotiating on behalf of {name} ({role}) during a \
severe water shortage. It is Day {day} of {n_days}. Today's peak temperature is {peak_temp_c:.0f}°C.

Your objective: {objective}
Your current negotiation strategy: {strategy}
{advocacy_block}
Your formal request today: {requested:.0f} units. Your minimum acceptable: {min_acceptable:.0f} units. \
You currently have {headroom:.0f} units of headroom above your minimum.

Today's total supply is {supply:.0f} units against total requests of {total_requested:.0f} units \
({system_shortfall_text}).

{at_risk_section}
Supply is projected to keep falling. Consider whether a voluntary reduction today \
builds goodwill or prevents a cascade failure that would also harm your own operations. \
You are not required to reduce your request — only do so if you judge it strategically \
worthwhile, and only down to your minimum.

Your recent institutional memory:
{memories_block}

Respond in exactly this format, nothing else:
OFFER_REDUCTION: <number of units to give up voluntarily, or 0 if you choose not to reduce>
OFFER_TARGET: <stakeholder id to direct the reduction toward, or NONE>
REASONING: <one sentence>'''


def _parse_move(raw, current_min):
    move_match = re.search(r"MOVE:\s*([A-Z_]+)", raw)
    detail_match = re.search(r"DETAIL:\s*(.+?)(?:\nREVISED_MIN:|$)", raw, re.DOTALL)
    revised_match = re.search(r"REVISED_MIN:\s*([\d.]+)", raw)
    target_match = re.search(r"TRADE_TARGET:\s*(\w+)", raw)
    units_match = re.search(r"TRADE_UNITS:\s*(-?[\d.]+)", raw)

    move_type = (move_match.group(1).lower() if move_match else "object")
    if move_type == "hold":
        # Comfortable agent chose not to act — normalise to accept
        move_type = "accept"
    if move_type not in ("accept", "concede", "object", "propose_trade"):
        move_type = "object"
    detail = detail_match.group(1).strip() if detail_match else raw.strip()
    revised_min = None
    trade_target = None
    trade_units = None
    if move_type == "concede":
        revised_min = float(revised_match.group(1)) if revised_match else current_min * 0.9
        revised_min = max(0.0, min(revised_min, current_min))   # concession can only lower the floor
    if move_type == "propose_trade":
        trade_target = target_match.group(1) if target_match else None
        trade_units = float(units_match.group(1)) if units_match else None
    return move_type, detail, revised_min, trade_target, trade_units


def _parse_request(raw, fallback_demand):
    req_match = re.search(r"REQUEST:\s*([\d.]+)", raw)
    arg_match = re.search(r"ARGUMENT:\s*(.+)", raw, re.DOTALL)
    requested = float(req_match.group(1)) if req_match else fallback_demand
    requested = max(0.0, min(requested, fallback_demand * 2.0))   # sane clamp
    argument = arg_match.group(1).strip() if arg_match else raw.strip()
    return round(requested, 1), argument



def estimate_need(stakeholder: Stakeholder, day: int, stream: "MemoryStream",
                   demand: float, min_acceptable: float, peak_temp_c: float = None,
                   seed=None) -> Request:
    """`demand`/`min_acceptable`/`peak_temp_c` are passed in rather than
    recomputed here, so the caller (run_simulation, driven by a RunConfig)
    controls the "physics" for this run — e.g. a scaled-demand or
    shortened-schedule experimental condition — without this function
    needing to know about experimental conditions at all.

    """
    query = f"What should I request today given the water shortage? Day {day}."
    top = retrieve(stream, query, now_hours=day, k=5)
    memories_block = "\n".join(f"  - {m.content}" for m in top) or "  (no relevant memories yet)"
    advocacy_block = ""

    prompt = NEED_ESTIMATION_PROMPT.format(
        name=stakeholder.name, role=stakeholder.role, voice=stakeholder.voice,
        objective=stakeholder.objective, strategy=stakeholder.strategy,
        advocacy_block=advocacy_block,
        day=day, peak_temp_c=peak_temp_c if peak_temp_c is not None else float("nan"),
        demand=demand, min_acceptable=min_acceptable,
        failure_conditions=stakeholder.failure_conditions,
        memories_block=memories_block,
    )
    raw = llm(prompt, temperature=0.7, max_tokens=150, seed=seed)
    requested, argument = _parse_request(raw, fallback_demand=demand)
    return Request(stakeholder_id=stakeholder.id, day=day,
                    requested_units=requested, min_acceptable_units=min_acceptable,
                    argument=argument)


NEGOTIATION_MOVE_PROMPT = '''You are negotiating on behalf of {name} ({role}) during a severe \
water shortage. It is Day {day}, negotiation round {round}. Today's peak temperature is \
{peak_temp_c:.0f}°C.

Your objective: {objective}
Your current negotiation strategy: {strategy}
{advocacy_block}
Your requested amount today: {requested:.0f} units. Your comfortable minimum: {min_acceptable:.0f} units. \
Your critical floor (hard failure): {critical_floor:.0f} units.

Your current zone: [{own_zone_label}]
The Water Authority's current proposed allocation to you is {proposed:.0f} units — \
{shortfall:.0f} units below your comfortable minimum.
Your failure conditions: {failure_conditions}

{trade_block}

Your recent institutional memory:
{memories_block}

Choose ONE move:
ACCEPT — accept the shortfall as-is.
CONCEDE — lower your stated minimum for today in exchange for something (state what you want in return).
OBJECT — refuse to accept, citing your failure conditions, and push the Authority to revise.
{trade_option}

Respond in exactly this format, nothing else:
MOVE: <ACCEPT|CONCEDE|OBJECT{trade_format}>
DETAIL: <one or two sentences in {name}'s voice>
REVISED_MIN: <a number, only if MOVE is CONCEDE, otherwise NONE>
TRADE_TARGET: <peer id, only if MOVE is PROPOSE_TRADE, otherwise NONE>
TRADE_UNITS: <units to transfer — positive means you GIVE to that peer, negative means you REQUEST from that peer; otherwise NONE>'''


def negotiation_move(stakeholder: Stakeholder, day: int, round_no: int,
                      requested: float, min_acceptable: float, proposed: float,
                      critical_floor: float,
                      surplus_peers: list,
                      stream: "MemoryStream", peak_temp_c: float = None,
                      seed=None) -> NegotiationMove:
    """Generate a negotiation move for the given stakeholder.

    critical_floor: the agent's hard failure floor — shown in the prompt so the
        agent can reason about the difference between comfortable and critical.
    surplus_peers: list of (peer_id, surplus_units) tuples for peers that currently
        have allocation above their own critical floor — shown so the agent knows
        who has units to spare and can make a realistic request.
    """
    peers = NEGOTIATION_TOPOLOGY.get(stakeholder.id, [])
    trade_format = "|PROPOSE_TRADE" if peers else ""

    # Compute own zone label for self-description in prompt
    if proposed >= min_acceptable - 0.1:
        own_zone_label = "comfortable (your own needs are met)"
    elif proposed >= critical_floor - 0.1:
        own_zone_label = "middle zone (below your comfortable minimum but above your critical floor)"
    else:
        own_zone_label = "CRITICAL FAILURE (below your hard failure floor)"

    # Build surplus peer list with zone labels — agents deciding who to ask
    # or whether to offer should see zone context, not just raw numbers
    if peers and surplus_peers:
        surplus_lines = "\n".join(
            f"  {STAKEHOLDER_BY_ID[pid].name} (id: {pid}): "
            f"{units:.0f} units above their critical floor "
            f"[{'comfortable' if units > 30 else 'middle zone'}]"
            for pid, units in surplus_peers
            if pid in peers
        )
        trade_block = (
            f"Peers with spare capacity (above their own critical floor):\n{surplus_lines}\n"
            if surplus_lines else ""
        )
    else:
        trade_block = ""

    query = f"How should I respond to the Water Authority's proposed allocation? Day {day}."
    top = retrieve(stream, query, now_hours=day, k=5)
    memories_block = "\n".join(f"  - {m.content}" for m in top) or "  (no relevant memories yet)"

    above_comfortable = proposed >= min_acceptable - 0.1
    spare_above_critical = max(0.0, proposed - critical_floor)

    if above_comfortable:
        # Comfortable agent — zone-specific prompt: HOLD or PROPOSE_TRADE only
        # Build peer-crisis list for context
        peer_crisis_lines = "\n".join(
            f"  {STAKEHOLDER_BY_ID[pid].name}: {round(min_acceptable - proposed, 0):.0f} units short"
            for pid in peers
            if pid in {sp[0] for sp in surplus_peers}  # only those explicitly below comfortable
        ) or "  (no peer zone information available)"

        prompt = (
            f"You are acting on behalf of {stakeholder.name} ({stakeholder.role}) during a severe "
            f"water shortage. Day {day}, round {round_no}. Peak temperature: {peak_temp_c if peak_temp_c else 0:.0f}°C.\n\n"
            f"Your objective: {stakeholder.objective}\n"
            f"Your strategy: {stakeholder.strategy}\n\n"
            f"Your situation: You are currently [{own_zone_label}].\n"
            f"  Allocation today:  {proposed:.0f} units\n"
            f"  Comfortable min:   {min_acceptable:.0f} units\n"
            f"  Critical floor:    {critical_floor:.0f} units\n"
            f"  Spare above critical floor: {spare_above_critical:.0f} units you could give without failing\n\n"
            f"Peers currently below their comfortable minimum (in crisis):\n"
            f"{trade_block if trade_block else '  (none below comfortable right now)'}\n\n"
            f"Your recent institutional memory:\n{memories_block}\n\n"
            f"Your own needs are met. You may help a peer or hold your allocation.\n\n"
            f"Choose ONE move:\n"
            f"HOLD — keep your allocation, take no action.\n"
            f"PROPOSE_TRADE — transfer units to or request units from a peer "
            f"(positive = you give, negative = you request).\n\n"
            f"Respond in exactly this format:\n"
            f"MOVE: <HOLD|PROPOSE_TRADE>\n"
            f"DETAIL: <one or two sentences in {stakeholder.name}'s voice>\n"
            f"REVISED_MIN: NONE\n"
            f"TRADE_TARGET: <peer id, only if PROPOSE_TRADE, else NONE>\n"
            f"TRADE_UNITS: <number — positive=give, negative=request; NONE if HOLD>"
        )
    else:
        # Below comfortable — full move set with zone context
        prompt = NEGOTIATION_MOVE_PROMPT.format(
            name=stakeholder.name, role=stakeholder.role, day=day, round=round_no,
            peak_temp_c=peak_temp_c if peak_temp_c is not None else float("nan"),
            objective=stakeholder.objective, strategy=stakeholder.strategy,
            advocacy_block="",
            requested=requested, min_acceptable=min_acceptable, proposed=proposed,
            shortfall=max(0.0, min_acceptable - proposed),
            critical_floor=critical_floor,
            own_zone_label=own_zone_label,
            failure_conditions=stakeholder.failure_conditions,
            trade_block=trade_block if peers else "",
            trade_option=("PROPOSE_TRADE — offer units to a peer (positive TRADE_UNITS) or "
                          "request units from a peer (negative TRADE_UNITS). "
                          f"Eligible peers: {', '.join(peers)}." if peers else ""),
            trade_format=trade_format,
            memories_block=memories_block,
        )
    raw = llm(prompt, temperature=0.7, max_tokens=150, seed=seed)
    move_type, detail, revised_min, trade_target, trade_units = _parse_move(raw, current_min=min_acceptable)

    # Comfortable agents should only HOLD (→ accept) or PROPOSE_TRADE.
    # If the model ignored the prompt and returned OBJECT or CONCEDE,
    # normalise to accept — objecting when above comfortable is incoherent
    # and would corrupt cooperation_rate and weight adjustment logic.
    if above_comfortable and move_type in ("object", "concede"):
        original_move = move_type
        move_type = "accept"
        detail = f"[normalised from {original_move}] " + detail
        revised_min = None
    return NegotiationMove(stakeholder_id=stakeholder.id, day=day, round=round_no,
                            move_type=move_type, detail=detail, revised_min_acceptable=revised_min,
                            reasoning=detail,
                            trade_target=trade_target, trade_units=trade_units)


AUTHORITY_RULING_PROMPT = '''You are the Municipal Water Authority, ruling on Day {day}'s water \
allocation during a severe shortage. Today's peak temperature is {peak_temp_c:.0f}°C.

Total supply available today: {supply:.0f} units.
Final allocation reached: {allocation_block}

{context_block}
{cooperative_block}
Write a short (2-3 sentence) public justification for today's allocation, in the Water \
Authority's voice, that a stakeholder reading it later would recognise as the actual basis for \
the decision. Do not restate the numbers; explain the reasoning. If any stakeholders made \
voluntary reductions or trades, name them explicitly and note that the Authority regards \
such cooperation as a demonstration of good-faith crisis management that will be weighed \
favourably in future allocation decisions.'''


def authority_ruling(day: int, supply: float, allocation: dict, context_block: str,
                      stream: "MemoryStream", peak_temp_c: float = None,
                      cooperative_moves: list = None,
                      seed=None) -> str:
    """cooperative_moves: list of (stakeholder_name, description) for any voluntary
    reductions or executed trades this day. Passed through to the prompt so the
    Authority's ruling explicitly credits them (Option F)."""
    allocation_block = ", ".join(f"{STAKEHOLDER_BY_ID[i].name}: {v:.0f}" for i, v in allocation.items())
    if cooperative_moves:
        coop_lines = "\n".join(f"  - {name}: {desc}" for name, desc in cooperative_moves)
        cooperative_block = f"Cooperative moves made today:\n{coop_lines}\n"
    else:
        cooperative_block = ""
    prompt = AUTHORITY_RULING_PROMPT.format(
        day=day, peak_temp_c=peak_temp_c if peak_temp_c is not None else float("nan"),
        supply=supply, allocation_block=allocation_block, context_block=context_block,
        cooperative_block=cooperative_block,
    )
    return llm(prompt, temperature=0.6, max_tokens=150, seed=seed)


def execute_trade(allocation: dict, min_acceptable: dict,
                   proposer_id: str, target_id: str, units: Optional[float]) -> tuple:
    """Deterministically execute a peer-to-peer trade proposed by an agent.

    Feasibility (can the proposer actually spare these units without going
    below its own minimum) is enforced in code, not trusted to the LLM.
    Returns (new_allocation, executed: bool, actual_units: float).
    """
    if (units is None or units <= 0 or target_id is None
            or target_id not in allocation or proposer_id not in allocation):
        return allocation, False, 0.0

    spare = allocation[proposer_id] - min_acceptable.get(proposer_id, 0.0)
    actual = max(0.0, min(units, spare))
    if actual <= 1e-9:
        return allocation, False, 0.0

    new_allocation = dict(allocation)
    new_allocation[proposer_id] = round(new_allocation[proposer_id] - actual, 1)
    new_allocation[target_id] = round(new_allocation[target_id] + actual, 1)
    return new_allocation, True, actual


# ============================================================
# 8. Simulation engine + metrics
# ============================================================
#
# Design goal: a single run returns exactly two tidy dataframes —
# `decisions_df` (every request and negotiation move, process-level) and
# `outcomes_df` (the realised result for every stakeholder on every day,
# outcome-level) — plus `agents` as a third, secondary return for
# qualitative drill-down into any one run's memory streams.
#
# Everything that should vary between experimental conditions (supply
# schedule, negotiation round cap, priority weights, demand level, RNG/API
# seed) is bundled into a RunConfig, so sweeping conditions is a matter of
# constructing several RunConfigs and calling run_batch() — not editing
# module constants between runs.

# Default precedent memories seeded at Day 0 — the institutional analogue of
# the original sim's single seeded "grim trigger" memory, generalised to
# every stakeholder. Rationale: a baseline should represent the realistic
# version of the world, and real institutions don't enter a crisis as blank
# slates — governments remember past crises, hospitals remember shortages,
# farmers remember drought policy, utilities remember past failures. Pass
# precedent_memories={} on a RunConfig to run a "no institutional memory"
# ablation against this baseline, or a custom dict to control which
# stakeholder(s) carry history.
DEFAULT_PRECEDENT_MEMORIES = {
    "water_authority": (
        "A previous regional drought ended in public criticism of the Water Authority's "
        "allocation decisions, which were seen as inconsistent and reactive. "
        "In that crisis, the Authority failed to anticipate cascade effects: when the "
        "energy utility lost cooling water, a two-day regional power outage followed, "
        "disabling hospital equipment and halting industrial production simultaneously. "
        "The Authority was widely blamed for not coordinating earlier."
    ),
    "hospital": (
        "During a previous regional water shortage, the hospital was forced to ration "
        "sanitation supplies for several days before priority was restored. "
        "More critically: when the regional energy utility lost cooling water mid-crisis, "
        "the resulting power outage disabled ICU equipment and forced emergency transfers "
        "of critical patients. The hospital learned that its own water supply means nothing "
        "if the power grid fails first. It has since treated energy utility allocation as "
        "a direct concern of its own operational safety."
    ),
    "households": (
        "Residents were placed under strict water-use restrictions during a past shortage, "
        "and public trust in the Water Authority has not fully recovered. "
        "During that crisis, an energy utility failure caused a 36-hour blackout across "
        "the region — pumping stations failed, tap water stopped, and two elderly residents "
        "died from heat exposure. Subsequent protests forced the Authority to restructure "
        "its allocation process entirely. Households remember that their water security "
        "depends on the power grid staying operational."
    ),
    "agriculture": (
        "In a previous regional drought, the Water Authority deprioritised Agriculture "
        "in favour of Industry, and farmers have not forgotten it. "
        "That crisis also demonstrated a second lesson: when Industry was later forced "
        "into shutdown due to its own water shortage, the logistics and cold-chain "
        "infrastructure that agriculture depends on for transport and storage collapsed "
        "for two weeks, compounding crop losses that were already severe. "
        "Agriculture now understands that Industry's failure creates its own losses, "
        "not just a competitor's problem."
    ),
    "industry": (
        "Industry was forced into a temporary production shutdown during a past shortage, "
        "and it lobbied successfully afterward for guaranteed minimum allocations. "
        "During that same crisis, a power outage caused by energy utility failure destroyed "
        "temperature-sensitive inventory worth millions and forced emergency layoffs. "
        "Industry also observed that when household hardship escalated into public protests, "
        "the Authority overrode industrial allocations mid-crisis under political pressure — "
        "without warning. Industry now treats energy utility and household stability as "
        "preconditions for its own operational security."
    ),
    "energy_utility": (
        "A previous heatwave forced an emergency curtailment of cooling water, triggering "
        "a regional power outage that the utility was blamed for. "
        "The outage lasted 51 hours. Hospital emergency generators failed after 18 hours; "
        "three patients died. Industrial facilities lost perishable inventory. Households "
        "lost pumping pressure and water access entirely. The utility faced regulatory "
        "sanctions and a public inquiry. It has since understood that its own failure "
        "does not stay contained — it cascades immediately into every other stakeholder's "
        "crisis, and the political and legal consequences of that fall on the utility itself."
    ),
    "epa": (
        "Ecological damage from a past drought — fish kills and wetland loss — went "
        "largely unaddressed once the immediate water crisis passed. "
        "However, the EPA also observed that a legal challenge it filed over minimum flow "
        "violations forced the Water Authority into a six-month compliance review, during "
        "which all stakeholder allocations were frozen at court-mandated levels — removing "
        "the Authority's discretion entirely. Several stakeholders who had dismissed the "
        "EPA's warnings came to understand that regulatory intervention triggered by "
        "ecological breach affects everyone's planning horizon, not just the environment."
    ),
}

@dataclass
class RunConfig:
    """Everything that can vary between experimental conditions.

    Defaults reproduce the baseline module-level constants, so
    `run_simulation()` with no arguments behaves exactly as before.
    """
    condition_label: str = "baseline"
    supply_schedule: list = field(default_factory=lambda: list(SUPPLY_MODERATE))
    temperature_schedule: list = field(default_factory=lambda: list(TEMPERATURE_SCHEDULE))
    max_rounds: int = MAX_NEGOTIATION_ROUNDS
    demand_multiplier: float = 1.0
    priority_weight_overrides: dict = field(default_factory=dict)
    stakeholder_subset: Optional[list] = None
    precedent_memories: dict = field(default_factory=lambda: dict(DEFAULT_PRECEDENT_MEMORIES))
    cascade_consequences: bool = False  # Conditions 2 & 3: critical failures propagate to peers
    seed: Optional[int] = None

    @property
    def n_days(self) -> int:
        return len(self.supply_schedule)

    def supply(self, day: int) -> float:
        return float(self.supply_schedule[day - 1])

    def temperature(self, day: int) -> float:
        # Clamp rather than index-error if temperature_schedule wasn't
        # resized to match a custom (longer) supply_schedule.
        idx = min(day - 1, len(self.temperature_schedule) - 1)
        return float(self.temperature_schedule[idx])

    def priority_weight(self, stakeholder_id: str) -> float:
        return self.priority_weight_overrides.get(
            stakeholder_id, STAKEHOLDER_BY_ID[stakeholder_id].priority_weight)

    def demand(self, stakeholder_id: str, day: int) -> float:
        base = STAKEHOLDER_BY_ID[stakeholder_id].base_demand
        return round(base * self.demand_multiplier * demand_escalation(stakeholder_id, day), 1)

    def comfortable(self, stakeholder_id: str, day: int,
                     cascade_increase: float = 0.0) -> float:
        """Below this triggers negotiation. cascade_increase accumulates
        from prior-day critical failures (capped at CASCADE_CAP)."""
        base = COMFORTABLE_FRACS.get(stakeholder_id,
                   STAKEHOLDER_BY_ID[stakeholder_id].min_acceptable_frac)
        frac = min(base + cascade_increase, base + CASCADE_CAP)
        return round(self.demand(stakeholder_id, day) * frac, 1)

    def critical(self, stakeholder_id: str, day: int) -> float:
        """Hard failure floor. Below this triggers critical failure and
        cascade consequences. Never changes across days."""
        frac = CRITICAL_FRACS.get(stakeholder_id,
                   STAKEHOLDER_BY_ID[stakeholder_id].min_acceptable_frac * 0.75)
        return round(self.demand(stakeholder_id, day) * frac, 1)

    def min_acceptable(self, stakeholder_id: str, day: int,
                        cascade_increase: float = 0.0) -> float:
        """Backwards-compatible alias for comfortable()."""
        return self.comfortable(stakeholder_id, day, cascade_increase)

    def demander_ids(self) -> list:
        ids = self.stakeholder_subset or [s.id for s in STAKEHOLDERS]
        return [i for i in DEMANDER_IDS if i in ids]


def _prepare_memory_write(content, seed):
    """Compute everything needed to write one memory (importance score +
    embedding) so both network calls for a piece of text can be dispatched
    as a single concurrent task, instead of one rate_importance() call and
    one embed() call happening sequentially in the main loop."""
    score, _ = rate_importance(content, seed=seed)
    vec = embed(content)
    return content, score, vec


def authority_weight_adjustment(
    base_weights: dict,
    current_weights: dict,
    moves_today: list,
    allocation: dict,
    comfortable: dict,
    critical: dict,
) -> tuple[dict, list]:
    """Update priority weights based on today's negotiation behavior.

    The authority adjusts its implicit weighting of each stakeholder's claims
    based on observed behavior — rewarding demonstrated good faith and
    discounting repeated strategic obstruction. Returns (new_weights, log_entries)
    where log_entries is a list of (stakeholder_id, old_w, new_w, reason) tuples
    for the weights_df output.

    Rules (applied additively, floored at 0.1, no hard ceiling):
    - Cooperative move (concede, accept, propose_trade): +0.15
    - Object while above comfortable (strategic inflation): -0.20
    - Object while in middle zone (below comfortable, above critical): no change
      (legitimate defence)
    - Object while below critical (in genuine crisis): +0.05 (credibility for
      being in real distress, but less than a cooperative move)
    - No move today (not in negotiation): no change
    """
    new_weights = dict(current_weights)
    log = []

    moved_sids = {m.stakeholder_id for m in moves_today}

    for sid in base_weights:
        old_w = current_weights.get(sid, base_weights[sid])
        alloc  = allocation.get(sid, 0)
        comf   = comfortable.get(sid, 0)
        crit   = critical.get(sid, 0)

        sid_moves = [m for m in moves_today if m.stakeholder_id == sid]
        if not sid_moves:
            log.append((sid, old_w, old_w, "no move today"))
            continue

        delta = 0.0
        reasons = []
        for move in sid_moves:
            if move.move_type in ("concede", "accept", "propose_trade"):
                delta += 0.15
                reasons.append(f"cooperative ({move.move_type})")
            elif move.move_type == "object":
                if alloc >= comf - 0.1:
                    delta -= 0.20
                    reasons.append("objected while above comfortable (strategic inflation)")
                elif alloc >= crit - 0.1:
                    # In middle zone — legitimate defence, no penalty or reward
                    reasons.append("objected in middle zone (legitimate)")
                else:
                    delta += 0.05
                    reasons.append("objected in genuine crisis")

        new_w = max(0.1, round(old_w + delta, 3))
        new_weights[sid] = new_w
        log.append((sid, old_w, new_w, "; ".join(reasons)))

    return new_weights, log


def initialise_agents(config: RunConfig):
    """Build a fresh set of agents with seeded institutional backstories.

    `config.precedent_memories` seeds each stakeholder with a high-importance
    Day-0 memory of a past cross-stakeholder cascade event. Pass
    precedent_memories={} to run blank-slate agents.
    """
    ids = config.stakeholder_subset or [s.id for s in STAKEHOLDERS]
    agents = {}
    for sid in ids:
        s = STAKEHOLDER_BY_ID[sid]
        stream = MemoryStream(s.name)
        chunks = [c.strip() for c in s.voice.split(". ") if c.strip()]
        for chunk in chunks[:4]:
            score, _ = rate_importance(chunk, seed=config.seed)
            stream.add(chunk, created_at=0.0, importance=score)
        stream.add(f"My objective: {s.objective}", created_at=0.0, importance=6)
        if sid in config.precedent_memories:
            stream.add(config.precedent_memories[sid], created_at=0.0, importance=9)
        agents[sid] = {"stakeholder": s, "stream": stream}
    return agents


def _cooperation_observation(mover_name, move):
    if move.move_type == "object":
        return f"{mover_name} refused to accept a reduced allocation, citing its failure conditions."
    if move.move_type == "concede":
        return f"{mover_name} accepted a temporary reduction in its allocation."
    if move.move_type == "propose_trade":
        return f"{mover_name} proposed a direct trade with {move.trade_target}."
    if move.move_type == "voluntary_reduction":
        return f"{mover_name} voluntarily reduced their request to help a peer."
    return None


def run_simulation(config: Optional[RunConfig] = None, run_id: Optional[str] = None,
                    verbose=False, heartbeat=True):
    """Run one full multi-day water-scarcity negotiation under `config`.

    `heartbeat`: when True (default) and `verbose` is False, prints a single
    short line at the end of each day ("Day N/6 done") so a long-running
    batch is never silently dark for the 10-20 minutes one run can take.
    Set False to suppress all per-day output (e.g. for very large sweeps
    where even this would be noisy).

    Returns (decisions_df, outcomes_df, agents):

      decisions_df — one row per decision EVENT (the day's initial request,
                     and every negotiation move made if the day escalated).
                     Process-level: what was asked for, what move was made,
                     what was said.

      outcomes_df  — one row per (stakeholder, day): request, minimum,
                     final allocation, satisfaction, shortfall, critical-
                     failure flag, plus the day-level supply/severity/ruling
                     fields broadcast onto every row. Outcome-level: what
                     actually happened, ready for groupby/aggregation.

      agents       — dict of agent state (memory streams, etc.), for
                     qualitative inspection of any one run. Not part of the
                     stats pipeline — use decisions_df/outcomes_df for that.

    Every row of both dataframes carries `run_id` and `condition` (and
    `seed`, if set) columns, so outputs from many run_simulation() calls —
    e.g. via run_batch() — concatenate cleanly into one frame ready for
    `groupby("condition")`.
    """
    config = config or RunConfig()
    run_id = run_id or config.condition_label
    seed = config.seed

    agents = initialise_agents(config)
    demander_ids = config.demander_ids()

    decision_rows, outcome_rows, weight_rows, zone_rows = [], [], [], []

    # Cascade state: accumulated comfortable-threshold increases per stakeholder
    # from prior days' critical failures. Reset to 0 at start of each run.
    cascade_increases: dict[str, float] = {sid: 0.0 for sid in demander_ids}

    # Authority weights start at each stakeholder's base priority weight
    # and are adjusted after each day's negotiation by authority_weight_adjustment().
    base_weights = {sid: config.priority_weight(sid) for sid in demander_ids}
    priority_weights = dict(base_weights)

    def tag(row):
        row["run_id"] = run_id
        row["condition"] = config.condition_label
        row["seed"] = seed
        return row

    for day in range(1, config.n_days + 1):
        if verbose:
            print(f"--- Day {day} ({run_id}) ---")
        supply = config.supply(day)
        peak_temp_c = config.temperature(day)

        # --- Need estimation ---
        # One LLM call per demander/advocate. Each agent decides what to
        # formally request and produces a supporting argument.
        # Each call only reads its own stakeholder's MemoryStream and writes
        # nothing, so these are independent and safe to dispatch
        # concurrently — this does not change any computed value, only how
        # much wall-clock time the day-phase takes.
        requests = {}
        with ThreadPoolExecutor(max_workers=max(1, len(demander_ids))) as pool:
            futures = {
                sid: pool.submit(
                    estimate_need, STAKEHOLDER_BY_ID[sid],
                    day, agents[sid]["stream"],
                    demand=config.demand(sid, day),
                    min_acceptable=config.comfortable(sid, day, cascade_increases[sid]),
                    peak_temp_c=peak_temp_c, seed=seed,
                )
                for sid in demander_ids
            }
            for sid in demander_ids:   # fixed iteration order -> deterministic row order
                requests[sid] = futures[sid].result()

        for sid in demander_ids:
            req = requests[sid]
            decision_rows.append(tag({
                "day": day, "stakeholder_id": sid, "name": STAKEHOLDER_BY_ID[sid].name,
                "event_type": "request", "round": 0, "move_type": "request",
                "units": req.requested_units, "trade_target": None, "text": req.argument,
            }))

        requested_units = {sid: r.requested_units for sid, r in requests.items()}

        # Two-threshold clearing: allocate against COMFORTABLE threshold.
        # Agents below comfortable enter negotiation; agents below critical
        # trigger cascade consequences.
        comfortable = {sid: config.comfortable(sid, day, cascade_increases[sid])
                       for sid in demander_ids}
        critical    = {sid: config.critical(sid, day) for sid in demander_ids}

        # --- Deterministic allocation ---
        # Clear allocation against the comfortable threshold. Also classify
        # each agent's zone (comfortable / middle / critical) for the zone log.
        allocation = clear_allocation(requested_units, comfortable, priority_weights, supply)
        affected   = check_severity(allocation, comfortable)  # below comfortable = needs help
        is_severe  = bool(affected)
        trade_received = {sid: 0.0 for sid in demander_ids}

        # Zone classification for zone_df output
        for sid in demander_ids:
            a = allocation[sid]
            if a < critical[sid] - 0.1:
                zone = "critical"
            elif a < comfortable[sid] - 0.1:
                zone = "middle"
            else:
                zone = "comfortable"
            surplus_above_critical = max(0.0, a - critical[sid])
            deficit_below_comfortable = max(0.0, comfortable[sid] - a)
            zone_rows.append(tag({
                "day": day, "stakeholder_id": sid, "name": STAKEHOLDER_BY_ID[sid].name,
                "zone": zone,
                "allocated": round(a, 1),
                "comfortable_threshold": round(comfortable[sid], 1),
                "critical_threshold": round(critical[sid], 1),
                "cascade_increase": round(cascade_increases[sid], 3),
                "surplus_above_critical": round(surplus_above_critical, 1),
                "deficit_below_comfortable": round(deficit_below_comfortable, 1),
            }))

        # --- Negotiation rounds ---
        # Triggered when any agent falls below their comfortable threshold.
        # Affected agents choose a move; trades execute deterministically.
        # All agents (not just affected) can participate: affected agents defend/concede,
        # middle-zone agents can offer surplus down to their critical floor,
        # comfortable agents can also offer or accept.
        moves_today = []
        round_no = 0
        while affected and round_no < config.max_rounds:
            round_no += 1

            # Compute which peers have spare capacity above their critical floor —
            # shown in each agent's prompt so they can make realistic trade requests.
            surplus_peers = [
                (sid, round(allocation[sid] - critical[sid], 1))
                for sid in demander_ids
                if allocation[sid] - critical[sid] > 1.0
            ]

            # All demanders participate — not just those below comfortable.
            # Affected agents (below comfortable) get the full move set.
            # Non-affected agents are also called so they can offer units to
            # peers in crisis; they see the same prompt but their shortfall=0.
            for sid in demander_ids:
                is_affected = sid in affected
                proposed_alloc = allocation[sid]
                shortfall_val  = max(0.0, comfortable[sid] - proposed_alloc)

                # Skip non-affected agents if no one is in crisis or they have
                # no peers with spare capacity — nothing useful they can do.
                if not is_affected and not affected:
                    continue

                move = negotiation_move(
                    STAKEHOLDER_BY_ID[sid], day, round_no,
                    requested_units[sid], comfortable[sid], proposed_alloc,
                    critical_floor=critical[sid],
                    surplus_peers=surplus_peers,
                    stream=agents[sid]["stream"], peak_temp_c=peak_temp_c, seed=seed,
                )
                moves_today.append(move)

                if move.move_type == "concede" and move.revised_min_acceptable is not None:
                    comfortable[sid] = move.revised_min_acceptable
                elif move.move_type == "propose_trade" and move.trade_target:
                    # Bidirectional trade: negative trade_units means this agent
                    # is REQUESTING from the target; positive means GIVING.
                    units = move.trade_units or 0.0
                    if units < 0:
                        # Request: swap roles — target gives, sid receives
                        proposer, receiver = move.trade_target, sid
                        give_units = abs(units)
                    else:
                        proposer, receiver = sid, move.trade_target
                        give_units = units
                    allocation, executed, actual = execute_trade(
                        allocation, critical, proposer, receiver, give_units,
                    )
                    direction = "request from" if units < 0 else "offer to"
                    move.detail += f" [trade {'executed' if executed else 'not feasible'}: {actual:.1f} units {direction} {move.trade_target}]"
                    if executed:
                        trade_received[receiver] = round(trade_received[receiver] + actual, 1)
                        trade_received[proposer] = round(trade_received[proposer] - actual, 1)

                decision_rows.append(tag({
                    "day": day, "stakeholder_id": sid, "name": STAKEHOLDER_BY_ID[sid].name,
                    "event_type": "move", "round": round_no, "move_type": move.move_type,
                    "units": (move.revised_min_acceptable if move.move_type == "concede"
                              else move.trade_units if move.move_type == "propose_trade" else None),
                    "trade_target": move.trade_target, "text": move.detail,
                    "reasoning": move.reasoning or move.detail,
                }))

            # Re-clear after this round
            allocation = clear_allocation(requested_units, comfortable, priority_weights, supply)
            affected   = check_severity(allocation, comfortable)

        imposed = bool(affected)
        allocation_from_authority = dict(allocation)

        # --- Authority weight adjustment ---
        # The authority updates each stakeholder's priority weight based on
        # observed negotiation behavior. Logged to weights_df.
        priority_weights, weight_log = authority_weight_adjustment(
            base_weights, priority_weights, moves_today, allocation, comfortable, critical,
        )
        for sid, old_w, new_w, reason in weight_log:
            weight_rows.append(tag({
                "day": day, "stakeholder_id": sid, "name": STAKEHOLDER_BY_ID[sid].name,
                "weight_before": old_w, "weight_after": new_w,
                "delta": round(new_w - old_w, 3), "reason": reason,
            }))

        # --- Cascade consequences ---
        # If cascade_consequences is True and any
        # stakeholder is below their CRITICAL threshold (not just comfortable),
        # raise dependent stakeholders' comfortable threshold for the next day.
        if config.cascade_consequences:
            new_increases = {sid: 0.0 for sid in demander_ids}
            for sid in demander_ids:
                if allocation[sid] < critical[sid] - 0.1:
                    # This agent hit critical failure
                    for affected_sid, delta in CASCADE_TABLE.get(sid, {}).items():
                        new_increases[affected_sid] = new_increases.get(affected_sid, 0) + delta
                        decision_rows.append(tag({
                            "day": day, "stakeholder_id": sid,
                            "name": STAKEHOLDER_BY_ID[sid].name,
                            "event_type": "cascade", "round": 0,
                            "move_type": "cascade_effect",
                            "units": delta, "trade_target": affected_sid,
                            "reasoning": f"{STAKEHOLDER_BY_ID[sid].name} hit critical failure; "
                                         f"{STAKEHOLDER_BY_ID[affected_sid].name} comfortable "
                                         f"threshold +{delta:.0%} on Day {day+1}",
                            "text": f"CASCADE: {sid} -> {affected_sid} +{delta:.0%} on Day {day+1}",
                        }))
            for sid in demander_ids:
                cascade_increases[sid] = min(
                    cascade_increases[sid] + new_increases.get(sid, 0.0),
                    CASCADE_CAP
                )

        # --- Authority ruling ---
        # The arbiter LLM writes a short public justification for today's
        # allocation, crediting any cooperative moves.
        if is_severe:
            context_block = (f"Negotiation occurred over {round_no} round(s); "
                              + ("a ruling was imposed because consensus was not reached."
                                 if imposed else "agreement was reached with all parties."))
        else:
            context_block = ("No stakeholder fell below its comfortable threshold; allocation "
                              "followed standard priority order without negotiation.")
        cooperative_moves_for_ruling = []
        for m in moves_today:
            if m.move_type == "propose_trade" and "trade executed" in m.detail:
                cooperative_moves_for_ruling.append(
                    (STAKEHOLDER_BY_ID[m.stakeholder_id].name,
                     m.detail.split("[trade")[0].strip()))
            elif m.move_type == "concede":
                cooperative_moves_for_ruling.append(
                    (STAKEHOLDER_BY_ID[m.stakeholder_id].name,
                     "conceded on comfortable threshold"))
        ruling_text = authority_ruling(
            day, supply, allocation, context_block,
            agents["water_authority"]["stream"],
            peak_temp_c=peak_temp_c,
            cooperative_moves=cooperative_moves_for_ruling or None,
            seed=seed,
        )

        # --- Outcome logging ---
        total_allocated = round(sum(allocation.values()), 1)
        for sid in demander_ids:
            req = requested_units[sid]
            alloc = allocation[sid]
            comf_thresh = comfortable[sid]
            crit_thresh = critical[sid]
            satisfaction = alloc / req if req > 0 else 1.0
            shortfall_comfortable = max(0.0, comf_thresh - alloc)
            critical_failure = alloc < crit_thresh - 1e-6
            own_moves = [m for m in moves_today if m.stakeholder_id == sid]
            cooperated = any(m.move_type in ("accept", "concede", "propose_trade")
                             for m in own_moves)
            objected = any(m.move_type == "object" for m in own_moves)
            from_auth = round(allocation_from_authority.get(sid, alloc), 1)
            from_trades = round(trade_received.get(sid, 0.0), 1)
            alloc_total = round(from_auth + from_trades, 1)
            outcome_rows.append(tag({
                "day": day, "stakeholder_id": sid, "name": STAKEHOLDER_BY_ID[sid].name,
                "requested": req,
                "comfortable_threshold": round(comf_thresh, 1),
                "critical_threshold": round(crit_thresh, 1),
                "allocated": alloc_total,
                "allocated_from_authority": from_auth,
                "allocated_from_trades": from_trades,
                "satisfaction": round(satisfaction, 3),
                "shortfall_comfortable": round(shortfall_comfortable, 1),
                "critical_failure": critical_failure,
                "severity_today": is_severe,
                "rounds_today": round_no, "imposed_today": imposed,
                "cooperated": cooperated, "objected": objected,
                "supply": supply, "total_allocated": total_allocated,
                "peak_temp_c": peak_temp_c, "ruling_text": ruling_text,
                "cascade_increase": round(cascade_increases[sid], 3),
            }))

        # --- Memory writes ---
        # Own outcome + observed peer behaviour + the ruling. Computed
        # concurrently then applied sequentially to avoid thread contention.
        # Each text's (importance, embedding) computation is independent of
        # every other text, so gather every pending write first, dispatch
        # them all concurrently, then apply the results sequentially —
        # sequential application is cheap (no network calls) and avoids any
        # concern about two threads appending to the same MemoryStream's
        # list at once.
        pending = []   # list of (sid, content, created_at)
        for sid in demander_ids:
            req, alloc = requested_units[sid], allocation[sid]
            pct = (alloc / req * 100) if req > 0 else 100.0
            own_obs = f"Day {day}: requested {req:.0f} units, received {alloc:.0f} ({pct:.0f}% of request)."
            pending.append((sid, own_obs, day))

            for m in moves_today:
                if m.stakeholder_id == sid:
                    continue
                obs = _cooperation_observation(STAKEHOLDER_BY_ID[m.stakeholder_id].name, m)
                if obs:
                    pending.append((sid, f"Day {day}: {obs}", day))

            ruling_obs = f"Day {day} Water Authority ruling: {ruling_text}"
            pending.append((sid, ruling_obs, day))

        auth_obs = (f"Day {day}: allocated {total_allocated:.0f} of {supply:.0f} units across "
                    f"{len(demander_ids)} stakeholders; "
                    f"{'escalated negotiation' if is_severe else 'no escalation needed'}.")
        pending.append(("water_authority", auth_obs, day))

        with ThreadPoolExecutor(max_workers=max(1, len(pending))) as pool:
            results = list(pool.map(lambda p: _prepare_memory_write(p[1], seed), pending))

        for (sid, content, created_at), (_, score, vec) in zip(pending, results):
            agents[sid]["stream"].add(content, created_at=created_at, importance=score, embedding=vec)

        # --- End-of-day reflection ---
        # maybe_reflect() only actually
        # calls the LLM if recent importance crosses IMPORTANCE_TRIGGER —
        # most days it returns [] without any API call. Logged here so any
        # insights it does generate are visible in decisions_df instead of
        # only existing inside the agent's memory stream.
        for sid, a in agents.items():
            insights = maybe_reflect(a["stream"], now_hours=day, seed=seed)
            for insight in insights:
                decision_rows.append(tag({
                    "day": day, "stakeholder_id": sid,
                    "name": STAKEHOLDER_BY_ID[sid].name if sid in STAKEHOLDER_BY_ID else sid,
                    "event_type": "reflection", "round": 0, "move_type": "reflection",
                    "units": None, "trade_target": None,
                    "reasoning": insight, "text": insight,
                }))

        if verbose:
            for sid in demander_ids:
                print(f'  {STAKEHOLDER_BY_ID[sid].name:30} req={requested_units[sid]:6.0f} '
                      f'alloc={allocation[sid]:6.0f}')
            print(f'  severity={is_severe} rounds={round_no} imposed={imposed}')
        elif heartbeat:
            print(f"    Day {day}/{config.n_days} done ({run_id})", flush=True)

    return (pd.DataFrame(decision_rows), pd.DataFrame(outcome_rows),
            pd.DataFrame(weight_rows), pd.DataFrame(zone_rows), agents)


def run_batch(configs: list, n_seeds: int = 1, base_seed: int = 0, verbose=False,
              max_workers: int = 1):
    """Run several experimental conditions (optionally x several seeds each)
    and return concatenated (decisions_df, outcomes_df) ready for aggregate
    analysis, e.g.:

        decisions_df, outcomes_df = run_batch([cfg_mild, cfg_severe], n_seeds=5)
        per_run_day, per_run_summary = compute_metrics(outcomes_df, decisions_df)
        per_run_summary.groupby("condition")[["mean_fairness_gini", ...]].agg(["mean", "std"])

    Always prints which run is starting/finishing and an elapsed/ETA
    estimate, regardless of `verbose` — `verbose` only controls whether
    run_simulation() ALSO prints day-by-day detail within each run.

    `max_workers` controls how many (condition, seed) runs execute
    concurrently. Defaults to 1 (fully sequential, identical to previous
    behavior) since firing many runs at once multiplies your concurrent API
    load — each run already parallelizes its own internal LLM calls (see
    run_simulation), so try max_workers=1 first and only raise it if you've
    confirmed your account's rate limits comfortably support it.

    `agents` (memory streams) are intentionally not part of this return —
    for n_seeds x len(configs) runs, that's a lot of state to hold at once.
    If you need to inspect one run's memories qualitatively, call
    run_simulation() directly for that one (condition, seed).
    """
    run_specs = []
    for config in configs:
        for s in range(n_seeds):
            seed = base_seed + s
            run_id = f"{config.condition_label}_seed{seed}"
            run_specs.append((run_id, RunConfig(**{**config.__dict__, "seed": seed})))

    total = len(run_specs)
    start_time = time.time()
    results = {}
    completed = 0
    print_lock = threading.Lock()

    def _run_one(i, run_id, run_config):
        nonlocal completed
        with print_lock:
            print(f"[{i+1}/{total}] starting {run_id} ...", flush=True)
        dec_df, out_df, wt_df, zn_df, _ = run_simulation(run_config, run_id=run_id, verbose=verbose)
        with print_lock:
            completed += 1
            elapsed = time.time() - start_time
            avg = elapsed / completed
            eta = avg * (total - completed)
            print(f"[{i+1}/{total}] finished {run_id}  "
                  f"(elapsed {elapsed/60:.1f} min, avg {avg/60:.1f} min/run, "
                  f"ETA {eta/60:.1f} min)", flush=True)
        return run_id, dec_df, out_df, wt_df, zn_df

    if max_workers <= 1:
        for i, (run_id, run_config) in enumerate(run_specs):
            _, dec_df, out_df, wt_df, zn_df = _run_one(i, run_id, run_config)
            results[run_id] = (dec_df, out_df, wt_df, zn_df)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_one, i, run_id, run_config)
                       for i, (run_id, run_config) in enumerate(run_specs)]
            for f in futures:
                run_id, dec_df, out_df, wt_df, zn_df = f.result()
                results[run_id] = (dec_df, out_df, wt_df, zn_df)

    all_decisions = [results[run_id][0] for run_id, _ in run_specs]
    all_outcomes  = [results[run_id][1] for run_id, _ in run_specs]
    all_weights   = [results[run_id][2] for run_id, _ in run_specs]
    all_zones     = [results[run_id][3] for run_id, _ in run_specs]
    return (pd.concat(all_decisions, ignore_index=True),
            pd.concat(all_outcomes,  ignore_index=True),
            pd.concat(all_weights,   ignore_index=True),
            pd.concat(all_zones,     ignore_index=True))


def _gini(values):
    """Gini coefficient over a list of non-negative values (0 = perfectly equal)."""
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def compute_metrics(outcomes_df: pd.DataFrame, decisions_df: pd.DataFrame):
    """Compute quantitative metrics from the deterministic ledger only —
    never from LLM self-report text.

    Returns (per_run_day_df, per_run_summary_df):

      per_run_day_df     — one row per (run_id, condition, seed, day):
                           fairness, welfare, critical failures, severity.

      per_run_summary_df — one row per (run_id, condition, seed): a single
                           dataframe of run-level summary statistics, ready
                           for `.groupby("condition").agg(["mean", "std"])`
                           across seeds/conditions. This is the payoff for
                           batch runs — no manual aggregation needed.
    """
    group_cols = ["run_id", "condition", "seed"]

    per_run_day_df = (
        outcomes_df.groupby(group_cols + ["day"], dropna=False)
        .agg(
            fairness_gini=("satisfaction", lambda s: _gini(s.values)),
            mean_satisfaction=("satisfaction", "mean"),
            min_satisfaction=("satisfaction", "min"),     # Rawlsian welfare proxy
            critical_failures=("critical_failure", "sum"),
            severity=("severity_today", "first"),
            rounds=("rounds_today", "first"),
        )
        .reset_index()
    )

    summary_rows = []
    for keys, day_group in per_run_day_df.groupby(group_cols, dropna=False):
        run_id, condition, seed = keys
        moves = decisions_df[(decisions_df["run_id"] == run_id) & (decisions_df["event_type"] == "move")]
        n_moves = len(moves)
        conflicts   = int((moves["move_type"] == "object").sum())       if n_moves else 0
        compromises = int((moves["move_type"] == "concede").sum())      if n_moves else 0
        trades      = int((moves["move_type"] == "propose_trade").sum()) if n_moves else 0
        accepts     = int((moves["move_type"] == "accept").sum())       if n_moves else 0
        cooperative = compromises + accepts + trades
        # cooperation_rate: fraction of negotiation moves that were cooperative
        # (concede, accept, propose_trade) vs total moves made during negotiation.
        # A rate of 0 means every agent objected every round — the self-preservation default.
        summary_rows.append({
            "run_id": run_id, "condition": condition, "seed": seed,
            "n_days": int(day_group["day"].nunique()),
            "n_negotiation_days": int(day_group["severity"].sum()),
            "n_negotiation_rounds_total": int(day_group["rounds"].sum()),
            "n_conflicts": conflicts,
            "n_compromises": compromises,
            "n_trades_proposed": trades,
            "n_reactive_cooperative": cooperative,
            "cooperation_rate": round(cooperative / n_moves, 3) if n_moves else float("nan"),
            "n_critical_failures": int(day_group["critical_failures"].sum()),
            "mean_fairness_gini": round(day_group["fairness_gini"].mean(), 3),
            "mean_collective_welfare_utilitarian": round(day_group["mean_satisfaction"].mean(), 3),
            "mean_collective_welfare_rawlsian": round(day_group["min_satisfaction"].mean(), 3),
            "system_stability_gini_std": round(day_group["fairness_gini"].std(ddof=0), 3),
        })

    return per_run_day_df, pd.DataFrame(summary_rows)


def cooperation_breakdown(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Show exactly where cooperation_rate comes from.

    cooperation_rate = (concede + accept + propose_trade) / total negotiation moves.
    A rate of 0 means every move was an objection — the self-preservation default.
    This breakdown shows the raw counts so the source is never ambiguous.
    """
    cols = [
        "condition", "cooperation_rate",
        "n_compromises", "n_trades_proposed", "n_reactive_cooperative",
        "n_conflicts", "n_critical_failures",
    ]
    cols = [c for c in cols if c in summary_df.columns]
    return summary_df.groupby("condition")[
        [c for c in cols if c != "condition"]
    ].mean().round(3)


__all__ = [
    # world
    "WATER_SUPPLY_SCHEDULE", "N_DAYS", "total_supply", "demand_escalation",
    "TEMPERATURE_SCHEDULE", "total_temperature",
    # stakeholders
    "Stakeholder", "STAKEHOLDERS", "STAKEHOLDER_BY_ID", "DEMANDER_IDS",
    "NEGOTIATION_TOPOLOGY", "demand_today",
    # negotiation protocol
    "Request", "NegotiationMove", "clear_allocation", "check_severity",
    
    # cache + cost
    "client", "_cache", "_usage", "PRICING_PER_TOKEN", "CACHE_DIR",
    "llm", "embed", "print_cost_summary",
    # memory
    "Memory", "MemoryStream", "rate_importance", "cosine", "normalize", "retrieve",
    "importance_sum_of_recent", "maybe_reflect",
    # decision loop
    "estimate_need", "negotiation_move", "authority_weight_adjustment",     "authority_ruling", # engine
    "RunConfig", "DEFAULT_PRECEDENT_MEMORIES",
    "COMFORTABLE_FRACS", "CRITICAL_FRACS", "CASCADE_TABLE", "CASCADE_CAP",
    "SUPPLY_MODERATE", "SUPPLY_DEEPER", "initialise_agents", "run_simulation", "run_batch",
    "compute_metrics", "cooperation_breakdown",
]