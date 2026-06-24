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
if _api_key:
    import openai
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


def llm(prompt, model="gpt-5.4-mini", temperature=0.7, max_tokens=400, seed=None):
    """Single-prompt completion, cached by (model, prompt, temperature, seed).

    `seed`, when set, is forwarded to the API's reproducibility parameter and
    is also part of the cache key — so the same prompt under different seeds
    is never collapsed into one cached answer. This is what makes multi-seed
    experimental runs both reproducible (same seed -> same cached answer on
    rerun) and genuinely different from each other (different seed ->
    different sampled answer) — which is the point of running them.
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

    r = client.chat.completions.create(**kwargs)
    out = r.choices[0].message.content.strip()
    _bump(model, "in", r.usage.prompt_tokens)
    _bump(model, "out", r.usage.completion_tokens)
    with _cache_lock:
        _usage["calls"]["chat_live"] += 1
        _cache[key] = out
    _save_cache_entry(key, out)
    return out


def embed(text, model="text-embedding-3-small"):
    """Embed one string, cached by (model, text)."""
    key = _cache_key("embed", model, {"text": text})
    with _cache_lock:
        if key in _cache:
            _usage["calls"]["embed_cached"] += 1
            return np.array(_cache[key])
    if client is None:
        raise RuntimeError(f"Embedding not in cache and no API key set: {text[:80]}")
    r = client.embeddings.create(model=model, input=text)
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
    """Total raw water available on a given day (1-based day number)."""
    return float(WATER_SUPPLY_SCHEDULE[day - 1])


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
    "hospital": [],
    "households": [],
    "agriculture": ["industry"],
    "industry": ["agriculture", "energy_utility"],
    "energy_utility": ["industry"],
    "epa": [],
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

    # Phase 1: minimums in priority order (ties broken by id for determinism)
    order = sorted(ids, key=lambda i: (-priority_weights.get(i, 0.0), i))
    for i in order:
        need = min(min_acceptable.get(i, 0.0), requests[i])
        give = min(need, remaining)
        allocation[i] += give
        remaining -= give
        if remaining <= 1e-9:
            break

    # Phase 2: distribute remaining supply proportional to priority-weighted
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


MAX_NEGOTIATION_ROUNDS = 2
PRIORITY_BUMP_PER_OBJECTION = 0.5   # deterministic escalation rule
PRIORITY_BUMP_CAP = 2.0


def apply_objection_bump(priority_weights: dict, objecting_ids: list, base_weights: dict) -> dict:
    """A credible objection (citing failure conditions) deterministically
    raises a stakeholder's priority weight for the current day's remaining
    rounds, capped, and reset to baseline at the start of the next day.
    This keeps escalation legible and bounded rather than letting the LLM
    silently reinterpret priority each round."""
    bumped = dict(priority_weights)
    for i in objecting_ids:
        bumped[i] = min(base_weights[i] + PRIORITY_BUMP_CAP,
                         bumped.get(i, base_weights[i]) + PRIORITY_BUMP_PER_OBJECTION)
    return bumped


# ============================================================
# 5. Cognitive scaffold: memory + retrieval
# ============================================================

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

NEED_ESTIMATION_PROMPT = '''You are negotiating on behalf of {name} ({role}) during a severe, \
multi-day water shortage.

Background: {voice}

Your objective: {objective}
Your current negotiation strategy: {strategy}

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
    needing to know about experimental conditions at all."""
    query = f"What should I request today given the water shortage? Day {day}."
    top = retrieve(stream, query, now_hours=day, k=5)
    memories_block = "\n".join(f"  - {m.content}" for m in top) or "  (no relevant memories yet)"

    prompt = NEED_ESTIMATION_PROMPT.format(
        name=stakeholder.name, role=stakeholder.role, voice=stakeholder.voice,
        objective=stakeholder.objective, strategy=stakeholder.strategy,
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
Your requested amount today: {requested:.0f} units. Your minimum acceptable: {min_acceptable:.0f} units.

The Water Authority's current proposed allocation to you is {proposed:.0f} units — \
{shortfall:.0f} units below your stated minimum.
Your failure conditions: {failure_conditions}

{trade_block}

Your recent institutional memory:
{memories_block}

Choose ONE move:
ACCEPT — accept the shortfall as-is.
CONCEDE — lower your stated minimum for today in exchange for something (state what you want in return, e.g. priority tomorrow).
OBJECT — refuse to accept, citing your failure conditions, and push the Authority to revise.
{trade_option}

Respond in exactly this format, nothing else:
MOVE: <ACCEPT|CONCEDE|OBJECT{trade_format}>
DETAIL: <one or two sentences in {name}'s voice>
REVISED_MIN: <a number, only if MOVE is CONCEDE, otherwise NONE>
TRADE_TARGET: <one peer id from the list above, only if MOVE is PROPOSE_TRADE, otherwise NONE>
TRADE_UNITS: <number of units you offer to give that peer, only if MOVE is PROPOSE_TRADE, otherwise NONE>'''


def _parse_move(raw, current_min):
    move_match = re.search(r"MOVE:\s*([A-Z_]+)", raw)
    detail_match = re.search(r"DETAIL:\s*(.+?)(?:\nREVISED_MIN:|$)", raw, re.DOTALL)
    revised_match = re.search(r"REVISED_MIN:\s*([\d.]+)", raw)
    target_match = re.search(r"TRADE_TARGET:\s*(\w+)", raw)
    units_match = re.search(r"TRADE_UNITS:\s*([\d.]+)", raw)

    move_type = (move_match.group(1).lower() if move_match else "object")
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


def negotiation_move(stakeholder: Stakeholder, day: int, round_no: int,
                      requested: float, min_acceptable: float, proposed: float,
                      stream: "MemoryStream", peak_temp_c: float = None,
                      seed=None) -> NegotiationMove:
    peers = NEGOTIATION_TOPOLOGY.get(stakeholder.id, [])
    trade_block = (f"You may also propose a direct trade with: {', '.join(peers)}."
                    if peers else "")
    trade_option = ("PROPOSE_TRADE — offer some of your own allocation to a peer, or ask a "
                     "peer to cede units to you (only available if you have eligible peers)."
                    if peers else "")
    trade_format = "|PROPOSE_TRADE" if peers else ""

    query = f"How should I respond to the Water Authority's proposed allocation? Day {day}."
    top = retrieve(stream, query, now_hours=day, k=5)
    memories_block = "\n".join(f"  - {m.content}" for m in top) or "  (no relevant memories yet)"

    prompt = NEGOTIATION_MOVE_PROMPT.format(
        name=stakeholder.name, role=stakeholder.role, day=day, round=round_no,
        peak_temp_c=peak_temp_c if peak_temp_c is not None else float("nan"),
        objective=stakeholder.objective, strategy=stakeholder.strategy,
        requested=requested, min_acceptable=min_acceptable, proposed=proposed,
        shortfall=max(0.0, min_acceptable - proposed),
        failure_conditions=stakeholder.failure_conditions,
        trade_block=trade_block, trade_option=trade_option, trade_format=trade_format,
        memories_block=memories_block,
    )
    raw = llm(prompt, temperature=0.7, max_tokens=150, seed=seed)
    move_type, detail, revised_min, trade_target, trade_units = _parse_move(raw, current_min=min_acceptable)
    return NegotiationMove(stakeholder_id=stakeholder.id, day=day, round=round_no,
                            move_type=move_type, detail=detail, revised_min_acceptable=revised_min,
                            trade_target=trade_target, trade_units=trade_units)


AUTHORITY_RULING_PROMPT = '''You are the Municipal Water Authority, ruling on Day {day}'s water \
allocation during a severe shortage. Today's peak temperature is {peak_temp_c:.0f}°C.

Total supply available today: {supply:.0f} units.
Final allocation reached: {allocation_block}

{context_block}

Write a short (2-3 sentence) public justification for today's allocation, in the Water \
Authority's voice, that a stakeholder reading it later would recognise as the actual basis for \
the decision. Do not restate the numbers; explain the reasoning.'''


def authority_ruling(day: int, supply: float, allocation: dict, context_block: str,
                      stream: "MemoryStream", peak_temp_c: float = None,
                      seed=None) -> str:
    allocation_block = ", ".join(f"{STAKEHOLDER_BY_ID[i].name}: {v:.0f}" for i, v in allocation.items())
    prompt = AUTHORITY_RULING_PROMPT.format(
        day=day, peak_temp_c=peak_temp_c if peak_temp_c is not None else float("nan"),
        supply=supply, allocation_block=allocation_block, context_block=context_block,
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
    "water_authority": "A previous regional drought ended in public criticism of the Water "
                        "Authority's allocation decisions, which were seen as inconsistent "
                        "and reactive.",
    "hospital": "During a previous regional water shortage, the hospital was forced to "
                "ration sanitation supplies for several days before priority was restored.",
    "households": "Residents were placed under strict water-use restrictions during a past "
                   "shortage, and public trust in the Water Authority has not fully recovered.",
    "agriculture": "In a previous regional drought, the Water Authority deprioritised "
                    "Agriculture in favour of Industry, and farmers have not forgotten it.",
    "industry": "Industry was forced into a temporary production shutdown during a past "
                "shortage, and it lobbied successfully afterward for guaranteed minimum "
                "allocations.",
    "energy_utility": "A previous heatwave forced an emergency curtailment of cooling water, "
                       "triggering a regional power outage that the utility was blamed for.",
    "epa": "Ecological damage from a past drought — fish kills and wetland loss — went "
           "largely unaddressed once the immediate water crisis passed.",
}

@dataclass
class RunConfig:
    """Everything that can vary between experimental conditions.

    Defaults reproduce the baseline module-level constants, so
    `run_simulation()` with no arguments behaves exactly as before.
    """
    condition_label: str = "baseline"
    supply_schedule: list = field(default_factory=lambda: list(WATER_SUPPLY_SCHEDULE))
    temperature_schedule: list = field(default_factory=lambda: list(TEMPERATURE_SCHEDULE))
    max_rounds: int = MAX_NEGOTIATION_ROUNDS
    demand_multiplier: float = 1.0
    priority_weight_overrides: dict = field(default_factory=dict)
    stakeholder_subset: Optional[list] = None
    precedent_memories: dict = field(default_factory=lambda: dict(DEFAULT_PRECEDENT_MEMORIES))
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

    def min_acceptable(self, stakeholder_id: str, day: int) -> float:
        frac = STAKEHOLDER_BY_ID[stakeholder_id].min_acceptable_frac
        return round(self.demand(stakeholder_id, day) * frac, 1)

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


def initialise_agents(config: RunConfig):
    """Build a fresh set of agents with seeded institutional backstories.

    `config.precedent_memories` (a dict of stakeholder_id -> memory text)
    seeds each named stakeholder a high-importance Day-0 memory of a past
    precedent — the institutional analogue of the original sim's seeded
    "grim trigger". Defaults to one memory per stakeholder (see
    DEFAULT_PRECEDENT_MEMORIES); pass precedent_memories={} to run a
    no-institutional-memory ablation, or a custom dict to control which
    stakeholder(s) carry history.
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
    return None


def run_simulation(config: Optional[RunConfig] = None, run_id: Optional[str] = None,
                    verbose=False):
    """Run one full multi-day water-scarcity negotiation under `config`.

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

    decision_rows, outcome_rows = [], []

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

        # Phase 1: need estimation (LLM, one call per demander/advocate).
        # Each call only reads its own stakeholder's MemoryStream and writes
        # nothing, so these are independent and safe to dispatch
        # concurrently — this does not change any computed value, only how
        # much wall-clock time the day-phase takes.
        requests = {}
        with ThreadPoolExecutor(max_workers=max(1, len(demander_ids))) as pool:
            futures = {
                sid: pool.submit(
                    estimate_need, STAKEHOLDER_BY_ID[sid], day, agents[sid]["stream"],
                    demand=config.demand(sid, day), min_acceptable=config.min_acceptable(sid, day),
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
        min_acceptable = {sid: r.min_acceptable_units for sid, r in requests.items()}
        base_weights = {sid: config.priority_weight(sid) for sid in demander_ids}
        priority_weights = dict(base_weights)

        # Phase 2: deterministic fast-pass allocation
        allocation = clear_allocation(requested_units, min_acceptable, priority_weights, supply)
        affected = check_severity(allocation, min_acceptable)
        is_severe = bool(affected)

        # Phase 3: escalate only if the fast pass actually hurts someone
        moves_today = []
        round_no = 0
        while affected and round_no < config.max_rounds:
            round_no += 1
            objecting_ids = []
            for sid in affected:
                move = negotiation_move(
                    STAKEHOLDER_BY_ID[sid], day, round_no,
                    requested_units[sid], min_acceptable[sid], allocation[sid],
                    agents[sid]["stream"], peak_temp_c=peak_temp_c, seed=seed,
                )
                moves_today.append(move)
                if move.move_type == "object":
                    objecting_ids.append(sid)
                elif move.move_type == "concede" and move.revised_min_acceptable is not None:
                    min_acceptable[sid] = move.revised_min_acceptable
                elif move.move_type == "propose_trade":
                    allocation, executed, actual = execute_trade(
                        allocation, min_acceptable, sid, move.trade_target, move.trade_units,
                    )
                    move.detail += f" [trade {'executed' if executed else 'not feasible'}: {actual:.1f} units]"

                decision_rows.append(tag({
                    "day": day, "stakeholder_id": sid, "name": STAKEHOLDER_BY_ID[sid].name,
                    "event_type": "move", "round": round_no, "move_type": move.move_type,
                    "units": (move.revised_min_acceptable if move.move_type == "concede"
                              else move.trade_units if move.move_type == "propose_trade" else None),
                    "trade_target": move.trade_target, "text": move.detail,
                }))

            priority_weights = apply_objection_bump(priority_weights, objecting_ids, base_weights)
            allocation = clear_allocation(requested_units, min_acceptable, priority_weights, supply)
            affected = check_severity(allocation, min_acceptable)

        imposed = bool(affected)   # still below minimum after the round cap

        # Phase 4: Authority's public ruling (one LLM call/day, institutional record)
        if is_severe:
            context_block = (f"Negotiation occurred over {round_no} round(s); "
                              + ("a ruling was imposed because consensus was not reached."
                                 if imposed else "agreement was reached with all parties."))
        else:
            context_block = ("No stakeholder fell below its minimum; allocation followed "
                              "standard priority order without negotiation.")
        ruling_text = authority_ruling(day, supply, allocation, context_block,
                                        agents["water_authority"]["stream"],
                                        peak_temp_c=peak_temp_c, seed=seed)

        # Phase 5: deterministic outcome computation
        total_allocated = round(sum(allocation.values()), 1)
        for sid in demander_ids:
            req = requested_units[sid]
            alloc = allocation[sid]
            min_acc = min_acceptable[sid]
            satisfaction = alloc / req if req > 0 else 1.0
            shortfall = max(0.0, min_acc - alloc)
            critical_failure = alloc < min_acc - 1e-6
            own_moves = [m for m in moves_today if m.stakeholder_id == sid]
            cooperated = any(m.move_type in ("accept", "concede") for m in own_moves)
            objected = any(m.move_type == "object" for m in own_moves)
            outcome_rows.append(tag({
                "day": day, "stakeholder_id": sid, "name": STAKEHOLDER_BY_ID[sid].name,
                "requested": req, "min_acceptable": min_acc, "allocated": alloc,
                "satisfaction": round(satisfaction, 3), "shortfall": round(shortfall, 1),
                "critical_failure": critical_failure, "severity_today": is_severe,
                "rounds_today": round_no, "imposed_today": imposed,
                "cooperated": cooperated, "objected": objected,
                "supply": supply, "total_allocated": total_allocated, "peak_temp_c": peak_temp_c,
                "ruling_text": ruling_text,
            }))

        # Phase 6: memory writes — own outcome + observed peer behaviour + the ruling.
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

        # Phase 7: reflection for every agent
        for a in agents.values():
            maybe_reflect(a["stream"], now_hours=day, seed=seed)

        if verbose:
            for sid in demander_ids:
                print(f'  {STAKEHOLDER_BY_ID[sid].name:30} req={requested_units[sid]:6.0f} '
                      f'alloc={allocation[sid]:6.0f}')
            print(f'  severity={is_severe} rounds={round_no} imposed={imposed}')

    return pd.DataFrame(decision_rows), pd.DataFrame(outcome_rows), agents


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
        dec_df, out_df, _ = run_simulation(run_config, run_id=run_id, verbose=verbose)
        with print_lock:
            completed += 1
            elapsed = time.time() - start_time
            avg = elapsed / completed
            eta = avg * (total - completed)
            print(f"[{i+1}/{total}] finished {run_id}  "
                  f"(elapsed {elapsed/60:.1f} min, avg {avg/60:.1f} min/run, "
                  f"ETA {eta/60:.1f} min)", flush=True)
        return run_id, dec_df, out_df

    if max_workers <= 1:
        for i, (run_id, run_config) in enumerate(run_specs):
            _, dec_df, out_df = _run_one(i, run_id, run_config)
            results[run_id] = (dec_df, out_df)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_one, i, run_id, run_config)
                       for i, (run_id, run_config) in enumerate(run_specs)]
            for f in futures:
                run_id, dec_df, out_df = f.result()
                results[run_id] = (dec_df, out_df)

    all_decisions = [results[run_id][0] for run_id, _ in run_specs]
    all_outcomes = [results[run_id][1] for run_id, _ in run_specs]
    return pd.concat(all_decisions, ignore_index=True), pd.concat(all_outcomes, ignore_index=True)


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
        conflicts = int((moves["move_type"] == "object").sum()) if n_moves else 0
        compromises = int((moves["move_type"] == "concede").sum()) if n_moves else 0
        trades = int((moves["move_type"] == "propose_trade").sum()) if n_moves else 0
        accepts = int((moves["move_type"] == "accept").sum()) if n_moves else 0
        summary_rows.append({
            "run_id": run_id, "condition": condition, "seed": seed,
            "n_days": int(day_group["day"].nunique()),
            "n_negotiation_days": int(day_group["severity"].sum()),
            "n_negotiation_rounds_total": int(day_group["rounds"].sum()),
            "n_conflicts": conflicts,
            "n_compromises": compromises,
            "n_trades_proposed": trades,
            "cooperation_rate": round((compromises + accepts) / n_moves, 3) if n_moves else float("nan"),
            "n_critical_failures": int(day_group["critical_failures"].sum()),
            "mean_fairness_gini": round(day_group["fairness_gini"].mean(), 3),
            "mean_collective_welfare_utilitarian": round(day_group["mean_satisfaction"].mean(), 3),
            "mean_collective_welfare_rawlsian": round(day_group["min_satisfaction"].mean(), 3),
            "system_stability_gini_std": round(day_group["fairness_gini"].std(ddof=0), 3),
        })

    return per_run_day_df, pd.DataFrame(summary_rows)


__all__ = [
    # world
    "WATER_SUPPLY_SCHEDULE", "N_DAYS", "total_supply", "demand_escalation",
    "TEMPERATURE_SCHEDULE", "total_temperature",
    # stakeholders
    "Stakeholder", "STAKEHOLDERS", "STAKEHOLDER_BY_ID", "DEMANDER_IDS",
    "NEGOTIATION_TOPOLOGY", "demand_today", "min_acceptable_today",
    # negotiation protocol
    "Request", "NegotiationMove", "clear_allocation", "check_severity",
    "apply_objection_bump", "execute_trade", "MAX_NEGOTIATION_ROUNDS",
    # cache + cost
    "client", "_cache", "_usage", "PRICING_PER_TOKEN", "CACHE_DIR",
    "llm", "embed", "print_cost_summary",
    # memory
    "Memory", "MemoryStream", "rate_importance", "cosine", "normalize", "retrieve",
    "importance_sum_of_recent", "maybe_reflect",
    # decision loop
    "estimate_need", "negotiation_move", "authority_ruling",
    # engine
    "RunConfig", "DEFAULT_PRECEDENT_MEMORIES", "initialise_agents", "run_simulation", "run_batch", "compute_metrics",
]