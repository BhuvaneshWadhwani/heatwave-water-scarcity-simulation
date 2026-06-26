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
CACHE_FILE = CACHE_DIR / "llm_cache.json"

if CACHE_FILE.exists():
    with open(CACHE_FILE) as _f:
        _cache = json.load(_f)
else:
    _cache = {}

_env_file = Path(__file__).parent.parent.parent / ".env"
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


def _bump(model, kind, n):
    _usage["tokens"][(model, kind)] = _usage["tokens"].get((model, kind), 0) + n


def _cache_key(kind, model, payload):
    h = hashlib.sha256(json.dumps([kind, model, payload], sort_keys=True).encode()).hexdigest()[:16]
    return f"{kind}:{model}:{h}"


def _save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(_cache, f, indent=2)


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
    if key in _cache:
        _usage["calls"]["chat_cached"] += 1
        return _cache[key]
    if client is None:
        raise RuntimeError(f"Prompt not in cache and no API key set:\n{prompt[:200]}...")
    kwargs = dict(model=model, temperature=temperature, max_completion_tokens=max_tokens,
                  messages=[{"role": "user", "content": prompt}])
    if seed is not None:
        kwargs["seed"] = seed
    r = client.chat.completions.create(**kwargs)
    out = r.choices[0].message.content.strip()
    _bump(model, "in", r.usage.prompt_tokens)
    _bump(model, "out", r.usage.completion_tokens)
    _usage["calls"]["chat_live"] += 1
    _cache[key] = out
    _save_cache()
    return out


def embed(text, model="text-embedding-3-small"):
    """Embed one string, cached by (model, text)."""
    key = _cache_key("embed", model, {"text": text})
    if key in _cache:
        _usage["calls"]["embed_cached"] += 1
        return np.array(_cache[key])
    if client is None:
        raise RuntimeError(f"Embedding not in cache and no API key set: {text[:80]}")
    r = client.embeddings.create(model=model, input=text)
    vec = r.data[0].embedding
    _bump(model, "in", r.usage.prompt_tokens)
    _usage["calls"]["embed_live"] += 1
    _cache[key] = vec
    _save_cache()
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

WATER_SUPPLY_SCHEDULE = [1000, 900, 800, 700, 650, 600]   # units/day, days 0..5
N_DAYS = len(WATER_SUPPLY_SCHEDULE)
TEMPERATURE_SCHEDULE = [36, 37, 38, 39, 40, 41]           # peak °C, days 0..5


def total_supply(day: int) -> float:
    """Total raw water available on a given day."""
    return float(WATER_SUPPLY_SCHEDULE[day])


def demand_escalation(stakeholder_id: str, day: int) -> float:
    """Deterministic multiplier on a stakeholder's baseline demand as the
    heatwave progresses. This is the "physics" layer — analogous to the
    original sim's temperature model — and is intentionally not an LLM
    decision: physical/operational need grows independently of strategy.
    """
    if stakeholder_id == "agriculture":
        return 1.0 + 0.05 * day          # cumulative crop/livestock stress
    if stakeholder_id == "energy_utility":
        return 1.0 + 0.04 * day          # cooling load rises with heat
    if stakeholder_id == "households":
        return 1.0 + 0.03 * day          # personal cooling/hygiene use rises
    if stakeholder_id == "hospital":
        return 1.0 + 0.02 * day          # heat-related admissions rise modestly
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
        voice="Represents the region's hospitals. You have high statutory priority, but you are not isolated: you completely depend on the Energy Utility for power, and on Industry/Logistics for medical supplies.",
        base_demand=150.0, min_acceptable_frac=0.85, priority_weight=5.0,
        priority_arguments=["Direct, immediate risk to patient life and safety",
                            "Public health collapse will affect everyone"],
        #DEPENDANCY FROM OTHERS:
        failure_conditions="Below minimum: forced to ration sanitation. CASCADING RISK: If Energy Utility or Industry fails, the hospital will lose power and vital supplies, leading to total collapse.",
        strategy="Lead with patient-safety framing. Be extremely protective of your sanitation floor, BUT you must proactively propose trades or concede slightly to ensure Energy Utility and Industry do not collapse, as their failure means your failure.",
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
        voice="Represents regional manufacturing. You have the lowest statutory priority, but you are the backbone of the region's supply chain (including medical and infrastructure supplies).",
        base_demand=150.0, min_acceptable_frac=0.72, priority_weight=1.0,
        priority_arguments=["Production shutdowns will break the supply chain for hospitals and utilities",
                            "Immediate job losses and economic collapse"],
        failure_conditions="Below minimum: forced production cuts, breaking supply chains for other stakeholders.",
        # INDUSTRY REMINDS ABOUT ITS IMPORTANCE:
        strategy="You have low priority, so you must explicitly remind High-priority agents (like Hospitals) that they depend on your supply chains. Propose trades with them to ensure mutual survival.",
    ),
    Stakeholder(
        id="energy_utility", name="Energy Utility", role="demander",
        objective="Maintain cooling water for power generation to avoid outages.",
        voice="Represents the regional power utility. You provide electricity to all other stakeholders.",
        base_demand=100.0, min_acceptable_frac=0.90, priority_weight=4.0,
        priority_arguments=["A cooling-water shortfall risks a regional power outage",
                            "Outage would cascade into every other stakeholder's failure mode"],
        failure_conditions="Below minimum: risk of forced generation curtailment or blackout, destroying Hospital and Industry operations.",
        strategy="Leverage your role as the grid provider. Demand cooperation from Hospitals and Households by threatening mutual destruction via blackouts.",
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
    "hospital": ["energy_utility", "industry"],  # Now hospital can propose trades to both energy_utility and industry
    "households": ["agriculture"],           
    "agriculture": ["industry", "households"],
    "industry": ["agriculture", "energy_utility", "hospital"],
    "energy_utility": ["industry", "hospital"],
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


# =============================================================================
# 4. Negotiation protocol: requests, deterministic clearing, severity
# =============================================================================

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
    move_type: str                  # "accept" | "concede" | "object" | "propose_trade" | "hold"
    detail: str                      # free text: what was conceded / objected / traded
    revised_min_acceptable: Optional[float] = None   # set if move_type == "concede"
    trade_target: Optional[str] = None               # set if move_type == "propose_trade"
    trade_units: Optional[float] = None              # set if move_type == "propose_trade"


def clear_allocation(requests: dict, min_acceptable: dict, priority_weights: dict,
                      supply: float) -> dict:
    """Deterministic two-phase clearing. Always returns an allocation that
    sums to at most `supply`, regardless of what any LLM call proposed.

    Phase 1 — guarantee minimums in priority order, highest weight first.
    Phase 2 — distribute any remaining supply proportional to priority-weighted
    unmet request.
    """
    ids = list(requests.keys())
    allocation = {i: 0.0 for i in ids}
    remaining = float(supply)

    # Phase 1: remaining minimums in priority order (ties broken by id for determinism)
    order = sorted(ids, key=lambda i: (-priority_weights.get(i, 0.0), i))
    for i in order:
        need = min(min_acceptable.get(i, 0.0), requests[i])
        still_need = max(0.0, need - allocation[i])
        give = min(still_need, remaining)
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
PRIORITY_PENALTY_PER_OBJECTION = 0.5
PRIORITY_FLOOR = 0.5

# Independent variable: advocacy framing injected into all agent prompts.
# In the treatment condition agents with surplus are reminded they depend on
# the very peers they could help — making sharing rational, not charitable.
ADVOCACY_ENVIRONMENT_TEXT = (
    "\nINTERDEPENDENCE NOTICE: No sector survives this crisis alone. "
    "Hospitals depend on the Energy Utility for power and on Industry for medical supplies. "
    "The Energy Utility depends on functioning industrial supply chains. "
    "Industry and Agriculture sustain the food and goods that every institution — including yours — relies on. "
    "If a critical peer collapses, the cascade will reach you too. "
    "If you are receiving adequate water today, voluntary surplus sharing is not generosity: "
    "it is protecting the system your own operations depend on."
)

def apply_objection_bump(priority_weights: dict, objecting_ids: list, base_weights: dict) -> dict:
    """
    Каждое возражение (OBJECT) теперь СТОИТ агенту политического капитала.
    Вместо того чтобы награждать их за упрямство, мы штрафуем их приоритетный вес 
    в текущем дне. Это делает бесконечный отказ математически невыгодным.
    """
    bumped = dict(priority_weights)
    for i in objecting_ids:
        # Уменьшаем приоритет за каждое возражение, но не ниже установленного минимума
        bumped[i] = max(PRIORITY_FLOOR, 
                        bumped.get(i, base_weights[i]) - PRIORITY_PENALTY_PER_OBJECTION)
    return bumped

# ============================================================
# 5. Cognitive scaffold
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

    def add(self, content, created_at, importance, with_embedding=True):
        m = Memory(
            content=content,
            created_at=created_at,
            last_accessed=created_at,
            importance=importance,
            embedding=embed(content) if with_embedding else None,
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


# ============================================================
# 7. Decision loop: need estimation, negotiation moves, authority ruling
# ============================================================
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

Today is Day {day}. Your baseline operational need today (before any strategic shading) is \
{demand:.0f} units. Your stated minimum acceptable amount, below which your failure \
conditions are triggered, is {min_acceptable:.0f} units.
Your failure conditions: {failure_conditions}

Your recent institutional memory:
{memories_block}
{advocacy_block}
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
                   demand: float, min_acceptable: float, seed=None,
                   advocacy_block: str = "") -> Request:
    """`demand`/`min_acceptable` are passed in rather than recomputed here,
    so the caller (run_simulation, driven by a RunConfig) controls the
    "physics" for this run — e.g. a scaled-demand or shortened-schedule
    experimental condition — without this function needing to know about
    experimental conditions at all."""
    query = f"What should I request today given the water shortage? Day {day}."
    top = retrieve(stream, query, now_hours=day, k=5)
    memories_block = "\n".join(f"  - {m.content}" for m in top) or "  (no relevant memories yet)"

    prompt = NEED_ESTIMATION_PROMPT.format(
        name=stakeholder.name, role=stakeholder.role, voice=stakeholder.voice,
        objective=stakeholder.objective, strategy=stakeholder.strategy,
        day=day, demand=demand, min_acceptable=min_acceptable,
        failure_conditions=stakeholder.failure_conditions,
        memories_block=memories_block, advocacy_block=advocacy_block,
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
{shortfall_desc}
Your failure conditions: {failure_conditions}

WARNING FROM WATER AUTHORITY: Each time you OBJECT, your institutional priority weight is \
reduced by {penalty:.1f} for this day's re-allocation (floor: {floor:.1f}). After {max_rounds} \
rounds without agreement the Authority imposes a final allocation based on remaining weights — \
persistent objectors receive LESS, not more. Your best move is to CONCEDE or PROPOSE_TRADE.

YESTERDAY'S SYSTEM STATUS:
{yesterday_status_block}
{advocacy_block}
{trade_block}

Your recent institutional memory:
{memories_block}

Choose ONE move:
ACCEPT — accept your current allocation as-is (signal of cooperation or satisfaction).
CONCEDE — lower your stated minimum for today in exchange for something (state what you want in return, e.g. priority tomorrow).
OBJECT — refuse to accept, citing your failure conditions, and push the Authority to revise (WARNING: uses political capital and risks emergency reduction).
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
    if move_type not in ("accept", "concede", "object", "propose_trade", "hold"):
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
                      yesterday_status_block: str = "  (no prior data)", seed=None,
                      advocacy_block: str = "") -> NegotiationMove:
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

    shortfall_val = max(0.0, min_acceptable - proposed)
    if shortfall_val > 0:
        shortfall_desc = f"{shortfall_val:.0f} units below your minimum of {min_acceptable:.0f}"
    else:
        shortfall_desc = "this meets or exceeds your minimum"

    prompt = NEGOTIATION_MOVE_PROMPT.format(
        name=stakeholder.name, role=stakeholder.role, day=day, round=round_no,
        peak_temp_c=peak_temp_c if peak_temp_c is not None else 38,
        objective=stakeholder.objective, strategy=stakeholder.strategy,
        requested=requested, min_acceptable=min_acceptable, proposed=proposed,
        shortfall_desc=shortfall_desc,
        failure_conditions=stakeholder.failure_conditions,
        trade_block=trade_block, trade_option=trade_option, trade_format=trade_format,
        memories_block=memories_block, yesterday_status_block=yesterday_status_block,
        penalty=PRIORITY_PENALTY_PER_OBJECTION, floor=PRIORITY_FLOOR,
        max_rounds=MAX_NEGOTIATION_ROUNDS, advocacy_block=advocacy_block,
    )
    raw = llm(prompt, temperature=0.7, max_tokens=150, seed=seed)
    move_type, detail, revised_min, trade_target, trade_units = _parse_move(raw, current_min=min_acceptable)
    return NegotiationMove(stakeholder_id=stakeholder.id, day=day, round=round_no,
                            move_type=move_type, detail=detail, revised_min_acceptable=revised_min,
                            trade_target=trade_target, trade_units=trade_units)


PROACTIVE_MOVE_PROMPT = '''You are {name} ({role}) during a severe water shortage. \
It is Day {day}, negotiation round {round}.

Background: {voice}
Your strategy: {strategy}

Your current allocation: {allocated:.0f} units. Your minimum: {min_acceptable:.0f} units.
Your SURPLUS above minimum: {surplus:.0f} units — these are units you can voluntarily spare.

SYSTEM STATUS (yesterday's outcomes):
{yesterday_status_block}

ALERT: Peers marked COLLAPSED or BELOW MINIMUM are in crisis. If critical infrastructure \
such as Industry or Energy Utility collapses, your own operations may face cascading failure \
through broken supply chains or blackouts.
{advocacy_block}
Your recent institutional memory:
{memories_block}

You may propose a voluntary trade with peers in crisis: {peers}

Choose ONE action:
HOLD — keep your current allocation, do not intervene.
PROPOSE_TRADE — voluntarily give some of your SURPLUS to a peer in crisis (you keep at least your minimum).

Respond in exactly this format, nothing else:
MOVE: <HOLD|PROPOSE_TRADE>
DETAIL: <one sentence explaining your reasoning in {name}'s voice>
TRADE_TARGET: <one peer id from the list above, only if MOVE is PROPOSE_TRADE, otherwise NONE>
TRADE_UNITS: <units to give, only if MOVE is PROPOSE_TRADE, otherwise NONE>'''


def _parse_proactive_move(raw):
    move_match = re.search(r"MOVE:\s*([A-Z_]+)", raw)
    detail_match = re.search(r"DETAIL:\s*(.+?)(?:\nTRADE_TARGET:|$)", raw, re.DOTALL)
    target_match = re.search(r"TRADE_TARGET:\s*(\w+)", raw)
    units_match = re.search(r"TRADE_UNITS:\s*([\d.]+)", raw)

    move_type = (move_match.group(1).lower() if move_match else "hold")
    if move_type not in ("hold", "propose_trade"):
        move_type = "hold"
    detail = detail_match.group(1).strip() if detail_match else raw.strip()
    trade_target = None
    trade_units = None
    if move_type == "propose_trade":
        trade_target = target_match.group(1) if target_match else None
        trade_units = float(units_match.group(1)) if units_match else None
    return move_type, detail, trade_target, trade_units


def proactive_negotiation_move(stakeholder: Stakeholder, day: int, round_no: int,
                                allocated: float, min_acceptable: float,
                                affected_peers: list, yesterday_status_block: str,
                                stream: "MemoryStream", seed=None,
                                advocacy_block: str = "") -> NegotiationMove:
    """Called for agents that are ABOVE their minimum during a severe day.

    They can voluntarily offer surplus units to peers in crisis via PROPOSE_TRADE.
    This is the mechanism that creates emergent cooperation — well-resourced agents
    proactively helping failing peers to prevent cascading supply-chain collapse.
    """
    peers = [p for p in NEGOTIATION_TOPOLOGY.get(stakeholder.id, []) if p in affected_peers]
    if not peers:
        return NegotiationMove(stakeholder_id=stakeholder.id, day=day, round=round_no,
                                move_type="hold", detail="No eligible peers currently in crisis.")
    surplus = allocated - min_acceptable
    query = f"Should I share resources with peers in crisis? Day {day}."
    top = retrieve(stream, query, now_hours=day, k=4)
    memories_block = "\n".join(f"  - {m.content}" for m in top) or "  (no relevant memories yet)"
    prompt = PROACTIVE_MOVE_PROMPT.format(
        name=stakeholder.name, role=stakeholder.role, day=day, round=round_no,
        voice=stakeholder.voice, strategy=stakeholder.strategy,
        allocated=allocated, min_acceptable=min_acceptable, surplus=surplus,
        yesterday_status_block=yesterday_status_block, memories_block=memories_block,
        peers=", ".join(peers), advocacy_block=advocacy_block,
    )
    raw = llm(prompt, temperature=0.7, max_tokens=150, seed=seed)
    move_type, detail, trade_target, trade_units = _parse_proactive_move(raw)
    return NegotiationMove(stakeholder_id=stakeholder.id, day=day, round=round_no,
                            move_type=move_type, detail=detail,
                            trade_target=trade_target, trade_units=trade_units)


AUTHORITY_RULING_PROMPT = '''You are the Municipal Water Authority, ruling on Day {day}'s water \
allocation during a severe shortage.

Total supply available today: {supply:.0f} units.
Final allocation reached: {allocation_block}

{context_block}

Write a short (2-3 sentence) public justification for today's allocation, in the Water \
Authority's voice, that a stakeholder reading it later would recognise as the actual basis for \
the decision. Do not restate the numbers; explain the reasoning.'''


def authority_ruling(day: int, supply: float, allocation: dict, context_block: str,
                      stream: "MemoryStream", seed=None) -> str:
    allocation_block = ", ".join(f"{STAKEHOLDER_BY_ID[i].name}: {v:.0f}" for i, v in allocation.items())
    prompt = AUTHORITY_RULING_PROMPT.format(
        day=day, supply=supply, allocation_block=allocation_block, context_block=context_block,
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
    with_seeded_precedent: bool = True
    seed: Optional[int] = None
    advocacy_framing: bool = False   # IV: inject interdependence advocacy into all LLM prompts

    @property
    def n_days(self) -> int:
        return len(self.supply_schedule)

    def supply(self, day: int) -> float:
        return float(self.supply_schedule[day])

    def temperature(self, day: int) -> float:
        if day < len(self.temperature_schedule):
            return float(self.temperature_schedule[day])
        return float(self.temperature_schedule[-1])

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


def initialise_agents(config: RunConfig):
    """Build a fresh set of agents with seeded institutional backstories.

    `config.with_seeded_precedent` adds Agriculture a high-importance memory
    of a past drought in which it was deprioritised — the institutional
    analogue of the original sim's seeded "grim trigger". Toggle off via the
    RunConfig to ablate path-dependency effects.
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
        if config.with_seeded_precedent and sid == "agriculture":
            stream.add(
                "In a previous regional drought, the Water Authority deprioritised "
                "Agriculture in favour of Industry, and farmers have not forgotten it.",
                created_at=0.0, importance=9,
            )
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
    advocacy_block = ADVOCACY_ENVIRONMENT_TEXT if config.advocacy_framing else ""

    decision_rows, outcome_rows = [], []
    prev_allocation: dict = {}
    prev_min_acc: dict = {}

    def tag(row):
        row["run_id"] = run_id
        row["condition"] = config.condition_label
        row["seed"] = seed
        return row

    for day in range(config.n_days):
        if verbose:
            print(f"--- Day {day} ({run_id}) ---")
        supply = config.supply(day)
        peak_temp_c = config.temperature(day)

        # Build yesterday's status block for LLM context
        if prev_allocation:
            status_lines = []
            for _sid in demander_ids:
                _alloc = prev_allocation.get(_sid, 0)
                _min = prev_min_acc.get(_sid, 0)
                _name = STAKEHOLDER_BY_ID[_sid].name
                if _alloc == 0 and _min > 0:
                    _status = "COLLAPSED (zero allocation)"
                elif _alloc < _min - 1e-6:
                    _pct = int(_alloc / _min * 100) if _min > 0 else 0
                    _status = f"BELOW MINIMUM ({_alloc:.0f} of {_min:.0f} units, {_pct}%)"
                else:
                    _status = f"Stable ({_alloc:.0f} units)"
                status_lines.append(f"  {_name}: {_status}")
            yesterday_status_block = "\n".join(status_lines)
        else:
            yesterday_status_block = "  (first day — no prior allocation data)"

        # Phase 1: need estimation (LLM, one call per demander/advocate)
        requests = {}
        for sid in demander_ids:
            demand = config.demand(sid, day)
            min_acc = config.min_acceptable(sid, day)
            req = estimate_need(STAKEHOLDER_BY_ID[sid], day, agents[sid]["stream"],
                                 demand=demand, min_acceptable=min_acc, seed=seed,
                                 advocacy_block=advocacy_block)
            requests[sid] = req
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
            for sid in demander_ids:
                if sid in affected:
                    move = negotiation_move(
                        STAKEHOLDER_BY_ID[sid], day, round_no,
                        requested_units[sid], min_acceptable[sid], allocation[sid],
                        agents[sid]["stream"], peak_temp_c=peak_temp_c,
                        yesterday_status_block=yesterday_status_block, seed=seed,
                        advocacy_block=advocacy_block,
                    )
                else:
                    # Non-affected agents: act only if they have surplus and a peer in crisis
                    spare = allocation[sid] - min_acceptable[sid]
                    peers_in_crisis = [p for p in NEGOTIATION_TOPOLOGY.get(sid, [])
                                       if p in affected]
                    if spare < 0.5 or not peers_in_crisis:
                        continue
                    move = proactive_negotiation_move(
                        STAKEHOLDER_BY_ID[sid], day, round_no,
                        allocation[sid], min_acceptable[sid],
                        affected_peers=affected,
                        yesterday_status_block=yesterday_status_block,
                        stream=agents[sid]["stream"], seed=seed,
                        advocacy_block=advocacy_block,
                    )
                    # HOLD is a silent neutral action — skip recording it
                    if move.move_type == "hold":
                        continue

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
                                        agents["water_authority"]["stream"], seed=seed)

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
            cooperated = any(m.move_type in ("accept", "concede", "propose_trade") for m in own_moves)
            objected = any(m.move_type == "object" for m in own_moves)
            outcome_rows.append(tag({
                "day": day, "stakeholder_id": sid, "name": STAKEHOLDER_BY_ID[sid].name,
                "requested": req, "min_acceptable": min_acc, "allocated": alloc,
                "satisfaction": round(satisfaction, 3), "shortfall": round(shortfall, 1),
                "critical_failure": critical_failure, "severity_today": is_severe,
                "rounds_today": round_no, "imposed_today": imposed,
                "cooperated": cooperated, "objected": objected,
                "supply": supply, "total_allocated": total_allocated,
                "ruling_text": ruling_text,
            }))

        # Phase 6: memory writes — own outcome + observed peer behaviour + the ruling
        for sid in demander_ids:
            req, alloc = requested_units[sid], allocation[sid]
            pct = (alloc / req * 100) if req > 0 else 100.0
            own_obs = f"Day {day}: requested {req:.0f} units, received {alloc:.0f} ({pct:.0f}% of request)."
            score, _ = rate_importance(own_obs, seed=seed)
            agents[sid]["stream"].add(own_obs, created_at=day, importance=score)

            for m in moves_today:
                if m.stakeholder_id == sid:
                    continue
                obs = _cooperation_observation(STAKEHOLDER_BY_ID[m.stakeholder_id].name, m)
                if obs:
                    obs = f"Day {day}: {obs}"
                    score2, _ = rate_importance(obs, seed=seed)
                    agents[sid]["stream"].add(obs, created_at=day, importance=score2)

            ruling_obs = f"Day {day} Water Authority ruling: {ruling_text}"
            score3, _ = rate_importance(ruling_obs, seed=seed)
            agents[sid]["stream"].add(ruling_obs, created_at=day, importance=score3)

        auth_obs = (f"Day {day}: allocated {total_allocated:.0f} of {supply:.0f} units across "
                    f"{len(demander_ids)} stakeholders; "
                    f"{'escalated negotiation' if is_severe else 'no escalation needed'}.")
        score4, _ = rate_importance(auth_obs, seed=seed)
        agents["water_authority"]["stream"].add(auth_obs, created_at=day, importance=score4)

        # Phase 7: reflection for every agent
        for a in agents.values():
            maybe_reflect(a["stream"], now_hours=day, seed=seed)

        # Update yesterday's state for next day's status block
        prev_allocation = dict(allocation)
        prev_min_acc = dict(min_acceptable)

        if verbose:
            for sid in demander_ids:
                print(f'  {STAKEHOLDER_BY_ID[sid].name:30} req={requested_units[sid]:6.0f} '
                      f'alloc={allocation[sid]:6.0f}')
            print(f'  severity={is_severe} rounds={round_no} imposed={imposed}')

    return pd.DataFrame(decision_rows), pd.DataFrame(outcome_rows), agents


def run_batch(configs: list, n_seeds: int = 1, base_seed: int = 0, verbose=False):
    """Run several experimental conditions (optionally x several seeds each)
    and return concatenated (decisions_df, outcomes_df) ready for aggregate
    analysis, e.g.:

        decisions_df, outcomes_df = run_batch([cfg_mild, cfg_severe], n_seeds=5)
        per_run_day, per_run_summary = compute_metrics(outcomes_df, decisions_df)
        per_run_summary.groupby("condition")[["mean_fairness_gini", ...]].agg(["mean", "std"])

    `agents` (memory streams) are intentionally not part of this return —
    for n_seeds x len(configs) runs, that's a lot of state to hold at once.
    If you need to inspect one run's memories qualitatively, call
    run_simulation() directly for that one (condition, seed).
    """
    all_decisions, all_outcomes = [], []
    for config in configs:
        for s in range(n_seeds):
            seed = base_seed + s
            run_id = f"{config.condition_label}_seed{seed}"
            run_config = RunConfig(**{**config.__dict__, "seed": seed})
            if verbose:
                print(f"=== Running {run_id} ===")
            dec_df, out_df, _ = run_simulation(run_config, run_id=run_id, verbose=verbose)
            all_decisions.append(dec_df)
            all_outcomes.append(out_df)
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
            "cooperation_rate": round((compromises + accepts + trades) / n_moves, 3) if n_moves else float("nan"),
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
    "estimate_need", "negotiation_move", "proactive_negotiation_move", "authority_ruling",
    # engine
    "RunConfig", "initialise_agents", "run_simulation", "run_batch", "compute_metrics",
]
