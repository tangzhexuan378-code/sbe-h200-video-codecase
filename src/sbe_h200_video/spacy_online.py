from __future__ import annotations

import importlib.util
import json
import math
import re
from functools import lru_cache
from typing import Any


FEATURE_ORDER = [
    "attribute",
    "presence",
    "containment",
    "spatial",
    "state_change",
    "counting",
    "contact",
    "transfer",
    "motion",
    "unknown",
]

DEFAULT_WEIGHTS = {
    "attribute": 0.20,
    "presence": 0.40,
    "containment": 0.50,
    "spatial": 1.00,
    "state_change": 1.00,
    "counting": 1.20,
    "contact": 1.10,
    "transfer": 1.10,
    "motion": 0.50,
    "unknown": 1.50,
}

COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
    "grey",
    "silver",
    "gold",
    "golden",
    "brown",
    "navy",
    "crimson",
    "emerald",
    "turquoise",
}

MATERIAL_WORDS = {
    "wooden",
    "metal",
    "ceramic",
    "glass",
    "plastic",
    "leather",
    "cotton",
    "wool",
    "marble",
    "transparent",
    "clear",
}

CONTAINMENT_PATTERNS = (
    "inside",
    "in a",
    "in an",
    "into",
    "within",
    "contained",
)

SPATIAL_PATTERNS = (
    "left of",
    "right of",
    "above",
    "below",
    "behind",
    "in front of",
    "next to",
    "beside",
    "between",
    "near",
)

STATE_WORDS = {
    "open",
    "closed",
    "lit",
    "unlit",
    "folded",
    "inflated",
    "deflated",
    "empty",
    "full",
    "peeled",
    "zipped",
    "unzipped",
    "broken",
    "burning",
    "wet",
    "dry",
    "wrapped",
    "unwrapped",
    "turned",
}

COUNT_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "single",
    "exactly",
    "several",
    "many",
    "multiple",
}

CONTACT_WORDS = {
    "touch",
    "touching",
    "contact",
    "hold",
    "holding",
    "grasp",
    "grabbing",
    "pick",
    "picks",
    "picking",
    "hand",
}

TRANSFER_WORDS = {
    "pick",
    "picks",
    "picked",
    "pour",
    "pours",
    "pouring",
    "move",
    "moves",
    "moving",
    "transfer",
    "transfers",
    "place",
    "places",
    "put",
    "puts",
}

MOTION_WORDS = {
    "walk",
    "walks",
    "walking",
    "run",
    "runs",
    "running",
    "rotate",
    "rotates",
    "rotating",
    "roll",
    "rolling",
    "fly",
    "flying",
    "fall",
    "falling",
    "pour",
    "pouring",
    "pick",
    "picking",
}

ENTITY_HINTS = {
    "cup",
    "bowl",
    "box",
    "plate",
    "spoon",
    "chair",
    "table",
    "desk",
    "hand",
    "person",
    "man",
    "woman",
    "apple",
    "book",
    "phone",
    "laptop",
    "glass",
    "candle",
    "ball",
    "cube",
    "bottle",
    "vase",
}


@lru_cache(maxsize=1)
def load_spacy_model() -> tuple[Any, str]:
    if importlib.util.find_spec("spacy") is None:
        return None, "missing_spacy"
    import spacy

    try:
        return spacy.load("en_core_web_sm"), "en_core_web_sm"
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp, "spacy_blank_rule_fallback"


def _tokens(prompt: str) -> tuple[list[str], str]:
    nlp, source = load_spacy_model()
    if nlp is None:
        return re.findall(r"[A-Za-z0-9']+", prompt.lower()), source
    doc = nlp(prompt)
    words = []
    for token in doc:
        lemma = getattr(token, "lemma_", "") or token.text
        text = lemma.lower() if lemma != "-PRON-" else token.text.lower()
        if re.search(r"[A-Za-z0-9]", text):
            words.append(text)
    return words, source


def extract_features(prompt: str) -> dict[str, Any]:
    words, parser_source = _tokens(prompt)
    text = " ".join(words)
    raw = prompt.lower()

    features = {name: 0.0 for name in FEATURE_ORDER}
    features["attribute"] = float(any(word in COLOR_WORDS or word in MATERIAL_WORDS for word in words))
    features["presence"] = float(any(word in ENTITY_HINTS for word in words) or bool(re.search(r"\b(a|an|one|single)\b", raw)))
    features["containment"] = float(any(pattern in raw for pattern in CONTAINMENT_PATTERNS))
    features["spatial"] = float(any(pattern in raw for pattern in SPATIAL_PATTERNS))
    features["state_change"] = float(any(word in STATE_WORDS for word in words))
    features["counting"] = float(any(word in COUNT_WORDS for word in words) or bool(re.search(r"\b[2-9]\b", raw)))
    features["contact"] = float(any(word in CONTACT_WORDS for word in words))
    features["transfer"] = float(any(word in TRANSFER_WORDS for word in words))
    features["motion"] = float(any(word in MOTION_WORDS for word in words))

    entity_count = sum(1 for word in words if word in ENTITY_HINTS)
    verb_like_count = sum(1 for word in words if word in MOTION_WORDS or word in TRANSFER_WORDS or word in CONTACT_WORDS)
    vague_count = sum(1 for word in words if word in {"something", "object", "thing", "somehow", "strange", "various"})
    ambiguous_relation = 1 if "near" in words or "around" in words else 0

    uncertainty = 0.0
    if parser_source != "en_core_web_sm":
        uncertainty += 0.05
    if vague_count:
        uncertainty += min(0.45, 0.15 * vague_count)
    if entity_count >= 4:
        uncertainty += 0.10
    if verb_like_count >= 2:
        uncertainty += 0.10
    if ambiguous_relation:
        uncertainty += 0.10
    uncertainty = min(1.0, uncertainty)

    if not any(features[name] for name in FEATURE_ORDER[:-1]) or uncertainty >= 0.50:
        features["unknown"] = 1.0

    return {
        "prompt": prompt,
        "parser_source": parser_source,
        "tokens": words,
        "features": features,
        "uncertainty": round(float(uncertainty), 4),
        "entity_count": entity_count,
        "verb_like_count": verb_like_count,
        "text_normalized": text,
    }


def risk_score(parsed: dict[str, Any], weights: dict[str, float] | None = None, uncertainty_lambda: float = 0.5) -> float:
    weight_map = dict(DEFAULT_WEIGHTS)
    if weights:
        weight_map.update({str(k): float(v) for k, v in weights.items()})
    features = parsed["features"]
    base = sum(weight_map[name] * float(features.get(name, 0.0)) for name in FEATURE_ORDER)
    return round(float(base + uncertainty_lambda * float(parsed["uncertainty"])), 4)


def continuous_q(risk: float, uncertainty: float, cfg: dict[str, Any]) -> float:
    r0 = float(cfg.get("r0", 0.6))
    scale = float(cfg.get("scale", 3.0))
    eta = float(cfg.get("eta", 0.35))
    value = ((float(risk) - r0) / max(scale, 1e-6)) + eta * float(uncertainty)
    return round(float(min(1.0, max(0.0, value))), 4)


def continuous_threshold_schedule(
    prompt: str,
    steps: int,
    cfg: dict[str, Any],
    *,
    use_uncertainty: bool = True,
) -> dict[str, Any]:
    parsed = extract_features(prompt)
    uncertainty = float(parsed["uncertainty"]) if use_uncertainty else 0.0
    parsed_for_risk = dict(parsed)
    parsed_for_risk["uncertainty"] = uncertainty
    risk = risk_score(parsed_for_risk, cfg.get("weights", {}), float(cfg.get("uncertainty_lambda", 0.5)))
    q = continuous_q(risk, uncertainty, cfg)

    base = list(cfg.get("base_thresholds", []))
    if not base:
        base = [None, None, 0.18, 0.25, 0.32, 0.35, 0.35, 0.32, 0.25, 0.18, None, None]
    if len(base) != steps:
        if len(base) < steps:
            base = base + [None] * (steps - len(base))
        else:
            base = base[:steps]

    alpha = float(cfg.get("alpha", 0.52))
    beta = float(cfg.get("beta", 0.025))
    floor = float(cfg.get("floor", 0.105))
    schedule: list[float | None] = []
    for item in base:
        if item is None:
            schedule.append(None)
            continue
        tau = float(item) * (1.0 - alpha * q) - beta * q
        schedule.append(None if tau < floor else round(float(tau), 4))

    return {
        "parsed": parsed,
        "risk": risk,
        "q": q,
        "use_uncertainty": use_uncertainty,
        "threshold_schedule": schedule,
        "threshold_schedule_json": json.dumps(schedule, ensure_ascii=False),
        "features_json": json.dumps(parsed["features"], ensure_ascii=False, sort_keys=True),
    }


def discrete_threshold_schedule(prompt: str, steps: int, cfg: dict[str, Any]) -> dict[str, Any]:
    result = continuous_threshold_schedule(prompt, steps, cfg, use_uncertainty=True)
    risk = float(result["risk"])
    low = float(cfg.get("discrete_low", 1.0))
    high = float(cfg.get("discrete_high", 2.4))
    if risk < low:
        schedule = cfg.get("discrete_low_schedule", [None, 0.20, 0.30, 0.35, 0.40, 0.40, 0.40, 0.35, 0.30, 0.20, None, None])
        level = "low"
    elif risk < high:
        schedule = cfg.get("discrete_medium_schedule", [None, None, 0.15, 0.20, 0.25, 0.25, 0.25, 0.20, 0.15, None, None, None])
        level = "medium"
    else:
        schedule = cfg.get("discrete_high_schedule", [None, None, None, 0.15, 0.20, 0.20, 0.20, 0.15, None, None, None, None])
        level = "high"
    schedule = list(schedule)[:steps] + [None] * max(0, steps - len(schedule))
    result.update(
        {
            "q": {"low": 0.0, "medium": 0.5, "high": 1.0}[level],
            "risk_level": level,
            "threshold_schedule": schedule,
            "threshold_schedule_json": json.dumps(schedule, ensure_ascii=False),
        }
    )
    return result
