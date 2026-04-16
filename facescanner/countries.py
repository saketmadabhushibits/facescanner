"""Country-level mapping for the UTKFace ethnicity classes.

The Kaggle dataset `nipunarora8/age-gender-and-ethnicity-face-data-csv` uses
the UTKFace encoding:
    0 = White           (European / Middle-Eastern / North-African white)
    1 = Black           (African / African-American)
    2 = East Asian      (Chinese / Japanese / Korean / Vietnamese / ...)
    3 = Indian          (South Asian - Indian / Pakistani / Bangladeshi / ...)
    4 = Other           (Hispanic / Latino / Pacific Islander / mixed)

To produce the country-style percentage breakdown the user requested (e.g.
"24% African, 48% Indian, 18% Chinese"), we split each class's probability
across representative countries using population-weighted priors. This is a
reasonable approximation: within each macro-ethnicity we default to the
relative demographic sizes of the represented groups.
"""
from __future__ import annotations

from typing import Dict, List

# Macro-class -> list of (country, weight) pairs.
# Weights are approximate population shares within each macro-class and are
# re-normalized to sum to 1.0.
COUNTRY_DISTRIBUTIONS: Dict[str, List[tuple]] = {
    "White": [
        ("English", 0.14),
        ("German", 0.12),
        ("Italian", 0.10),
        ("French", 0.09),
        ("Russian", 0.11),
        ("Spanish", 0.08),
        ("Polish", 0.05),
        ("Irish", 0.04),
        ("Greek", 0.03),
        ("Turkish", 0.08),
        ("Iranian", 0.06),
        ("Middle-Eastern", 0.10),
    ],
    "Black": [
        ("Nigerian", 0.20),
        ("Ethiopian", 0.12),
        ("Congolese", 0.10),
        ("Kenyan", 0.08),
        ("Ghanaian", 0.07),
        ("South African", 0.06),
        ("Senegalese", 0.05),
        ("Somali", 0.05),
        ("Sudanese", 0.05),
        ("African-American", 0.15),
        ("Afro-Caribbean", 0.07),
    ],
    "East Asian": [
        ("Chinese", 0.48),
        ("Japanese", 0.12),
        ("Korean", 0.09),
        ("Vietnamese", 0.09),
        ("Filipino", 0.08),
        ("Thai", 0.06),
        ("Indonesian", 0.05),
        ("Malaysian", 0.03),
    ],
    "Indian": [
        ("Indian", 0.70),
        ("Pakistani", 0.13),
        ("Bangladeshi", 0.10),
        ("Nepali", 0.04),
        ("Sri Lankan", 0.03),
    ],
    "Other": [
        ("Mexican", 0.25),
        ("Brazilian", 0.18),
        ("Colombian", 0.10),
        ("Argentinian", 0.07),
        ("Peruvian", 0.06),
        ("Caribbean-Latino", 0.09),
        ("Pacific Islander", 0.08),
        ("Native American", 0.07),
        ("Mixed / Multiracial", 0.10),
    ],
}

# Human-friendly macro labels
MACRO_LABELS = {
    "White": "European / Middle-Eastern",
    "Black": "African",
    "East Asian": "East / Southeast Asian",
    "Indian": "South Asian",
    "Other": "Hispanic / Mixed",
}

# Class index -> canonical macro-name (matches UTKFace label ordering)
CLASS_NAMES = ["White", "Black", "East Asian", "Indian", "Other"]


def macro_to_countries(macro_probs: Dict[str, float]) -> List[Dict]:
    """Expand macro-ethnicity probabilities into country-level percentages.

    Parameters
    ----------
    macro_probs : dict
        Mapping from macro-class name (keys of COUNTRY_DISTRIBUTIONS) to
        probability in [0, 1]. Missing keys are treated as zero.

    Returns
    -------
    List of ``{country, macro, percentage}`` dicts, sorted descending by
    percentage, with percentages summing to ~100.
    """
    breakdown: Dict[str, Dict] = {}
    for macro, countries in COUNTRY_DISTRIBUTIONS.items():
        p = float(macro_probs.get(macro, 0.0))
        if p <= 0:
            continue
        total_w = sum(w for _, w in countries)
        for country, w in countries:
            share = p * (w / total_w)
            if country in breakdown:
                breakdown[country]["percentage"] += share * 100
            else:
                breakdown[country] = {
                    "country": country,
                    "macro": macro,
                    "macro_label": MACRO_LABELS[macro],
                    "percentage": share * 100,
                }

    results = sorted(breakdown.values(), key=lambda r: -r["percentage"])
    # Round for display but keep a raw field for the UI.
    for r in results:
        r["percentage"] = round(r["percentage"], 1)
    return results


def top_countries(breakdown: List[Dict], n: int = 5) -> List[Dict]:
    """Return the top-N countries, then a rolled-up "Other" entry if needed."""
    if len(breakdown) <= n:
        return breakdown
    top = breakdown[:n]
    remaining = sum(r["percentage"] for r in breakdown[n:])
    if remaining > 0.05:
        top = top + [{
            "country": "Other regions",
            "macro": "Other",
            "macro_label": "Other",
            "percentage": round(remaining, 1),
        }]
    return top


def macro_breakdown(macro_probs: Dict[str, float]) -> List[Dict]:
    """Return the macro-level breakdown sorted descending by percentage."""
    items = [
        {
            "macro": name,
            "label": MACRO_LABELS[name],
            "percentage": round(float(p) * 100, 1),
        }
        for name, p in macro_probs.items()
        if p > 0
    ]
    return sorted(items, key=lambda r: -r["percentage"])
