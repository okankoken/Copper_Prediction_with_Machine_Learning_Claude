"""Copper-specific headline cleaning and price-impact scoring.

FinBERT reads general financial tone. For copper that is not enough: a mine strike is
"negative" news but tightens supply and usually lifts the price. So every headline is
  1. kept only if it is about copper (copper itself, LME/COMEX/SHFE, or a named copper
     mine/producer) and is not a job ad, a market-research blurb, a product/alloy item
     or a near-duplicate of a headline seen in the previous 3 days;
  2. scored for its likely effect on the copper price with transparent rules:
       price   : copper/LME price rises (+) / falls (-)
       supply  : disruption, strike, outage, lower output, deficit, falling stocks (+)
                 output growth, restart, new supply, surplus, rising stocks (-)
       demand  : stronger demand or imports, stimulus (+) / weaker demand, recession (-)
     If no rule fires, a headline about the copper price falls back to half its FinBERT
     tone; otherwise it is neutral (0).
The score is in [-1, 1]; `reason` records which rule decided it.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

COPPER = re.compile(
    r"\bcopper\b|\bcu\b|\bred metal\b|\bbase metals?\b|\blme\b|\bcomex\b|\bshfe\b|codelco|escondida|collahuasi|"
    r"el teniente|chuquicamata|pelambres|centinela|quebrada blanca|grasberg|kamoa|kakula|oyu tolgoi|las bambas|"
    r"antamina|cerro verde|cobre panam|quellaveco|toquepala|cuajone|kghm|southern copper|freeport|first quantum|"
    r"antofagasta|jiangxi copper|zijin|mmg\b|hudbay|capstone|lundin mining|aurubis|ivanhoe|ok tedi|bingham|"
    r"kennecott|morenci|mopani|konkola|kansanshi|sentinel mine|andacollo|candelaria|caserones|mina justa|"
    r"constancia|tia maria|ministro hales|radomiro tomic|gabriela mistral|salvador division", re.I)

EXCLUDE = {
    "iş ilanı": re.compile(
        r"\b(engineer|manager|superintendent|geologist|technician|supervisor|coordinator|specialist|operator|"
        r"analyst|planner|foreman|electrician|mechanic|intern|metallurgist|accountant|officer|assistant)\b[^.]*\bat\b"
        r"|\bhiring\b|\bjob(s)? (opening|vacanc)|\bvacanc|\bcareers?\b|\brecruit", re.I),
    "pazar raporu": re.compile(
        r"market (size|share|report|research|analysis|insights?|outlook|trends)|\bcagr\b|\b20\d\d\s*[-–]\s*20\d\d\b|"
        r"market (to|will|is expected to|projected to) (reach|hit|grow|surpass)|industry report|forecast period", re.I),
    "ürün / alaşım": re.compile(
        r"beryllium|oxygen[- ]free|copper[- ](wire|tube|tubing|pipe|foil|clad|powder|sulfate|sulphate|plate|rod|busbar|"
        r"cookware|mug|bracelet|jewel|bottle|peptide|ion|iud|nanoparticle|alloy|strip|sheet|cable)s?\b|wire stripping|"
        r"copper-nickel|scrap market", re.I),
}

# (pattern, score, reason). Order matters only for the reason label; scores are summed.
# words that make "copper <word> <verb>" about something other than the price
_NOT_PRICE = (r"(?!(?:stocks|inventor\w*|output|production|imports?|exports?|demand|supply|smelt\w*|mines?|miners?|"
              r"mining|shipments|concentrate|deficit|surplus|project|deal|assets?|division)\b)")
PRICE_SUBJ = re.compile(r"(copper|lme|comex|shfe|red metal|base metals?)\b[^.]{0,12}?\b(price|prices|futures|contract)s?\b|"
                        r"\b(copper|red metal|lme copper|comex copper|shfe copper)\b\s+(" + _NOT_PRICE + r"\w+\s+){0,2}"
                        r"(ris(e|es|ing)|falls?|falling|gains?|drops?|climbs?|slips?|jumps?|slumps?|rall(y|ies)|surges?|"
                        r"tumbl\w*|rebounds?|retreats?|hits?|holds?|eases?|extends?|sinks?|soars?|steadies|edges?)", re.I)
UP = (r"(ris(e|es|en|ing)|rose|gain\w*|climb\w*|jump\w*|surg\w*|soar\w*|rall(y|ies|ied|ying)|rebound\w*|recover\w*|"
      r"record high|all-time high|highest|advanc\w*|hits? \$?[\d,.]+[- ]?(month|year)? high)")
DOWN = (r"(fall\w*|fell|drop\w*|slump\w*|tumbl\w*|slid(e|es|ing)?|slip\w*|sink\w*|sank|plung\w*|declin\w*|"
        r"retreat\w*|weak\w*|loss(es)?|los(e|es|ing)|lowest|eas(e|es|ed|ing))")

RULES = [
    # price direction (only when the headline's subject is the copper price)
    ("price", re.compile(rf"\b{UP}\b", re.I), +1.0, "fiyat yükselişi"),
    ("price", re.compile(rf"\b{DOWN}\b", re.I), -1.0, "fiyat düşüşü"),
    # supply side
    ("any", re.compile(r"\bstrikes?\b|walk-?out|\bprotests?\b|blockad|\bhalt(s|ed)?\b|suspen(d|ds|ded|sion)|"
                       r"shut(s|down)?\b|outage|force majeure|\bflood|mudslide|landslide|accident|collapse|"
                       r"disrupt|\bdeficit\b|shortage|tight(er|ening)? (supply|market)|supply (concern|worr|risk|crunch)|"
                       r"(supply|market|concentrate)\b[^.]{0,20}\btight|\bcrunch\b|\bsqueeze\b|\bstrik(ing|ers)\b|"
                       r"constrain|\bscarc(e|ity)\b", re.I), +1.0,
     "arz kesintisi / açığı"),
    ("any", re.compile(r"record highs?|all-time highs?|fresh high", re.I), +1.0, "fiyat yükselişi"),
    ("any", re.compile(r"expan(d|ds|ded|sion)\b|new mines?\b(?![^.]{0,20}\b(scarce|lacking|few|delay))|green ?light|"
                       r"go-ahead|approves? (\w+ ){0,2}(mine|project)", re.I), -0.5, "kapasite artışı (uzun vade)"),
    ("any", re.compile(r"(output|production|shipments|exports|supply|mined copper)\b[^.]{0,30}\b(fall|falls|fell|drop|drops|"
                       r"dropped|declin|slump|lower|cut|cuts|down)|(cut|cuts|lower|lowers|slash|slashes) "
                       r"(\w+ ){0,2}(output|production|guidance)|(drop|fall|decline|cut|reduction) in (\w+ ){0,2}"
                       r"(output|production|supply)", re.I), +1.0, "üretim düşüşü"),
    ("any", re.compile(r"(stocks|inventor(y|ies)|warehouse)\b[^.]{0,25}\b(fall|falls|fell|drop|drops|declin|lowest|draw)",
                       re.I), +0.7, "stok düşüşü"),
    ("any", re.compile(r"(output|production|shipments|exports|supply)\b[^.]{0,30}\b(ris(e|es|en|ing)\b|rose|increas|jump|climb|record|grow|"
                       r"higher|boost|up\b)|(increase|rise|growth|jump|boost) in (\w+ ){0,2}(output|production|supply)|"
                       r"ramp(s|ed)?[- ]up|restart|resum(e|es|ed|ption)|first (production|copper|"
                       r"concentrate)|new supply|\bsurplus\b|\bglut\b|oversuppl", re.I), -1.0, "arz artışı / fazlası"),
    ("any", re.compile(r"(stocks|inventor(y|ies)|warehouse)\b[^.]{0,25}\b(ris(e|es|en|ing)\b|rose|jump|climb|build|increas|highest|swell)",
                       re.I), -0.7, "stok artışı"),
    # demand side
    ("any", re.compile(r"demand\b[^.]{0,30}\b(ris(e|es|en|ing)\b|rose|surg|jump|strong|robust|boost|record|grow|doubl|"
                       r"soar|recover|boom)|(strong|robust|record|rising|booming|surging) demand|imports?\b[^.]{0,25}\b(ris|rose|jump|surg|"
                       r"record|climb|soar)|stimulus", re.I), +0.8, "talep artışı"),
    ("any", re.compile(r"demand\b[^.]{0,30}\b(weak|slow|fall|fell|drop|soft|sluggish|declin|cool|slump)|(weak|soft|"
                       r"sluggish|slowing|falling) demand|imports?\b[^.]{0,25}\b(fall|fell|drop|slump|declin|plung)|"
                       r"recession|slowdown", re.I), -0.8, "talep zayıflığı"),
]


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def clean(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Relevant, de-duplicated headlines + counts of what was removed and why."""
    h = h.copy()
    h["drop"] = ""
    has = lambda pat: h["title"].map(lambda t: bool(pat.search(t)))  # noqa: E731
    for reason, pat in EXCLUDE.items():
        h.loc[(h["drop"] == "") & has(pat), "drop"] = reason
    h.loc[(h["drop"] == "") & ~has(COPPER), "drop"] = "bakırla ilgisiz"
    h = h.sort_values("date")
    h["key"] = h["title"].map(norm_title)
    keep = h["drop"] == ""
    last_seen: dict[str, pd.Timestamp] = {}
    for i, row in h[keep].iterrows():
        prev = last_seen.get(row["key"])
        if prev is not None and (row["date"] - prev).days <= 3:
            h.at[i, "drop"] = "tekrar"
        else:
            last_seen[row["key"]] = row["date"]
    removed = h.loc[h["drop"] != "", "drop"].value_counts()
    return h[h["drop"] == ""].drop(columns=["drop", "key"]), removed


def impact(title: str, finbert_net: float = 0.0) -> tuple[float, str]:
    is_price = bool(PRICE_SUBJ.search(title))
    score, reasons = 0.0, []
    for scope, pat, s, why in RULES:
        if scope == "price" and not is_price:
            continue
        if scope == "any" and is_price and why in ("fiyat yükselişi", "fiyat düşüşü"):
            continue
        if pat.search(title):
            score += s
            reasons.append(why)
    # a price headline: the first direction word after the subject decides ("Copper rises on ..., weaker dollar")
    if is_price and any(r.startswith("fiyat") for r in reasons):
        start = PRICE_SUBJ.search(title).start()
        hits = [(m.start(), s, w) for _, p, s, w in RULES[:2] for m in [p.search(title, start)] if m] or \
               [(m.start(), s, w) for _, p, s, w in RULES[:2] for m in [p.search(title)] if m]
        _, score, why = min(hits)
        reasons = [why]
    if not reasons:
        if is_price and not np.isnan(finbert_net):
            return float(np.clip(0.5 * finbert_net, -1, 1)), "FinBERT tonu (fiyat haberi)"
        return 0.0, "nötr"
    return float(np.clip(score, -1, 1)), " + ".join(dict.fromkeys(reasons))


def score_frame(h: pd.DataFrame) -> pd.DataFrame:
    out = h.copy()
    res = [impact(t, n) for t, n in zip(out["title"], out.get("net", pd.Series(0.0, index=out.index)))]
    out["impact"] = [r[0] for r in res]
    out["reason"] = [r[1] for r in res]
    out["direction"] = np.where(out["impact"] > 0.15, "up", np.where(out["impact"] < -0.15, "down", "neutral"))
    return out
