"""Copper headline filter and price-impact rules."""
from __future__ import annotations

import unittest

import pandas as pd

from copperlab.headlines import clean, impact

CASES = [
    ("Copper holds above $14,400 as Chile strike risks escalate", +1),
    ("Escondida workers vote to strike", +1),
    ("Freeport declares force majeure at Grasberg after mudslide", +1),
    ("Chile copper output drops 5% in September", +1),
    ("Copper hits record high as tariff fears spark US buying", +1),
    ("Copper rises on supply risk, weaker dollar", +1),
    ("Copper Crunch Worsens as Key Spread Heads for Highest Since 2021", +1),
    ("Copper demand to double by 2035 as mining struggles to keep pace: Teck CEO", +1),
    ("Copper price set for first monthly loss since June as AI spending worries weigh", -1),
    ("Copper Prices Looked Unstoppable. Why They're Suddenly Tumbling", -1),
    ("Copper falls as Chile output rises", -1),
    ("Kamoa-Kakula output rises to record", -1),
    ("Peru anticipates slight increase in copper production", -1),
    ("LME copper inventories jump to highest in a year", -1),
    ("China copper imports slump as high prices curb buying", -1),
    ("B.C. unions rally at Highland Valley Copper in support of striking miners", +1),
    ("Copper supply constraints: why new mines are scarce", +1),
    ("Copper output falls in September at Chile's giant Escondida mine", +1),
    ("Glencore in final talks over Mopani sale", 0),
]


class TestImpact(unittest.TestCase):
    def test_direction(self):
        for title, sign in CASES:
            score, reason = impact(title, 0.0)
            got = 0 if abs(score) < 0.15 else (1 if score > 0 else -1)
            self.assertEqual(got, sign, f"{title!r} -> {score:+.2f} ({reason})")

    def test_finbert_fallback_only_for_price_headlines(self):
        self.assertAlmostEqual(impact("Copper prices in focus this week", 0.8)[0], 0.4)
        self.assertEqual(impact("Codelco names new chairman", 0.8)[0], 0.0)


class TestClean(unittest.TestCase):
    def test_removes_junk_and_duplicates(self):
        h = pd.DataFrame({"title": [
            "Long Term Mine Planning Engineer at Hudbay Minerals",
            "Beryllium Copper Market To 2035: Aerospace and Electronics to Drive Growth",
            "Oxygen Free Copper Market Size, Share, Trends, Analysis, 2034",
            "Gold price falls nearly 3% as CPI tops forecasts",
            "Copper falls as China demand weakens",
            "Copper falls as China demand weakens!",
            "Copper falls as China demand weakens",
        ], "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01",
                                   "2026-01-02", "2026-01-03", "2026-01-20"])})
        kept, removed = clean(h)
        self.assertEqual(kept["title"].tolist(), ["Copper falls as China demand weakens"] * 2)
        self.assertEqual(removed.to_dict(), {"iş ilanı": 1, "pazar raporu": 1, "ürün / alaşım": 1,
                                             "bakırla ilgisiz": 1, "tekrar": 1})


if __name__ == "__main__":
    unittest.main()
