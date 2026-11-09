"""Central configuration: paths, horizons, and the source registry.

Every exogenous series carries an explicit publication lag (months). At a
forecast origin t (the last completed month) the model only sees the value for
month t - lag, which mimics what was actually knowable when the forecast is made
in the first days of month t+1.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_REF = ROOT / "data" / "reference"
PANEL_DIR = ROOT / "data" / "panel"
OUT = ROOT / "outputs"
ARCHIVE = OUT / "archive"

# Read-only source of the raw files. Nothing is ever written there.
SOURCE_PROJECT = Path(
    os.environ.get("COPPER_SOURCE_PROJECT", Path.home() / "Copper_Prediction_with_Machine_Learning")
)

# Time unit: the dekad (a third of a month: days 1-10, 11-20, 21-end). 36 dekads = 12 months.
HORIZONS = tuple(range(1, 37))
FIRST_ORIGIN = (2015, 1, 1)   # first simulated forecast origin (year, month, dekad)
MIN_TRAIN = 108               # minimum matured training dekads per horizon (3 years)
MIN_TRAIN_NEWS = 90           # news starts in 2017, so its expert needs a shorter minimum
COVERAGE = 0.9                # a macro column is used only with >=90% history in the training rows
TUNE_HORIZONS = (1, 3, 9, 18, 36)   # hyperparameters are tuned here; other horizons use the nearest
LGBM_TUNE_ITER = 10           # random-search candidates per yearly LightGBM re-tuning
FORWARD_TENOR = 9             # dekads covered by the LME 3-month price
FFILL_LIMIT = 2               # ragged edge: carry the last published monthly value at most 2 months
N_JOBS = int(os.environ.get("COPPER_N_JOBS", "6"))
RANDOM_STATE = 42

BENCHMARKS = ("rw", "forward")
ML_EXPERTS = ("enet", "lgbm", "enet_news")
FORECAST = "ml"               # the published forecast: equal-weight average of the ML experts
DRIVER_HORIZONS = (3, 9, 18, 36)    # 1, 3, 6 and 12 months ahead


@dataclass(frozen=True)
class Series:
    name: str
    file: str
    column: str
    kind: str        # "log" -> log changes, "level" -> plain differences
    lag: int         # publication lag in months
    family: str
    daily: bool = False


def _s(name, file, kind, lag, family, column=None, daily=False):
    return Series(name, file, column or name, kind, lag, family, daily)


LME_OTHER = "market/lme_other_metals_daily.csv"
FRED = "macro/fred_macro_monthly.csv"
GLOBAL = "macro/global_macro_monthly.csv"
WB = "market/worldbank_commodities_monthly.csv"

SERIES: tuple[Series, ...] = (
    # --- LME base metals & inventories (daily, known at month end)
    _s("aluminum", LME_OTHER, "log", 0, "metals", "aluminum_cash_usd_per_ton", True),
    _s("nickel", LME_OTHER, "log", 0, "metals", "nickel_cash_usd_per_ton", True),
    _s("zinc", LME_OTHER, "log", 0, "metals", "zinc_cash_usd_per_ton", True),
    _s("lead", LME_OTHER, "log", 0, "metals", "lead_cash_usd_per_ton", True),
    _s("tin", LME_OTHER, "log", 0, "metals", "tin_cash_usd_per_ton", True),
    _s("aluminum_stock", LME_OTHER, "log", 0, "inventories", "aluminum_stock_ton", True),
    _s("nickel_stock", LME_OTHER, "log", 0, "inventories", "nickel_stock_ton", True),
    _s("zinc_stock", LME_OTHER, "log", 0, "inventories", "zinc_stock_ton", True),
    # --- other commodities (World Bank Pink Sheet monthly averages, published ~2nd business day)
    _s("gold", WB, "log", 0, "precious", "gold_usd_per_troy_ounce"),
    _s("silver", WB, "log", 0, "precious", "silver_usd_per_troy_ounce"),
    _s("platinum", WB, "log", 0, "precious", "platinum_usd_per_troy_ounce"),
    _s("iron_ore", WB, "log", 0, "bulk", "iron_ore_usd_per_dmtu"),
    _s("coal", WB, "log", 0, "energy", "coal_australia_usd_per_ton"),
    _s("brent", FRED, "log", 0, "energy", "brent_oil_usd"),
    _s("natural_gas", FRED, "log", 0, "energy", "natural_gas_usd"),
    # --- financial markets (monthly averages of daily market data)
    _s("dollar_index", FRED, "log", 0, "fx"),
    _s("eur_usd", FRED, "log", 0, "fx"),
    _s("usd_cny", FRED, "log", 0, "fx"),
    _s("usd_jpy", FRED, "log", 0, "fx"),
    _s("usd_krw", FRED, "log", 0, "fx"),
    _s("usd_inr", FRED, "log", 0, "fx"),
    _s("peru_usd_pen", "mining/peru_copper_cost_drivers_monthly.csv", "log", 0, "fx"),
    _s("vix", FRED, "log", 0, "risk", "vix_index"),
    _s("fed_rate", FRED, "level", 0, "rates", "fed_interest_rate"),
    _s("us_10y", FRED, "level", 0, "rates", "us_10y_treasury_yield"),
    _s("us_2y", FRED, "level", 0, "rates", "us_2y_yield"),
    _s("us_curve", FRED, "level", 0, "rates", "us_10y_2y_spread"),
    _s("bund_10y", FRED, "level", 0, "rates", "germany_10y_bund_yield"),
    _s("japan_10y", FRED, "level", 0, "rates", "japan_10y_yield"),
    _s("korea_10y", FRED, "level", 0, "rates", "korea_10y_yield"),
    _s("ecb_rate", GLOBAL, "level", 0, "rates", "ecb_policy_rate"),
    # --- risk / uncertainty (news-based indices)
    _s("geopolitical_risk", "risk/geopolitical_risk_monthly.csv", "log", 1, "risk", "geopolitical_risk_index"),
    _s("policy_uncertainty", "risk/economic_policy_uncertainty_monthly.csv", "log", 2, "risk",
       "global_economic_policy_uncertainty_index"),
    # --- real economy (official statistics, conservative 2-month release lag)
    _s("us_cpi", FRED, "log", 2, "us_macro", "us_cpi_index"),
    _s("us_ppi", FRED, "log", 2, "us_macro"),
    _s("us_industrial_production", FRED, "log", 2, "us_macro"),
    _s("us_housing_starts", FRED, "log", 2, "us_macro"),
    _s("us_m2", FRED, "log", 2, "us_macro", "m2_money_supply"),
    _s("euro_area_ppi", GLOBAL, "log", 2, "europe"),
    _s("euro_area_copper_ppi", GLOBAL, "log", 2, "europe", "euro_area_copper_production_ppi"),
    _s("euro_area_industrial_production", GLOBAL, "log", 2, "europe"),
    _s("germany_exports", FRED, "log", 2, "europe"),
    _s("germany_order_books", FRED, "level", 2, "europe", "germany_manufacturing_order_books"),
    _s("japan_industrial_production", GLOBAL, "log", 2, "asia"),
    _s("japan_ppi", GLOBAL, "log", 2, "asia"),
    _s("korea_exports", FRED, "log", 2, "asia"),
    _s("korea_industrial_production", FRED, "level", 2, "asia"),
    _s("india_industrial_production", FRED, "level", 2, "asia"),
    _s("china_ppi_yoy", "macro/china_ppi_monthly.csv", "level", 2, "china", "china_ppi_yoy_pct"),
    _s("china_industrial_yoy", "macro/china_industrial_output_monthly.csv", "level", 2, "china",
       "china_industrial_value_added_yoy_pct"),
    _s("china_fai_yoy", "macro/china_fixed_asset_investment_monthly.csv", "level", 2, "china",
       "china_fixed_asset_investment_yoy_pct"),
    _s("china_real_estate_yoy", "macro/china_real_estate_investment_monthly.csv", "level", 2, "china",
       "china_real_estate_investment_yoy_pct"),
    _s("china_electricity", "macro/china_electricity_generation_monthly.csv", "log", 2, "china",
       "china_electricity_generation_100m_kwh"),
    _s("china_refined_copper", "market/china_refined_copper_monthly.csv", "log", 2, "china",
       "china_refined_copper_production_ton"),
    _s("peru_diesel", "mining/peru_copper_cost_drivers_monthly.csv", "log", 2, "mining_costs",
       "peru_diesel_price_index"),
)

COPPER_FILE = "market/lme_copper_daily.csv"

# Files copied into data/raw (snapshot) from the source project.
RAW_FILES = tuple(sorted({COPPER_FILE, *(s.file for s in SERIES)}))
REFERENCE_FILES: tuple[str, ...] = ()

SERIES_TR = {
    "aluminum": "Alüminyum fiyatı", "nickel": "Nikel fiyatı", "zinc": "Çinko fiyatı", "lead": "Kurşun fiyatı",
    "tin": "Kalay fiyatı", "aluminum_stock": "LME alüminyum stoku", "nickel_stock": "LME nikel stoku",
    "zinc_stock": "LME çinko stoku", "gold": "Altın", "silver": "Gümüş", "platinum": "Platin",
    "iron_ore": "Demir cevheri", "coal": "Kömür", "brent": "Brent petrol", "natural_gas": "Doğal gaz",
    "dollar_index": "Dolar endeksi (DXY)", "eur_usd": "EUR/USD", "usd_cny": "USD/CNY (yuan)",
    "usd_jpy": "USD/JPY", "usd_krw": "USD/KRW", "usd_inr": "USD/INR", "peru_usd_pen": "USD/PEN (Peru)",
    "vix": "VIX korku endeksi", "fed_rate": "Fed faizi", "us_10y": "ABD 10 yıllık faiz",
    "us_2y": "ABD 2 yıllık faiz", "us_curve": "ABD faiz eğrisi (10Y-2Y)", "bund_10y": "Almanya 10 yıllık faiz",
    "japan_10y": "Japonya 10 yıllık faiz", "korea_10y": "Kore 10 yıllık faiz", "ecb_rate": "ECB faizi",
    "geopolitical_risk": "Jeopolitik risk endeksi", "policy_uncertainty": "Ekonomi politikası belirsizliği",
    "us_cpi": "ABD enflasyonu (TÜFE)", "us_ppi": "ABD ÜFE", "us_industrial_production": "ABD sanayi üretimi",
    "us_housing_starts": "ABD konut başlangıçları", "us_m2": "ABD para arzı (M2)",
    "euro_area_ppi": "Euro bölgesi ÜFE", "euro_area_copper_ppi": "Euro bölgesi bakır ÜFE",
    "euro_area_industrial_production": "Euro bölgesi sanayi üretimi", "germany_exports": "Almanya ihracatı",
    "germany_order_books": "Almanya sanayi siparişleri", "japan_industrial_production": "Japonya sanayi üretimi",
    "japan_ppi": "Japonya ÜFE", "korea_exports": "Kore ihracatı", "korea_industrial_production": "Kore sanayi üretimi",
    "india_industrial_production": "Hindistan sanayi üretimi", "china_ppi_yoy": "Çin ÜFE",
    "china_industrial_yoy": "Çin sanayi üretimi", "china_fai_yoy": "Çin sabit sermaye yatırımı",
    "china_real_estate_yoy": "Çin gayrimenkul yatırımı", "china_electricity": "Çin elektrik üretimi",
    "china_refined_copper": "Çin rafine bakır üretimi", "peru_diesel": "Peru dizel fiyatı",
}
FAMILY_TR = {
    "metals": "Metaller", "inventories": "Stoklar", "precious": "Değerli metaller", "bulk": "Hammadde",
    "energy": "Enerji", "fx": "Kurlar", "risk": "Risk", "rates": "Faizler", "us_macro": "ABD ekonomisi",
    "europe": "Avrupa ekonomisi", "asia": "Asya ekonomisi", "china": "Çin ekonomisi",
    "mining_costs": "Madencilik maliyeti", "copper": "Bakırın kendisi", "news": "Haberler",
}
