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

HORIZONS = tuple(range(1, 13))
FIRST_ORIGIN = "2015-01"      # first simulated forecast origin of the walk-forward backtest
MIN_TRAIN = 48                # minimum matured training rows per horizon
N_FACTORS = 5                 # PCA factors extracted from the broad macro/market panel
FACTOR_MIN_COVERAGE = 0.9     # a broad column enters PCA only with >=90% history at the origin
COMBO_LOOKBACK = 36           # matured forecasts used to weight the experts (at least 12*h)
COMBO_MIN_EFFECTIVE = 12      # independent (non-overlapping) matured forecasts required: n >= 12*h,
                              # below this the combination is 100% the anchor rule
CONFORMAL_MIN = 24            # matured combination errors needed for empirical intervals
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
FORWARD_TENOR = 3             # months covered by the LME 3-month price
FFILL_LIMIT = 2               # ragged edge: carry the last published value at most 2 months
N_JOBS = int(os.environ.get("COPPER_N_JOBS", "6"))
RANDOM_STATE = 42

MODELS = ("rw", "rw_spot", "forward", "own_ridge", "factor_ridge", "gbm")
ML_EXPERTS = ("own_ridge", "factor_ridge", "gbm")
SCENARIO = "ml_scenario"     # equal-weight ML average, published separately, never in the combination
BENCHMARK = "rw"            # skill and DM tests are reported against the plain random walk
ANCHOR = "forward"          # the bar an expert must clear to receive combination weight


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
# The original project's published forecast, copied for side-by-side comparison only.
REFERENCE_FILES = ("data/predictions/copper_production_forecast_latest.csv",)
