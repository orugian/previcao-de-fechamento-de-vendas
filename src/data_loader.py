"""
Data Loader & Preprocessing Module for Lead Scorer Solution
"""
from pathlib import Path
import pandas as pd
import numpy as np

def load_raw_data(data_dir: str = "data") -> dict[str, pd.DataFrame]:
    """Load the four raw CSV tables from the data directory."""
    path = Path(data_dir)
    if not path.exists() and (Path("..") / data_dir).exists():
        path = Path("..") / data_dir
    return {
        "pipeline": pd.read_csv(path / "sales_pipeline.csv"),
        "accounts": pd.read_csv(path / "accounts.csv"),
        "products": pd.read_csv(path / "products.csv"),
        "teams": pd.read_csv(path / "sales_teams.csv"),
    }

def clean_and_merge_data(
    pipeline: pd.DataFrame,
    accounts: pd.DataFrame,
    products: pd.DataFrame,
    teams: pd.DataFrame,
    snapshot_date: str = "2017-12-31"
) -> pd.DataFrame:
    """
    Cleans inconsistencies and merges the 4 CRM tables into a unified master dataset.
    
    Data Hygiene applied:
    1. Product catalog normalization: 'GTXPro' -> 'GTX Pro'
    2. Sector spelling fix: 'technolgy' -> 'technology'
    3. Proper date parsing for engage_date and close_date
    4. Account imputation for top-of-funnel / early stage open deals
    """
    p = pipeline.copy()
    a = accounts.copy()
    pr = products.copy()
    t = teams.copy()

    # 1. Product catalog normalization
    p["product"] = p["product"].replace({"GTXPro": "GTX Pro"})

    # 2. Sector spelling fix
    a["sector"] = a["sector"].replace({"technolgy": "technology"})

    # 3. Date conversion
    p["engage_date"] = pd.to_datetime(p["engage_date"])
    p["close_date"] = pd.to_datetime(p["close_date"])
    snap = pd.to_datetime(snapshot_date)

    # 4. Merge products
    merged = p.merge(pr, on="product", how="left")

    # 5. Merge sales teams
    merged = merged.merge(t, on="sales_agent", how="left")

    # 6. Merge accounts
    merged = merged.merge(a, on="account", how="left")

    # 7. Base indicators for data presence
    merged["is_account_assigned"] = merged["account"].notnull().astype(int)
    merged["has_parent_company"] = merged["subsidiary_of"].notnull().astype(int)

    # Fill missing categorical account values
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["office_location"] = merged["office_location"].fillna("Unknown")
    merged["subsidiary_of"] = merged["subsidiary_of"].fillna("Independent")

    # Numerical imputations for missing accounts (using median of known accounts)
    median_revenue = a["revenue"].median()
    median_employees = a["employees"].median()
    median_year = a["year_established"].median()

    merged["revenue"] = merged["revenue"].fillna(median_revenue)
    merged["employees"] = merged["employees"].fillna(median_employees)
    merged["year_established"] = merged["year_established"].fillna(median_year)

    # Calculate deal age (duration for closed; age to snapshot for active)
    # If engage_date is missing (e.g. Prospecting), use 0 age with is_prospecting flag
    is_closed = merged["deal_stage"].isin(["Won", "Lost"])
    
    deal_age = np.zeros(len(merged))
    # Closed deals: close_date - engage_date
    closed_mask = is_closed & merged["engage_date"].notnull() & merged["close_date"].notnull()
    deal_age[closed_mask] = (merged.loc[closed_mask, "close_date"] - merged.loc[closed_mask, "engage_date"]).dt.days

    # Active deals: snapshot_date - engage_date
    active_mask = (~is_closed) & merged["engage_date"].notnull()
    deal_age[active_mask] = (snap - merged.loc[active_mask, "engage_date"]).dt.days

    merged["deal_age_days"] = np.maximum(0, deal_age)
    merged["is_prospecting"] = (merged["deal_stage"] == "Prospecting").astype(int)
    merged["is_zombie_deal"] = (merged["deal_age_days"] > 120).astype(int)

    # Target definition for closed deals (Won = 1, Lost = 0)
    merged["target"] = np.nan
    merged.loc[merged["deal_stage"] == "Won", "target"] = 1
    merged.loc[merged["deal_stage"] == "Lost", "target"] = 0

    return merged
