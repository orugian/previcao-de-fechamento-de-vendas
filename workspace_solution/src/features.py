"""
Feature Engineering & Point-in-Time Snapshot Reconstruction Module
"""
import pandas as pd
import numpy as np


def create_as_of_dataset(
    df: pd.DataFrame, snapshot_date: str, horizon_days: int = 90
) -> pd.DataFrame:
    """
    Reconstructs the exact state of the active pipeline as of `snapshot_date` (T_cut).
    Eliminates survivorship bias and temporal leakage by freezing all open deals at T_cut
    and evaluating whether they convert to 'Won' within the forward window (T_cut, T_cut + horizon_days].
    """
    t_snap = pd.to_datetime(snapshot_date)
    t_horizon = t_snap + pd.Timedelta(days=horizon_days)

    # Filter deals that entered 'Engaging' on or before t_snap and were NOT yet closed at t_snap
    active_mask = (
        (df["engage_date"].notnull())
        & (df["engage_date"] <= t_snap)
        & ((df["close_date"].isnull()) | (df["close_date"] > t_snap))
    )
    snap = df[active_mask].copy()
    snap["snapshot_date"] = t_snap
    snap["deal_age_days"] = (t_snap - snap["engage_date"]).dt.days
    snap["is_zombie_deal"] = (snap["deal_age_days"] > 90).astype(int)
    snap["in_sweet_spot"] = (
        (snap["deal_age_days"] >= 7) & (snap["deal_age_days"] <= 45)
    ).astype(int)
    snap["engage_month"] = snap["engage_date"].dt.month

    # Target: Did the deal close as 'Won' within the forward horizon?
    won_in_window = (
        (snap["deal_stage"] == "Won")
        & (snap["close_date"].notnull())
        & (snap["close_date"] <= t_horizon)
    )
    snap["target"] = won_in_window.astype(int)
    snap["realized_value"] = np.where(won_in_window, snap["close_value"], 0.0)
    return snap


class FeatureEngineer:
    """
    Fits historical priors and Empirical Bayes (Shrinkage) target encodings strictly
    on historical deals closed BEFORE the cutoff date, preventing any future data leakage.
    """

    def __init__(self, bayes_weight: float = 10.0, interaction_weight: float = 5.0):
        self.bayes_weight = bayes_weight
        self.interaction_weight = interaction_weight
        self.global_win_rate = 0.5
        self.agent_priors = {}
        self.product_priors = {}
        self.sector_priors = {}
        self.region_priors = {}
        self.agent_product_priors = {}
        self.fitted = False

    def fit(self, historical_closed_df: pd.DataFrame):
        """
        Fit Empirical Bayes priors strictly on deals that closed prior to the temporal split.
        """
        closed = historical_closed_df[historical_closed_df["target"].notnull()].copy()
        self.global_win_rate = float(closed["target"].mean())

        def _calc_bayes(group_col, weight):
            stats = closed.groupby(group_col)["target"].agg(["count", "sum"])
            priors = (stats["sum"] + weight * self.global_win_rate) / (
                stats["count"] + weight
            )
            return priors.to_dict()

        self.agent_priors = _calc_bayes("sales_agent", self.bayes_weight)
        self.product_priors = _calc_bayes("product", self.bayes_weight)
        self.sector_priors = _calc_bayes("sector", self.bayes_weight)
        self.region_priors = _calc_bayes("regional_office", self.bayes_weight)

        ap_stats = closed.groupby(["sales_agent", "product"])["target"].agg(
            ["count", "sum"]
        )
        self.agent_product_priors = (
            (ap_stats["sum"] + self.interaction_weight * self.global_win_rate)
            / (ap_stats["count"] + self.interaction_weight)
        ).to_dict()

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform any snapshot or active dataset using the pre-fitted priors."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted on historical training data first!")

        d = df.copy()
        d["agent_win_rate"] = (
            d["sales_agent"].map(self.agent_priors).fillna(self.global_win_rate)
        )
        d["product_win_rate"] = (
            d["product"].map(self.product_priors).fillna(self.global_win_rate)
        )
        d["sector_win_rate"] = (
            d["sector"].map(self.sector_priors).fillna(self.global_win_rate)
        )
        d["region_win_rate"] = (
            d["regional_office"].map(self.region_priors).fillna(self.global_win_rate)
        )
        d["agent_product_win_rate"] = [
            self.agent_product_priors.get((a, p), self.global_win_rate)
            for a, p in zip(d["sales_agent"], d["product"])
        ]

        # Ensure temporal age features exist even for raw active inference rows
        if "in_sweet_spot" not in d.columns:
            d["in_sweet_spot"] = (
                (d["deal_age_days"] >= 7) & (d["deal_age_days"] <= 45)
            ).astype(int)
        d["is_zombie_deal"] = (d["deal_age_days"] > 90).astype(int)

        # Log & ratio transformations
        d["log_sales_price"] = np.log1p(d["sales_price"])
        d["log_revenue"] = np.log1p(d["revenue"])
        d["log_employees"] = np.log1p(d["employees"])
        d["revenue_per_employee"] = (d["revenue"] * 1_000_000.0) / np.maximum(
            1.0, d["employees"]
        )
        d["account_age"] = 2017 - d["year_established"]
        d["age_to_price_ratio"] = d["deal_age_days"] / np.maximum(
            1.0, d["sales_price"] / 100.0
        )

        # Survival decay proxy based on empirical hazard curve
        d["velocity_decay"] = np.exp(-np.maximum(0.0, d["deal_age_days"] - 15.0) / 45.0)

        return d

    def get_feature_columns(self) -> list[str]:
        """Return canonical list of feature columns for ML training and inference."""
        return [
            "deal_age_days",
            "velocity_decay",
            "is_zombie_deal",
            "in_sweet_spot",
            "is_account_assigned",
            "has_parent_company",
            "sales_price",
            "log_sales_price",
            "revenue",
            "log_revenue",
            "employees",
            "log_employees",
            "revenue_per_employee",
            "account_age",
            "age_to_price_ratio",
            "agent_win_rate",
            "product_win_rate",
            "sector_win_rate",
            "region_win_rate",
            "agent_product_win_rate",
        ]
