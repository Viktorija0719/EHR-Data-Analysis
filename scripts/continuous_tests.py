from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import (
    shapiro,
    levene,
    ttest_1samp,
    ttest_ind,
    mannwhitneyu,
    f_oneway,
    kruskal,
    binomtest,
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


@dataclass
class TestResult:
    variable: str
    test: str
    statistic: Optional[float]
    p_value: Optional[float]
    note: str = ""
    posthoc: Optional[pd.DataFrame] = None
    details: Optional[Dict[str, Any]] = None


def safe_numeric(x) -> np.ndarray:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    return x


def safe_shapiro(x: np.ndarray, random_state: int = 42) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 3:
        return np.nan
    if np.unique(x).size < 3:
        return np.nan

    if n > 5000:
        rng = np.random.default_rng(random_state)
        x = rng.choice(x, size=5000, replace=False)

    return float(shapiro(x).pvalue)


def sign_test_against_value(x: np.ndarray, reference: float) -> Tuple[float, float]:
    diff = x - reference
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return np.nan, np.nan

    n_positive = np.sum(diff > 0)
    res = binomtest(k=n_positive, n=n, p=0.5, alternative="two-sided")
    return float(n_positive), float(res.pvalue)


def fmt_mean_sd(x: np.ndarray) -> str:
    return f"{np.mean(x):.2f} ± {np.std(x, ddof=1):.2f}"


def fmt_median_iqr(x: np.ndarray) -> str:
    q1, q3 = np.percentile(x, [25, 75])
    return f"{np.median(x):.2f} [{q1:.2f}, {q3:.2f}]"


def compare_continuous_groups(
    df: pd.DataFrame,
    variable: str,
    group_col: Optional[str] = None,
    alpha: float = 0.05,
    reference_value: Optional[float] = None,
    dunn_p_adjust: str = "holm",
) -> TestResult:
    if group_col is None:
        x = safe_numeric(df[variable])
        if len(x) < 2:
            return TestResult(variable, "Insufficient data", np.nan, np.nan, "Need at least 2 values")

        p_norm = safe_shapiro(x)
        if np.isnan(p_norm):
            return TestResult(variable, "Insufficient normality info", np.nan, np.nan)

        if p_norm > alpha:
            stat, p = ttest_1samp(x, popmean=reference_value)
            return TestResult(
                variable,
                "One-sample t-test",
                float(stat),
                float(p),
                details={"shapiro_p": p_norm, "summary": fmt_mean_sd(x)},
            )
        else:
            stat, p = sign_test_against_value(x, reference_value)
            return TestResult(
                variable,
                "Sign test",
                stat,
                p,
                note="Nonparametric test against reference location/value",
                details={"shapiro_p": p_norm, "summary": fmt_median_iqr(x)},
            )

    groups = [g for g in df[group_col].dropna().unique()]
    grouped = {g: safe_numeric(df.loc[df[group_col] == g, variable]) for g in groups}
    grouped = {g: x for g, x in grouped.items() if len(x) > 0}

    if len(grouped) < 2:
        return TestResult(variable, "Insufficient groups", np.nan, np.nan)

    shapiro_ps = {g: safe_shapiro(x) for g, x in grouped.items()}
    all_normal = all((not np.isnan(p)) and (p > alpha) for p in shapiro_ps.values())

    if len(grouped) == 2:
        g1, g2 = list(grouped.keys())
        x1, x2 = grouped[g1], grouped[g2]

        if len(x1) < 2 or len(x2) < 2:
            return TestResult(variable, "Insufficient data", np.nan, np.nan)

        if all_normal:
            p_lev = levene(x1, x2, center="median").pvalue
            equal_var = p_lev > alpha
            stat, p = ttest_ind(x1, x2, equal_var=equal_var)
            return TestResult(
                variable,
                "Student t-test" if equal_var else "Welch t-test",
                float(stat),
                float(p),
                details={
                    "group_summaries": {g1: fmt_mean_sd(x1), g2: fmt_mean_sd(x2)},
                    "shapiro_p": shapiro_ps,
                    "levene_p": float(p_lev),
                },
            )
        else:
            stat, p = mannwhitneyu(x1, x2, alternative="two-sided")
            return TestResult(
                variable,
                "Mann-Whitney U",
                float(stat),
                float(p),
                details={
                    "group_summaries": {g1: fmt_median_iqr(x1), g2: fmt_median_iqr(x2)},
                    "shapiro_p": shapiro_ps,
                },
            )

    if all_normal:
        arrays = list(grouped.values())
        stat, p = f_oneway(*arrays)

        posthoc = None
        if p < alpha:
            stacked = (
                df[[group_col, variable]]
                .assign(**{variable: pd.to_numeric(df[variable], errors="coerce")})
                .dropna()
            )
            posthoc_res = pairwise_tukeyhsd(
                endog=stacked[variable],
                groups=stacked[group_col],
                alpha=alpha,
            )
            posthoc = pd.DataFrame(
                posthoc_res._results_table.data[1:],
                columns=posthoc_res._results_table.data[0],
            )

        return TestResult(
            variable,
            "One-way ANOVA",
            float(stat),
            float(p),
            posthoc=posthoc,
            details={
                "group_summaries": {g: fmt_mean_sd(x) for g, x in grouped.items()},
                "shapiro_p": shapiro_ps,
            },
        )

    arrays = list(grouped.values())
    stat, p = kruskal(*arrays)

    posthoc = None
    if p < alpha:
        long_df = (
            df[[group_col, variable]]
            .assign(**{variable: pd.to_numeric(df[variable], errors="coerce")})
            .dropna()
        )
        posthoc = sp.posthoc_dunn(
            long_df,
            val_col=variable,
            group_col=group_col,
            p_adjust=dunn_p_adjust,
        )

    return TestResult(
        variable,
        "Kruskal-Wallis",
        float(stat),
        float(p),
        posthoc=posthoc,
        details={
            "group_summaries": {g: fmt_median_iqr(x) for g, x in grouped.items()},
            "shapiro_p": shapiro_ps,
        },
    )