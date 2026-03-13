from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


@dataclass
class CategoricalResult:
    variable: str
    test: str
    statistic: Optional[float]
    p_value: Optional[float]
    note: str = ""
    contingency_table: Optional[pd.DataFrame] = None
    expected: Optional[np.ndarray] = None
    details: Optional[Dict[str, Any]] = None


def fmt_count_pct(n: int, den: int) -> str:
    """Format count and percentage as 'n/den (pct%)'."""
    if den == 0:
        return "NA"
    pct = 100.0 * n / den
    return f"{int(n)}/{int(den)} ({pct:.1f}%)"


def compare_categorical_groups(
    df: pd.DataFrame,
    variable: str,
    group_col: str,
) -> CategoricalResult:
    """
    Compare a categorical variable across groups.

    Rules:
    - If fewer than 2 groups remain after filtering -> no test
    - If no variability in the variable -> no test
    - For 2x2 tables:
        * use Fisher's exact test if any expected count < 5
        * otherwise use Chi-square test
    - For larger tables:
        * use Chi-square test
        * add note if any expected count < 5
    """
    # Basic column checks
    missing_cols = [col for col in [group_col, variable] if col not in df.columns]
    if missing_cols:
        raise KeyError("Missing required column(s): {}".format(", ".join(missing_cols)))

    # Keep only rows where both grouping and tested variable are present
    tmp = df[[group_col, variable]].dropna().copy()

    if tmp.empty:
        return CategoricalResult(
            variable=variable,
            test="NA",
            statistic=np.nan,
            p_value=np.nan,
            note="No non-missing data available",
        )

    table = pd.crosstab(tmp[variable], tmp[group_col])

    # Need at least 2 groups
    if table.shape[1] < 2:
        return CategoricalResult(
            variable=variable,
            test="NA",
            statistic=np.nan,
            p_value=np.nan,
            note="Only one group present after filtering",
            contingency_table=table,
        )

    # Need at least 2 levels in tested variable
    if table.shape[0] < 2:
        return CategoricalResult(
            variable=variable,
            test="NA",
            statistic=np.nan,
            p_value=np.nan,
            note="No variability in variable",
            contingency_table=table,
        )

    # Compute chi-square first so expected counts can be inspected
    chi2, p_chi2, dof, expected = chi2_contingency(table, correction=False)

    # Fisher only for 2x2 tables with sparse expected counts
    if table.shape == (2, 2) and (expected < 5).any():
        odds_ratio, p_fisher = fisher_exact(table.values)
        return CategoricalResult(
            variable=variable,
            test="Fisher exact",
            statistic=float(odds_ratio),
            p_value=float(p_fisher),
            note="Used Fisher exact because at least one expected count was < 5 in a 2x2 table",
            contingency_table=table,
            expected=expected,
            details={
                "table_shape": table.shape,
                "dof": int(dof),
            },
        )

    note = ""
    if (expected < 5).any():
        note = "Chi-square used, but some expected counts are < 5"

    return CategoricalResult(
        variable=variable,
        test="Chi-square",
        statistic=float(chi2),
        p_value=float(p_chi2),
        note=note,
        contingency_table=table,
        expected=expected,
        details={
            "table_shape": table.shape,
            "dof": int(dof),
        },
    )