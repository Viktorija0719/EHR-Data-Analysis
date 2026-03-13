import pandas as pd

from scripts.continuous_tests import compare_continuous_groups
from scripts.categorical_tests import compare_categorical_groups
from scripts.summaries import summarize_categorical_by_group


def build_table1(
    df: pd.DataFrame,
    group_col: str,
    continuous_cols,
    categorical_cols,
    alpha: float = 0.05,
    include_missing_categorical: bool = False,
    value_labels: dict = None,
    category_orders: dict = None,
):
    rows = []
    details = {"posthoc": {}}

    for col in continuous_cols:
        res = compare_continuous_groups(
            df=df,
            variable=col,
            group_col=group_col,
            alpha=alpha,
        )

        row = {
            "Variable": col,
            "Level": "",
            "Type": "continuous",
            "Test": res.test,
            "Statistic": res.statistic,
            "p_value": res.p_value,
            "Note": res.note,
        }

        if res.details and "group_summaries" in res.details:
            row.update(res.details["group_summaries"])

        rows.append(row)

        if res.posthoc is not None:
            details["posthoc"][col] = res.posthoc

    for col in categorical_cols:
        res = compare_categorical_groups(
            df=df,
            variable=col,
            group_col=group_col,
        )

        summary_df = summarize_categorical_by_group(
            df=df,
            variable=col,
            group_col=group_col,
            include_missing=include_missing_categorical,
            value_labels=(value_labels or {}).get(col),
            category_order=(category_orders or {}).get(col),
        )

        first = True
        for level, level_row in summary_df.iterrows():
            row = {
                "Variable": col,
                "Level": level,
                "Type": "categorical" if first else "",
                "Test": res.test if first else "",
                "Statistic": res.statistic if first else "",
                "p_value": res.p_value if first else "",
                "Note": res.note if first else "",
            }
            row.update(level_row.to_dict())
            rows.append(row)
            first = False

    out = pd.DataFrame(rows).set_index(["Variable", "Level"])
    return out, details