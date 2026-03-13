import pandas as pd


def fmt_count_pct(n: int, den: int) -> str:
    if den == 0:
        return "NA"
    return f"{int(n)}/{int(den)} ({100.0 * n / den:.1f}%)"


def summarize_categorical_by_group(
    df: pd.DataFrame,
    variable: str,
    group_col: str,
    include_missing: bool = False,
    value_labels: dict = None,
    category_order: list = None,
) -> pd.DataFrame:
    tmp = df[[group_col, variable]].copy()
    tmp = tmp.dropna(subset=[group_col])

    if include_missing:
        tmp[variable] = tmp[variable].astype(object).where(tmp[variable].notna(), "Missing")
    else:
        tmp = tmp.dropna(subset=[variable])

    groups = list(tmp[group_col].dropna().unique())
    levels = list(pd.Index(tmp[variable].dropna().unique()))

    if category_order is not None:
        levels = [x for x in category_order if x in levels] + [x for x in levels if x not in category_order]

    count_table = pd.crosstab(tmp[variable], tmp[group_col], dropna=False).reindex(index=levels, fill_value=0)

    denoms = tmp.groupby(group_col)[variable].size()

    out = pd.DataFrame(index=count_table.index)
    for g in groups:
        out[g] = [fmt_count_pct(int(count_table.loc[level, g]), int(denoms.get(g, 0))) for level in count_table.index]

    if value_labels:
        out.index = [value_labels.get(x, x) for x in out.index]

    return out