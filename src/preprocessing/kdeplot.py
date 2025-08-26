 import sqlite3 as sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotKDE(
    tablename="mergedTable",
    dbPath="../data-v2-working/Data/dataset_working.db",
    target_col=None,                 # e.g., "OUTCOME" or "LABEL"
    classes=None,                    # optionally pass a list/tuple of class labels to plot/order
    features=None,                   # optionally limit to a subset of numeric feature names
    min_per_class=5,                 # skip class curves with fewer than this many points
    jitter_scale=1e-9,               # tiny jitter to avoid singular covariance
    bw_method=None,                  # pass through to pandas .plot.kde (scipy bw_method)
    output_prefix="kde"              # filename prefix
):
    # --- load ---
    with sql.connect(dbPath) as connection:
        df = pd.read_sql_query(f"SELECT * FROM {tablename}", connection)

    # --- keep numeric features (we handle class/target separately) ---
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    # optional feature filter
    if features is not None:
        features = [c for c in features if c in num.columns]
        num = num[features]

    # --- filter "good" columns globally: need variation overall ---
    good_cols = []
    for c in num.columns:
        s = pd.to_numeric(num[c], errors="coerce").dropna()
        if len(s) >= 2 and s.nunique(dropna=True) >= 2 and s.std(ddof=1) > 0:
            good_cols.append(c)

    if not good_cols:
        print("No numeric columns with enough variation for KDE.")
        return

    # slight jitter to be extra safe
    num_jittered = num[good_cols].apply(lambda s: s + np.random.normal(0, jitter_scale, size=len(s)))

    # --- branch 1: class-conditioned KDEs (per feature, one curve per class) ---
    if target_col is not None and target_col in df.columns:
        # establish class order
        if classes is None:
            classes = list(pd.Series(df[target_col]).dropna().unique())

        for col in good_cols:
            ax = None
            plotted_any = False

            for cls in classes:
                mask = df[target_col] == cls
                s = pd.to_numeric(num_jittered.loc[mask, col], errors="coerce").dropna()

                # per-class sanity: must have enough points and variation
                if len(s) < min_per_class or s.nunique(dropna=True) < 2 or s.std(ddof=1) == 0:
                    continue

                # plot this class curve
                if ax is None:
                    ax = s.plot.kde(label=f"{target_col}={cls}", bw_method=bw_method)
                else:
                    s.plot.kde(ax=ax, label=f"{target_col}={cls}", bw_method=bw_method)
                plotted_any = True

            if not plotted_any:
                print(f"Skipped {col}: insufficient per-class data/variation.")
                continue

            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.set_title(f"KDE of {col} by {target_col}")
            ax.legend(title=target_col)

            fig = ax.get_figure()
            safe_col = str(col).replace("/", "_")
            safe_target = str(target_col).replace("/", "_")
            fig.savefig(f"{output_prefix}_{safe_col}_by_{safe_target}.png", bbox_inches="tight")
            fig.clf()
        return

    # --- branch 2: original behavior (no target) — plot numeric cols in groups of 3 ---
    step = 3
    for i in range(0, len(good_cols), step):
        cols = good_cols[i:i+step]
        ax = num_jittered[cols].plot.kde(bw_method=bw_method)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(f"KDE of columns {cols}")
        fig = ax.get_figure()
        fig.savefig(f"{output_prefix}_cols_{i}.png", bbox_inches="tight")
        fig.clf()



plotKDE(
    tablename="mergedTable",
    dbPath="../data-v2-working/Data/dataset_working.db",
    target_col="VITALSTATUS",          # your class column (0/1 or strings)
    features= [
        "PATIENTID",
        "ETHNICITY",
        "VITALSTATUS",
        "LINKNUMBER",
        "AVE_HEIGHT",
        "AVE_WEIGHT",
        "INTENT_OF_TREATMENT",
        "FIRST_DECISION_TO_TREAT",
        "LAST_DECISION_TO_TREAT",
        "FIRST_START_DATE",
        "LAST_START_DATE",
        "NUM_REGIMENS",
        "NUM_TUMOURS",
        "MAX_T_BEST",
        "MAX_N_BEST",
        "MAX_M_BEST",
        "MAX_STAGE_BEST",
        "GRADE",
        "AGE",
        "GLEASON_PRIMARY",
        "GLEASON_SECONDARY"
    ],       # optional; else it will run for all numeric features
    min_per_class=10,              # require at least 10 per class to plot
    output_prefix="kde_by_class"
)