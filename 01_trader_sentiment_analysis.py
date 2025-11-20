import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ========= 1. LOAD DATA =========

def load_data():
    """Load your CSVs using the real absolute paths you provided."""
    trades_path = r"/Users/chandini/Creative Cloud Files/WorkSpace /bajarangs-trader-sentiment/data/historical_data - historical_data.csv"
    sentiment_path = r"/Users/chandini/Creative Cloud Files/WorkSpace /bajarangs-trader-sentiment/data/fear_greed_index - fear_greed_index.csv"

    print("Reading Trades (Historical Data) from:")
    print("  ", trades_path)
    print("Reading Sentiment (Fear-Greed Index) from:")
    print("  ", sentiment_path)

    trades = pd.read_csv(trades_path)
    sentiment = pd.read_csv(sentiment_path)

    print("\n[DEBUG] Trades columns:", list(trades.columns))
    print("[DEBUG] Sentiment columns:", list(sentiment.columns))

    return trades, sentiment


# ========= 2. PREPROCESS TRADES =========

def preprocess_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Detect a time/date column in trades and create trade_date."""

    trades = trades.copy()

    print("\n=========== STEP 2: CLEAN TRADES ===========")
    print("[DEBUG] Trades columns:", list(trades.columns))

    # Auto-detect any column that looks like time/date
    time_col = None
    for col in trades.columns:
        col_lower = col.lower()
        if ("time" in col_lower) or ("date" in col_lower) or ("timestamp" in col_lower):
            time_col = col
            break

    if time_col is None:
        raise Exception(
            "❌ No timestamp/date-like column found in trades CSV.\n"
            "Look at the [DEBUG] Trades columns above and hard-code the correct one in preprocess_trades()."
        )

    print(f"[INFO] Using '{time_col}' as the time column for trades.")

    trades[time_col] = pd.to_datetime(trades[time_col], errors="coerce")
    trades = trades.dropna(subset=[time_col])

    trades["trade_date"] = trades[time_col].dt.date

    print("\n[DEBUG] Trades after datetime conversion (first 5 rows):")
    print(trades[[time_col, "trade_date"]].head())

    return trades


# ========= 3. PREPROCESS SENTIMENT =========

def detect_sentiment_date_column(sentiment: pd.DataFrame) -> str:
    """Force 'date' as the correct date column (avoid UNIX timestamp)."""
    if "date" in sentiment.columns:
        return "date"
    if "Date" in sentiment.columns:
        return "Date"
    raise Exception(
        "❌ No 'date' column found in sentiment CSV. Available columns: "
        f"{list(sentiment.columns)}"
    )


def detect_sentiment_label_column(sentiment: pd.DataFrame):
    """
    Try to find a text column that already contains Fear/Greed labels.
    """
    for col in sentiment.columns:
        col_lower = col.lower()
        if any(key in col_lower for key in ["class", "fear", "greed", "sentiment", "label", "regime"]):
            return col
    return None


def detect_sentiment_numeric_column(sentiment: pd.DataFrame):
    """
    Try to find a numeric column that looks like an index/score (0–100).
    Often this is 'value', 'index', 'score', etc.
    """
    candidates = []
    for col in sentiment.columns:
        if pd.api.types.is_numeric_dtype(sentiment[col]):
            candidates.append(col)

    if not candidates:
        return None

    # First try common names
    for name in ["value", "index", "score", "fg_value", "fear_greed"]:
        for col in candidates:
            if name in col.lower():
                return col

    # fallback: just return the first numeric column
    return candidates[0]


def bucket_fear_greed_from_numeric(series: pd.Series) -> pd.Series:
    """
    Create buckets from a numeric 0–100 fear/greed index:
      0-25: Extreme Fear
      25-50: Fear
      50-75: Greed
      75-100: Extreme Greed
    """
    return pd.cut(
        series,
        bins=[-1, 25, 50, 75, 101],
        labels=["Extreme Fear", "Fear", "Greed", "Extreme Greed"]
    )


def preprocess_sentiment(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Normalize sentiment to have columns: sentiment_date, Classification."""

    sentiment = sentiment.copy()

    print("\n=========== STEP 3: CLEAN SENTIMENT ===========")
    print("[DEBUG] Sentiment columns:", list(sentiment.columns))
    print("[DEBUG] Sentiment head:")
    print(sentiment.head())

    # ---- Find and parse date column ----
    date_col = detect_sentiment_date_column(sentiment)
    print(f"[INFO] Using '{date_col}' as the sentiment date column.")

    sentiment[date_col] = pd.to_datetime(sentiment[date_col], errors="coerce")
    sentiment = sentiment.dropna(subset=[date_col])
    sentiment["sentiment_date"] = sentiment[date_col].dt.date

    # ---- Find or build classification column ----
    label_col = detect_sentiment_label_column(sentiment)

    if label_col is not None:
        print(f"[INFO] Using '{label_col}' as the sentiment classification column.")
        sentiment["Classification"] = sentiment[label_col].astype(str)
    else:
        print("[WARN] No explicit sentiment label column found. Trying to derive from numeric index...")
        num_col = detect_sentiment_numeric_column(sentiment)

        if num_col is None:
            raise Exception(
                "❌ Could not find a sentiment label column or numeric index to derive Fear/Greed.\n"
                f"Columns are: {list(sentiment.columns)}"
            )

        print(f"[INFO] Using numeric column '{num_col}' to derive Fear/Greed buckets.")
        sentiment["Classification"] = bucket_fear_greed_from_numeric(sentiment[num_col]).astype(str)

    # ---- Keep only needed columns ----
    sentiment_small = (
        sentiment[["sentiment_date", "Classification"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("\n[DEBUG] Cleaned sentiment sample:")
    print(sentiment_small.head())

    return sentiment_small


# ========= 4. MERGE =========

def merge_trades_sentiment(trades: pd.DataFrame, sentiment_small: pd.DataFrame) -> pd.DataFrame:
    """Map each trade to the Fear/Greed classification of that day."""
    print("\n=========== STEP 4: MERGE TRADES + SENTIMENT ===========")

    merged = trades.merge(
        sentiment_small,
        left_on="trade_date",
        right_on="sentiment_date",
        how="left"
    )

    print("\n[DEBUG] Sentiment mapping counts (including NaN):")
    print(merged["Classification"].value_counts(dropna=False))

    merged = merged.dropna(subset=["Classification"])

    print(f"\n[INFO] Trades with valid sentiment attached: {len(merged):,}")
    print("[DEBUG] Merged sample:")
    print(merged[["trade_date", "Classification"]].head())

    return merged


# ========= 5. DETECT KEY COLUMNS FOR PERFORMANCE ANALYSIS =========

def detect_account_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ["account", "trader", "user", "wallet", "id"]):
            return col
    raise Exception(
        "❌ Could not detect account column. Please check your trades columns and "
        "hard-code the correct column in detect_account_column()."
    )


def detect_pnl_column(df: pd.DataFrame) -> str:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        if "pnl" in col.lower():
            return col
    raise Exception(
        "❌ Could not detect PnL column. Look for something like 'Closed PnL' or 'pnl' "
        "and hard-code it in detect_pnl_column()."
    )


def detect_leverage_column(df: pd.DataFrame):
    for col in df.columns:
        if "lev" in col.lower() or "leverage" in col.lower():
            return col
    return None   # optional


def detect_side_column(df: pd.DataFrame):
    for col in df.columns:
        if any(k in col.lower() for k in ["side", "direction", "position_side"]):
            return col
    return None   # optional


def detect_size_column(df: pd.DataFrame):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        if any(k in col.lower() for k in ["size", "qty", "quantity", "volume"]):
            return col
    return None   # optional


# ========= 6. BUILD DAILY PERFORMANCE PER TRADER =========

def build_daily_performance(trades_sent: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per account + day + sentiment into performance metrics."""

    print("\n=========== STEP 5: BUILD DAILY PERFORMANCE ===========")
    print("[DEBUG] Columns available in merged trades:")
    print(list(trades_sent.columns))

    account_col = detect_account_column(trades_sent)
    pnl_col = detect_pnl_column(trades_sent)
    lev_col = detect_leverage_column(trades_sent)
    side_col = detect_side_column(trades_sent)
    size_col = detect_size_column(trades_sent)

    print(f"[INFO] Using '{account_col}' as account column.")
    print(f"[INFO] Using '{pnl_col}' as PnL column.")
    if lev_col:
        print(f"[INFO] Using '{lev_col}' as leverage column.")
    if side_col:
        print(f"[INFO] Using '{side_col}' as side column.")
    if size_col:
        print(f"[INFO] Using '{size_col}' as size/volume column.")

    group_cols = [account_col, "trade_date", "Classification"]

    agg_dict = {
        "total_pnl": (pnl_col, "sum"),
        "avg_pnl": (pnl_col, "mean"),
        "max_drawdown": (pnl_col, "min"),
    }

    # number of trades
    if size_col is not None:
        agg_dict["n_trades"] = (size_col, "size")
    else:
        agg_dict["n_trades"] = (account_col, "size")

    # leverage
    if lev_col is not None:
        agg_dict["avg_leverage"] = (lev_col, "mean")

    # long / short counts
    if side_col is not None:
        agg_dict["long_trades"] = (
            side_col,
            lambda x: x.astype(str).str.lower().str.contains("buy|long").sum(),
        )
        agg_dict["short_trades"] = (
            side_col,
            lambda x: x.astype(str).str.lower().str.contains("sell|short").sum(),
        )

    daily_perf = trades_sent.groupby(group_cols).agg(**agg_dict).reset_index()

    # Win rate = % of trades with positive pnl
    tmp = trades_sent.copy()
    tmp["win"] = tmp[pnl_col] > 0
    winrate_df = (
        tmp.groupby(group_cols)["win"]
        .mean()
        .reset_index()
        .rename(columns={"win": "win_rate"})
    )

    daily_perf = daily_perf.merge(winrate_df, on=group_cols, how="left")

    print("\n[DEBUG] Daily performance sample:")
    print(daily_perf.head())

    # Store which column is account for later use
    daily_perf.attrs["account_col"] = account_col

    return daily_perf


# ========= 7. SUMMARY BY SENTIMENT =========

def summarize_by_sentiment(daily_perf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily performance into sentiment-level stats."""
    print("\n=========== STEP 6: SUMMARY BY SENTIMENT ===========")

    account_col = daily_perf.attrs.get("account_col", "account")

    agg_dict = {
        "mean_pnl": ("total_pnl", "mean"),
        "median_pnl": ("total_pnl", "median"),
        "mean_win_rate": ("win_rate", "mean"),
        "n_account_days": (account_col, "size"),
    }

    if "avg_leverage" in daily_perf.columns:
        agg_dict["mean_leverage"] = ("avg_leverage", "mean")

    sent_summary = daily_perf.groupby("Classification").agg(**agg_dict).reset_index()

    print("\n[DEBUG] Sentiment summary:")
    print(sent_summary)

    return sent_summary


# ========= 8. BASIC PLOTS =========

def plot_sentiment_summary(sent_summary: pd.DataFrame, base_output_dir: Path):
    """Save bar charts comparing sentiment regimes."""
    print("\n=========== STEP 7: GENERATE PLOTS ===========")
    sns.set(style="whitegrid")

    output_dir = base_output_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mean PnL
    plt.figure(figsize=(7, 4))
    sns.barplot(data=sent_summary, x="Classification", y="mean_pnl")
    plt.title("Average Daily PnL per Account by Sentiment")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_pnl_by_sentiment.png")
    plt.close()

    # Win rate
    plt.figure(figsize=(7, 4))
    sns.barplot(data=sent_summary, x="Classification", y="mean_win_rate")
    plt.title("Average Win Rate per Account by Sentiment")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "win_rate_by_sentiment.png")
    plt.close()

    # Leverage (if present)
    if "mean_leverage" in sent_summary.columns:
        plt.figure(figsize=(7, 4))
        sns.barplot(data=sent_summary, x="Classification", y="mean_leverage")
        plt.title("Average Leverage per Account by Sentiment")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(output_dir / "mean_leverage_by_sentiment.png")
        plt.close()

    print(f"[INFO] Plots saved in: {output_dir}")


# ========= 9. EXTENSION 1: TRADER SEGMENTATION =========

def run_trader_segmentation(daily_perf: pd.DataFrame, outputs_dir: Path):
    """
    Segment traders into High vs Low Leverage (if leverage exists),
    otherwise into High vs Low Activity (by n_trades).
    """
    if daily_perf.empty:
        print("\n[WARN] daily_perf is empty, skipping segmentation.")
        return

    df = daily_perf.copy()

    if "avg_leverage" in df.columns and not df["avg_leverage"].isna().all():
        metric_col = "avg_leverage"
        label_high = "High Leverage"
        label_low = "Low Leverage"
    else:
        metric_col = "n_trades"
        label_high = "High Activity"
        label_low = "Low Activity"

    threshold = df[metric_col].median()
    df["segment"] = np.where(df[metric_col] >= threshold, label_high, label_low)

    segment_summary = (
        df.groupby(["segment", "Classification"])
        .agg(
            mean_pnl=("total_pnl", "mean"),
            win_rate=("win_rate", "mean"),
            n_days=("trade_date", "size"),
        )
        .reset_index()
    )

    print("\n=== Trader Segmentation Summary ===")
    print(segment_summary)

    segment_summary.to_csv(outputs_dir / "segment_summary.csv", index=False)


# ========= 10. EXTENSION 2: COIN-LEVEL ANALYSIS =========

def run_coin_level_analysis(merged: pd.DataFrame, outputs_dir: Path):
    """
    Analyze performance by Coin and Sentiment classification.
    """
    if merged.empty:
        print("\n[WARN] merged is empty, skipping coin-level analysis.")
        return

    required_cols = {"Coin", "Classification", "Closed PnL"}
    if not required_cols.issubset(set(merged.columns)):
        print("\n[WARN] Missing columns for coin analysis, skipping.")
        return

    coin_summary = (
        merged.groupby(["Coin", "Classification"])
        .agg(
            mean_pnl=("Closed PnL", "mean"),
            win_rate=("Closed PnL", lambda x: (x > 0).mean()),
            n_trades=("Closed PnL", "size"),
        )
        .reset_index()
    )

    print("\n=== Coin-Level Sentiment Summary (head) ===")
    print(coin_summary.head())

    coin_summary.to_csv(outputs_dir / "coin_sentiment_summary.csv", index=False)


# ========= 11. EXTENSION 3: LONG vs SHORT =========

def run_long_short_analysis(merged: pd.DataFrame, outputs_dir: Path):
    """
    Compare long vs short performance across sentiment regimes.
    """
    if merged.empty:
        print("\n[WARN] merged is empty, skipping long/short analysis.")
        return

    required_cols = {"Side", "Classification", "Closed PnL"}
    if not required_cols.issubset(set(merged.columns)):
        print("\n[WARN] Missing columns for long/short analysis, skipping.")
        return

    long_short_summary = (
        merged.groupby(["Side", "Classification"])
        .agg(
            mean_pnl=("Closed PnL", "mean"),
            win_rate=("Closed PnL", lambda x: (x > 0).mean()),
            n_trades=("Closed PnL", "size"),
        )
        .reset_index()
    )

    print("\n=== Long vs Short Performance by Sentiment ===")
    print(long_short_summary)

    long_short_summary.to_csv(outputs_dir / "long_short_summary.csv", index=False)


# ========= 12. EXTENSION 4: SIMPLE PREDICTIVE MODEL =========

def run_profitability_model(daily_perf: pd.DataFrame, outputs_dir: Path):
    """
    Simple RandomForest model to predict whether a day is profitable (total_pnl > 0)
    using aggregated features + sentiment dummies.
    """
    if daily_perf.empty:
        print("\n[WARN] daily_perf empty, skipping model.")
        return

    df = daily_perf.copy()
    df["profit_day"] = (df["total_pnl"] > 0).astype(int)

    if df["profit_day"].nunique() < 2:
        print("\n[WARN] Only one class in profit_day, skipping model.")
        return

    account_col = df.attrs.get("account_col", "Account")

    # One-hot encode sentiment
    if "Classification" in df.columns:
        df = pd.get_dummies(df, columns=["Classification"], drop_first=True)

    # Drop non-numeric / ID / date columns
    drop_cols = [account_col, "trade_date"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Keep only numeric features
    feature_cols = [c for c in df.columns if c != "profit_day" and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols]
    y = df["profit_day"]

    if X.empty:
        print("\n[WARN] No numeric features left after filtering, skipping model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"\n=== Profitability Model Accuracy: {acc:.3f} ===")

    # Feature importance
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 10 Features by Importance:")
    print(importances.head(10))

    importances.to_csv(outputs_dir / "model_feature_importances.csv", index=False)

    with open(outputs_dir / "model_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("Top features:\n")
        f.write(importances.head(20).to_string(index=False))


# ========= 13. MAIN =========

def main():
    print("=========== STEP 1: LOADING DATA ===========")
    trades, sentiment = load_data()

    trades = preprocess_trades(trades)
    sentiment_small = preprocess_sentiment(sentiment)
    merged = merge_trades_sentiment(trades, sentiment_small)

    print("\n=========== STEP 5+: PERFORMANCE & SUMMARY ===========")

    # Build daily per-account performance
    daily_perf = build_daily_performance(merged)

    # Summarize by sentiment regime (Fear/Greed buckets)
    sent_summary = summarize_by_sentiment(daily_perf)

    # Save summary CSV & plots
    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True, parents=True)

    sent_summary.to_csv(outputs_dir / "sentiment_summary.csv", index=False)
    print(f"\n[INFO] Saved sentiment_summary.csv to: {outputs_dir}")

    plot_sentiment_summary(sent_summary, base_dir)

    # ---------- Extensions ----------
    run_trader_segmentation(daily_perf, outputs_dir)
    run_coin_level_analysis(merged, outputs_dir)
    run_long_short_analysis(merged, outputs_dir)
    run_profitability_model(daily_perf, outputs_dir)

    print("\n=========== ALL DONE ===========")
    print("Final sentiment summary:")
    print(sent_summary)


if __name__ == "__main__":
    main()
