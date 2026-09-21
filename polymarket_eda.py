"""
Polymarket Trader EDA
=====================

Exploratory analysis of a Polymarket trader dataset: one row per trader,
with profit/loss, volume, trade frequency, execution behavior, topic
exposure, and a skill label (awful / bad / good / sharp).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

DATA_PATH = r"C:\Users\angel\Downloads\SIF EDA\data.parquet"

BLACK, RED, YELLOW, BLUE, GREEN = "#000000", "#FF0000", "#FFFF00", "#0000FF", "#00FF00"
LABEL_COLORS = {"awful": RED, "bad": YELLOW, "good": BLUE, "sharp": GREEN}
LABEL_ORDER = ["awful", "bad", "good", "sharp"]

plt.rcParams.update({
    "font.size": 10.5,
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df = pd.read_parquet(DATA_PATH)

#OBSERVATION 1
print("=" * 70)
print("1. SHAPE & DATA QUALITY")
print("=" * 70)
print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
print(f"Duplicate trader IDs: {df['trader'].duplicated().sum()}")

missing = df.isna().sum()
missing = missing[missing > 0]
print("\nColumns with missing values:")
print(pd.DataFrame({
    "missing_count": missing,
    "missing_pct": (missing / len(df) * 100).round(2),
}))


inf_counts = {c: int(np.isinf(df[c]).sum()) for c in df.select_dtypes("number").columns
    if np.isinf(df[c]).sum() > 0}

print("\nColumns with infinite values:", inf_counts or "none")

one_trade_traders = (df["transaction_count"] == 1).sum()
for col in ["std_delta", "std_time", "std_tx_value"]:
    lines_up = (df.loc[df[col].isna(), "transaction_count"] == 1).mean() * 100
    print(f"  {col}: {lines_up:.1f}% of its missing rows are 1-trade traders")

why = {
    "std_delta": "Std. dev. undefined for 1-trade traders",
    "std_time": "Std. dev. undefined for 1-trade traders",
    "std_tx_value": "Std. dev. undefined for 1-trade traders",
    "std_time_vw": f"Undefined for traders with 1 distinct trade time; also has {inf_counts.get('std_time_vw', 0)} infinite value",
}

missing_table = pd.DataFrame({
    "Missing": missing,
    "% of rows": (missing / len(df) * 100).round(2),
    "Why": [why.get(c, "") for c in missing.index],
})

key_cols = [
    "trader_pnl", "trader_volume", "transaction_count",
    "transactions_per_day", "mean_tx_value", "trader_ppv",
]
print("\nSummary statistics (core activity and profitability columns):")
print(df[key_cols].describe().T.round(3))

corr_cols = [
    "trader_pnl", "trader_volume", "transaction_count", "transactions_per_day",
    "volume_per_day", "markets_per_day", "mean_delta", "mean_tx_value", "trader_ppv",
]

corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=90)
ax.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(v) > 0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.85)
plt.savefig("correlation_heatmap.png", dpi=200, bbox_inches="tight")

#OBSERVATION 2
print("\n" + "=" * 70)
print("2. PROFIT/LOSS AND VOLUME")
print("=" * 70)

total_pnl = df["trader_pnl"].sum()
pct_profitable = (df["trader_pnl"] > 0).mean() * 100
pct_losing = (df["trader_pnl"] < 0).mean() * 100
pct_even = (df["trader_pnl"] == 0).mean() * 100

print(f"Total PnL across all traders: ${total_pnl:.2f}  (should be ~$0 in a zero-sum market)")
print(f"Profitable traders: {pct_profitable:.1f}%   Losing traders: {pct_losing:.1f}%   Exactly even: {pct_even:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

pnl = df["trader_pnl"].replace([np.inf, -np.inf], np.nan).dropna()
lo, hi = pnl.quantile(0.05), pnl.quantile(0.95)
pnl_mid = pnl[(pnl >= lo) & (pnl <= hi)]
axes[0].hist(pnl_mid, bins=50, color="black", edgecolor="white", linewidth = 0.3)
axes[0].axvline(0, color="black", linewidth=1, linestyle = "--", alpha = 0.6)
axes[0].set_title("Trader pnl", loc="left")
axes[0].set_xlabel("Profit / loss in dollars (middle 90% of traders)")
axes[0].set_ylabel("# of traders")

vol = df["trader_volume"].replace([np.inf, -np.inf], np.nan).dropna()
vol = vol[vol > 0]
axes[1].hist(np.log10(vol), bins=45, color="black", edgecolor="white", linewidth=0.3)
axes[1].set_title("Trader volume", loc="left")
axes[1].set_xlabel("Total amount traded, log scale (0=$1, 2=$100, 4=$10k)", fontsize=9.5)
axes[1].set_ylabel("# of traders")

plt.tight_layout()
plt.savefig("fig1_distributions.png", dpi=200)
plt.close()
print("Saved fig1_distributions.png")

#OBSERVATION 3
print("\n" + "=" * 70)
print("3. WHAT trader_label ACTUALLY IS")
print("=" * 70)

sorted_df = df.sort_values("trader_ppv")
boundaries = []
for label, group in sorted_df.groupby("trader_label", sort=False):
    boundaries.append((label, group["trader_ppv"].min(), group["trader_ppv"].max()))
boundaries.sort(key=lambda x: x[1])

print("Label ranges, sorted by trader_ppv (profit per dollar traded):")
for label, lo_val, hi_val in boundaries:
    n = (df["trader_label"] == label).sum()
    print(f"  {label:6s}: {lo_val:>10.4f} to {hi_val:>10.4f}   "
        f"({n:,} traders, {n/len(df)*100:.1f}%)")

overlaps = any(boundaries[i][2] >= boundaries[i + 1][1] for i in range(len(boundaries) - 1))
print(f"\nAny overlap between adjacent label ranges? {overlaps}")
print("(if False, trader_label is fully determined by trader_ppv alone)\n")

label_stats = df.groupby("trader_label")[["trader_pnl", "trader_volume"]].agg(
    mean_pnl=("trader_pnl", "mean"),
    median_pnl=("trader_pnl", "median"),
    mean_volume=("trader_volume", "mean"),
).reindex(LABEL_ORDER)

print(label_stats.round(2).rename(columns={
    "mean_pnl": "Mean PnL", "median_pnl": "Median PnL", "mean_volume": "Mean Volume",
}))

fig, ax = plt.subplots(figsize=(9.5, 4.5))
CLIP = 0.25
bins = np.linspace(-CLIP, CLIP, 121)
for label in LABEL_ORDER:
    sub = df.loc[df["trader_label"] == label, "trader_ppv"].clip(-CLIP, CLIP)
    ax.hist(sub, bins=bins, color=LABEL_COLORS[label], label=label, alpha=0.85,
            edgecolor="white", linewidth=0.15)

cut_points = [boundaries[i + 1][1] for i in range(len(boundaries) - 1)]
ymax = ax.get_ylim()[1]
for c in cut_points:
    ax.axvline(c, color="black", linewidth=1.1, linestyle="--", alpha=0.7)
    ax.annotate(f"{c:.3f}", xy=(c, ymax), xytext=(c, ymax * 1.04), fontsize=8.7,
                ha="center", annotation_clip=False)

ax.set_xlabel(f"Return per dollar traded (values past ±{CLIP} are grouped into the end bars)")
ax.set_ylabel("Number of traders")
ax.set_title("The trader_label is just this one number, split into four ranges", pad=28)
ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
plt.tight_layout()
plt.savefig("fig2_ppv_thresholds.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved fig2_ppv_thresholds.png")

#OBSERVATION 4
print("\n" + "=" * 70)
print("4. HOW CONCENTRATED IS TRADING VOLUME?")
print("=" * 70)

vol_sorted = np.sort(df["trader_volume"].values)[::-1]  # largest first
cum_share = np.cumsum(vol_sorted) / vol_sorted.sum()
n = len(vol_sorted)

checkpoints = [0.001, 0.01, 0.10, 1.0]  # top 0.1%, 1%, 10%, everyone
labels_cp = ["Top 0.1%\nof traders", "Top 1%\nof traders", "Top 10%\nof traders", "Bottom 90%\nof traders"]
shares = []
prev = 0.0
for pct in checkpoints:
    k = int(n * pct)
    share = cum_share[k - 1] * 100
    shares.append(share - prev)
    prev = share
    print(f"Top {pct*100:>5.1f}% of traders account for {share:.1f}% of total volume (cumulative)")

fig, ax = plt.subplots(figsize=(6.2, 4.4))
bars = ax.bar(labels_cp, shares, color="black", width=0.6)
for b, v in zip(bars, shares):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center")
ax.set_ylabel("Share of total trading volume (%)")
ax.set_ylim(0, 60)
plt.tight_layout()
plt.savefig("fig3_volume_concentration.png", dpi=200)
plt.close()
print("Saved fig3_volume_concentration.png")

#OBSERVATION 5
print("\n" + "=" * 70)
print("5. DOES TRADING MORE OFTEN CHANGE THE SIZE OF A TRADER'S EDGE?")
print("=" * 70)

d = df.copy()
d["activity"] = pd.cut(
    d["transaction_count"],
    bins=[0, 3, 15, 10**9],
    labels=["Low activity\n(1-3 trades)", "Medium activity\n(4-15 trades)", "High activity\n(16+ trades)"],
)
edge_by_activity = d.groupby("activity", observed=True)["trader_ppv"].apply(lambda x: x.abs().mean())
print("Average size of return per dollar traded (ignoring win/loss direction), by activity level:")
print(edge_by_activity.round(3).rename(index=lambda s: s.replace("\n", " ")))

fig, ax = plt.subplots(figsize=(6.4, 4.2))
bars = ax.bar(edge_by_activity.index.astype(str), edge_by_activity.values,
            color=["black", "black", "black"], width=0.55)
for b, v in zip(bars, edge_by_activity.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center")
ax.set_ylabel("Average size of return per dollar traded\n(ignoring win/loss direction)")
ax.set_title("Traders who trade more often win or lose less per dollar", loc="left")
plt.tight_layout()
plt.savefig("fig4_activity_vs_edge.png", dpi=200)
plt.close()
print("Saved fig4_activity_vs_edge.png")

#OBSERVATION 6
print("\n" + "=" * 70)
print("6. DO THE BEST/WORST TRADERS TRADE DIFFERENTLY?")
print("=" * 70)

execution_by_label = df.groupby("trader_label")["price_levels_per_transaction"].mean().reindex(LABEL_ORDER)
print("Average order-book levels crossed per trade, by label:")
print(execution_by_label.round(4))

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(LABEL_ORDER, execution_by_label.values, color="black", width=0.55)

for b, v in zip(bars, execution_by_label.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.00008, f"{v:.4f}", ha="center", fontsize=9)
ax.set_ylabel("Average number of price\n levels crossed per trade")
plt.tight_layout()
plt.savefig("fig5_execution.png", dpi=200)
plt.close()
print("Saved fig5_execution.png")

#OBSERVATION 7
print("\n" + "=" * 70)
print("7. DOES WHAT A TRADER BETS ON RELATE TO HOW THEY DO?")
print("=" * 70)

topic_cols = [c for c in df.columns if c.startswith("topic_")]
topic_avg = df[topic_cols].mean().sort_values(ascending=False)
print("Average share of a trader's activity spent on each topic:")
print((topic_avg * 100).round(1).to_string())

topic_corr = df[topic_cols].corrwith(df["trader_ppv"]).sort_values()
print("\nHow each topic relates to profitability (correlation with trader_ppv, -1 to 1):")
print(topic_corr.round(3).to_string())
print("(all close to 0 = topic choice barely matters)")

plot_data = topic_avg.sort_values(ascending=True)
plot_data.index = [c.replace("topic_", "").split(",")[0].title() for c in plot_data.index]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.barh(plot_data.index, plot_data.values * 100, color="black")
ax.set_xlabel("Average share of a trader's activity (%)")
plt.tight_layout()
plt.savefig("fig6_topics.png", dpi=200)
plt.close()
print("Saved fig6_topics.png")

corr_plot = topic_corr.copy()
corr_plot.index = [c.replace("topic_", "").split(",")[0].title() for c in corr_plot.index]
corr_plot = corr_plot.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7.5, 5.2))
bar_colors = ["black" if v < 0 else "black" for v in corr_plot.values]
ax.barh(corr_plot.index, corr_plot.values, color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Correlation with return per dollar traded (trader_ppv)")
ax.set_xlim(-0.1, 0.1)
plt.tight_layout()
plt.savefig("fig7_topic_correlation.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig7_topic_correlation.png")


print("\n" + "=" * 70)
print("DONE. All 6 charts saved to the current folder.")
print("=" * 70)