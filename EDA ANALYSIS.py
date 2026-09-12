"""
02_eda_analysis.py
E-commerce Return Rate Reduction Analysis — Step 2: Exploratory Data Analysis

Analyzes return % by: Product_Category, User_Location (geography),
Payment_Method / Shipping_Method (channel proxies), Return_Reason,
price band, age group, discount level.

Outputs: CSV summary tables + PNG charts under outputs/ and charts/
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("outputs/cleaned_ecommerce_returns.csv")

CHART_DIR = "charts"


def return_rate_table(group_col, min_n=1):
    g = df.groupby(group_col).agg(
        Orders=("Order_ID", "count"),
        Returns=("Is_Returned", "sum"),
    )
    g["Return_Rate_%"] = (g["Returns"] / g["Orders"] * 100).round(2)
    g = g[g["Orders"] >= min_n].sort_values("Return_Rate_%", ascending=False)
    return g


def save_bar(table, title, fname, top_n=None, figsize=(9, 5)):
    t = table.copy()
    if top_n:
        t = t.head(top_n)
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=t.index.astype(str), y=t["Return_Rate_%"], color="#4C72B0")
    ax.axhline(df["Is_Returned"].mean() * 100, ls="--", color="red", lw=1,
               label=f"Overall avg ({df['Is_Returned'].mean()*100:.1f}%)")
    ax.set_ylabel("Return Rate (%)")
    ax.set_xlabel("")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/{fname}", dpi=150)
    plt.close()


# ---------------------------------------------------------------
# 1. Return rate by Product Category
# ---------------------------------------------------------------
cat_table = return_rate_table("Product_Category")
cat_table.to_csv("outputs/return_rate_by_category.csv")
save_bar(cat_table, "Return Rate by Product Category", "return_rate_by_category.png")
print("By Category:\n", cat_table, "\n")

# ---------------------------------------------------------------
# 2. Return rate by Geography (User_Location) — top 15 by volume
# ---------------------------------------------------------------
geo_table = return_rate_table("User_Location", min_n=30)
geo_table.to_csv("outputs/return_rate_by_location.csv")
save_bar(geo_table.sort_values("Return_Rate_%", ascending=False), "Highest Return Rate Locations (Top 15)",
         "return_rate_by_location_top15.png", top_n=15)
print("By Location (top 5):\n", geo_table.head(), "\n")

# ---------------------------------------------------------------
# 3. Return rate by channel proxies: Payment_Method, Shipping_Method
# ---------------------------------------------------------------
pay_table = return_rate_table("Payment_Method")
pay_table.to_csv("outputs/return_rate_by_payment_method.csv")
save_bar(pay_table, "Return Rate by Payment Method", "return_rate_by_payment_method.png")

ship_table = return_rate_table("Shipping_Method")
ship_table.to_csv("outputs/return_rate_by_shipping_method.csv")
save_bar(ship_table, "Return Rate by Shipping Method", "return_rate_by_shipping_method.png")

print("By Payment Method:\n", pay_table, "\n")
print("By Shipping Method:\n", ship_table, "\n")

# ---------------------------------------------------------------
# 4. Category x Location cross tab (heatmap) - why returns happen where
# ---------------------------------------------------------------
top_locations = df["User_Location"].value_counts().head(15).index
pivot = df[df["User_Location"].isin(top_locations)].pivot_table(
    index="User_Location", columns="Product_Category", values="Is_Returned", aggfunc="mean"
) * 100
plt.figure(figsize=(9, 7))
sns.heatmap(pivot.round(1), annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "Return Rate %"})
plt.title("Return Rate (%): Category x Top 15 Locations")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/heatmap_category_location.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Return reasons distribution (among returned orders only)
# ---------------------------------------------------------------
reason_counts = df[df["Is_Returned"] == 1]["Return_Reason"].value_counts()
reason_pct = (reason_counts / reason_counts.sum() * 100).round(1)
reason_pct.to_csv("outputs/return_reason_distribution.csv")
plt.figure(figsize=(7, 5))
sns.barplot(x=reason_pct.values, y=reason_pct.index, color="#DD8452")
plt.xlabel("% of Returns")
plt.title("Return Reason Distribution")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/return_reason_distribution.png", dpi=150)
plt.close()
print("Return reasons:\n", reason_pct, "\n")

# Return reason by category (which category returns for which reason most)
reason_cat = pd.crosstab(df.loc[df["Is_Returned"] == 1, "Product_Category"],
                          df.loc[df["Is_Returned"] == 1, "Return_Reason"], normalize="index") * 100
reason_cat.round(1).to_csv("outputs/return_reason_by_category.csv")

# ---------------------------------------------------------------
# 6. Price band & discount vs return rate
# ---------------------------------------------------------------
price_table = return_rate_table("Price_Band")
price_table.to_csv("outputs/return_rate_by_price_band.csv")
save_bar(price_table.reindex(["<$100", "$100-199", "$200-299", "$300-399", "$400+"]).dropna(),
         "Return Rate by Price Band", "return_rate_by_price_band.png")

df["Discount_Band"] = pd.cut(df["Discount_Applied"], bins=[-1, 10, 20, 30, 40, 51],
                              labels=["0-10", "10-20", "20-30", "30-40", "40-50"])
disc_table = return_rate_table("Discount_Band")
disc_table.to_csv("outputs/return_rate_by_discount_band.csv")

# ---------------------------------------------------------------
# 7. Age group & gender
# ---------------------------------------------------------------
age_table = return_rate_table("Age_Group")
age_table.to_csv("outputs/return_rate_by_age_group.csv")

gender_table = return_rate_table("User_Gender")
gender_table.to_csv("outputs/return_rate_by_gender.csv")

# ---------------------------------------------------------------
# 8. Monthly trend
# ---------------------------------------------------------------
trend = df.groupby("Order_Month").agg(Orders=("Order_ID", "count"), Returns=("Is_Returned", "sum"))
trend["Return_Rate_%"] = (trend["Returns"] / trend["Orders"] * 100).round(2)
trend.to_csv("outputs/return_rate_monthly_trend.csv")
plt.figure(figsize=(11, 4.5))
plt.plot(trend.index, trend["Return_Rate_%"], marker="o", ms=3)
plt.xticks(rotation=60, fontsize=7)
plt.ylabel("Return Rate (%)")
plt.title("Monthly Return Rate Trend")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/monthly_trend.png", dpi=150)
plt.close()

print("EDA complete. Tables saved to outputs/, charts saved to charts/.")
