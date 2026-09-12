"""
01_data_cleaning.py
E-commerce Return Rate Reduction Analysis — Step 1: Data Cleaning

Input : ecommerce_returns_synthetic_data.csv (raw)
Output: cleaned_ecommerce_returns.csv, data_quality_report.txt
"""

import pandas as pd
import numpy as np

RAW_PATH = "ecommerce_returns_synthetic_data.csv"
OUT_CSV = "outputs/cleaned_ecommerce_returns.csv"
REPORT_PATH = "outputs/data_quality_report.txt"

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


def main():
    df = pd.read_csv(RAW_PATH, parse_dates=["Order_Date", "Return_Date"])
    log(f"Raw rows loaded: {len(df)}")

    # ---------------------------------------------------------------
    # 1. Duplicates
    # ---------------------------------------------------------------
    dupes = df.duplicated(subset=["Order_ID"]).sum()
    log(f"Duplicate Order_IDs found: {dupes}")
    df = df.drop_duplicates(subset=["Order_ID"])

    # ---------------------------------------------------------------
    # 2. Standardize text fields
    # ---------------------------------------------------------------
    text_cols = ["Product_Category", "Return_Reason", "Return_Status",
                 "User_Gender", "User_Location", "Payment_Method", "Shipping_Method"]
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["nan", "None", ""]), c] = np.nan

    # ---------------------------------------------------------------
    # 3. Binary target flag
    # ---------------------------------------------------------------
    df["Is_Returned"] = (df["Return_Status"].str.lower() == "returned").astype(int)
    log(f"Return rate overall: {df['Is_Returned'].mean():.2%}")

    # ---------------------------------------------------------------
    # 4. Data quality issue: Days_to_Return / Return_Date vs Order_Date
    #    Days_to_Return == (Return_Date - Order_Date).days exactly, but
    #    since the two dates were generated independently in the source
    #    data, ~50% of returned orders show a NEGATIVE gap (i.e. the
    #    return happened before the order was even placed). This is a
    #    synthetic-data artifact, not a real-world return pattern.
    # ---------------------------------------------------------------
    returned = df["Is_Returned"] == 1
    neg_gap = (df.loc[returned, "Days_to_Return"] < 0).sum()
    log(f"Returned rows with impossible negative Days_to_Return: {neg_gap} "
        f"out of {returned.sum()} ({neg_gap/returned.sum():.1%})")

    # Flag them explicitly instead of silently dropping (keeps dataset size intact,
    # analysts can filter Data_Quality_Flag == 'OK' for time-based analysis).
    df["Data_Quality_Flag"] = "OK"
    df.loc[returned & (df["Days_to_Return"] < 0), "Data_Quality_Flag"] = "Invalid_Return_Timing"

    # A cleaned, always-positive version of the field for any analysis/model
    # that needs a magnitude (e.g. "how many days did the return process take").
    df["Days_to_Return_Abs"] = df["Days_to_Return"].abs()

    # ---------------------------------------------------------------
    # 5. Missing values
    # ---------------------------------------------------------------
    log("\nMissing values per column:")
    log(df.isna().sum().to_string())
    # Return_Reason / Return_Date / Days_to_Return are legitimately missing
    # for non-returned orders — that is NOT bad data, so we leave those as-is
    # and just make it explicit:
    df["Return_Reason"] = df["Return_Reason"].fillna("Not Applicable")

    # ---------------------------------------------------------------
    # 6. Sanity-check numeric ranges
    # ---------------------------------------------------------------
    log("\nProduct_Price range: {:.2f} to {:.2f}".format(df.Product_Price.min(), df.Product_Price.max()))
    log("Order_Quantity range: {} to {}".format(df.Order_Quantity.min(), df.Order_Quantity.max()))
    log("User_Age range: {} to {}".format(df.User_Age.min(), df.User_Age.max()))
    log("Discount_Applied range: {:.2f} to {:.2f}".format(df.Discount_Applied.min(), df.Discount_Applied.max()))

    # ---------------------------------------------------------------
    # 7. Derived fields useful for BI + modelling
    # ---------------------------------------------------------------
    df["Order_Value"] = df["Product_Price"] * df["Order_Quantity"]
    df["Discount_Pct_Of_Price"] = (df["Discount_Applied"] / df["Product_Price"]).clip(upper=1) * 100

    age_bins = [0, 25, 35, 45, 55, 65, 200]
    age_labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["Age_Group"] = pd.cut(df["User_Age"], bins=age_bins, labels=age_labels, right=False)

    price_bins = [0, 100, 200, 300, 400, 1000]
    price_labels = ["<$100", "$100-199", "$200-299", "$300-399", "$400+"]
    df["Price_Band"] = pd.cut(df["Product_Price"], bins=price_bins, labels=price_labels)

    df["Order_Year"] = df["Order_Date"].dt.year
    df["Order_Month"] = df["Order_Date"].dt.to_period("M").astype(str)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    df.to_csv(OUT_CSV, index=False)
    log(f"\nCleaned dataset saved: {OUT_CSV}  ({len(df)} rows, {df.shape[1]} columns)")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
