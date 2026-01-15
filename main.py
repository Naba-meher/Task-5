"""
Customer Sales Analysis
-----------------------
Analyze customer purchasing patterns, identify top customers,
and generate sales insights with visualizations.

Author: <Your Name>
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """Load sales and customer datasets"""
    sales = pd.read_csv("sales_data.csv")
    customers = pd.read_csv("customer_data.csv")
    return sales, customers


def preprocess_data(sales, customers):
    """Clean, prepare, and merge data"""
    # Convert date
    sales["order_date"] = pd.to_datetime(sales["order_date"])

    # Calculated columns
    sales["revenue"] = sales["quantity"] * sales["price"]
    sales["month"] = sales["order_date"].dt.to_period("M")

    # Merge datasets (mandatory requirement)
    df = pd.merge(sales, customers, on="customer_id", how="inner")
    return df


def customer_analysis(df):
    """Customer-level aggregations"""
    customer_revenue = (
        df.groupby("customer_name")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    clv = (
        df.groupby("customer_id")
        .agg(
            total_spent=("revenue", "sum"),
            total_orders=("order_id", "nunique"),
        )
    )
    clv["avg_order_value"] = clv["total_spent"] / clv["total_orders"]

    return customer_revenue, clv


def sales_analysis(df):
    """Sales pattern analysis"""
    monthly_sales = (
        df.groupby("month")["revenue"]
        .sum()
        .reset_index()
    )

    top_products = (
        df.groupby("product_name")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    region_sales = (
        df.groupby("region")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    return monthly_sales, top_products, region_sales


def pivot_analysis(df):
    """Pivot table for summarization (mandatory requirement)"""
    pivot_table = pd.pivot_table(
        df,
        values="revenue",
        index="region",
        columns="product_name",
        aggfunc="sum"
    )
    return pivot_table


def create_visualizations(monthly_sales, top_customers, top_products, region_sales):
    """Generate professional visualizations"""

    # Monthly Sales Trend
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_sales["month"].astype(str), monthly_sales["revenue"])
    plt.xticks(rotation=45)
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

    # Top Customers
    top_customers.head(10).plot(
        kind="bar", figsize=(10, 5), title="Top 10 Customers by Revenue"
    )
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

    # Best-Selling Products
    top_products.plot(
        kind="barh", figsize=(8, 6), title="Top Products by Revenue"
    )
    plt.tight_layout()
    plt.show()

    # Region-wise Sales
    region_sales.plot(
        kind="pie", autopct="%1.1f%%", figsize=(7, 7),
        title="Sales Distribution by Region"
    )
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def print_summary(df, customer_revenue):
    """Print final report output"""
    total_revenue = df["revenue"].sum()
    total_customers = df["customer_id"].nunique()
    avg_order_value = df.groupby("order_id")["revenue"].sum().mean()

    top_customer = customer_revenue.index[0]
    top_customer_value = customer_revenue.iloc[0]

    print("\nCUSTOMER SALES ANALYSIS REPORT")
    print("-" * 35)
    print(f"Total Revenue: ${total_revenue:,.0f}")
    print(f"Total Customers: {total_customers}")
    print(f"Average Order Value: ${avg_order_value:,.0f}")
    print(f"Top Customer: {top_customer} - ${top_customer_value:,.0f}")


def main():
    sales, customers = load_data()
    df = preprocess_data(sales, customers)

    customer_revenue, clv = customer_analysis(df)
    monthly_sales, top_products, region_sales = sales_analysis(df)
    pivot_table = pivot_analysis(df)

    create_visualizations(
        monthly_sales,
        customer_revenue,
        top_products,
        region_sales
    )

    print_summary(df, customer_revenue)

    # Optional: save pivot table
    pivot_table.to_csv("region_product_pivot.csv")


if __name__ == "__main__":
    main()
