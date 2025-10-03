import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np  # Added for numeric handling
import json
import sqlite3  # Built-in for SQL
import os

# Page config
st.set_page_config(page_title="Customer Shopping Trends", layout="wide")

# Title and description
st.title("🛒 Customer Shopping Trends Analysis")
st.markdown("""
This interactive dashboard analyzes shopping trends from the Kaggle dataset. 
Filter by demographics and explore insights on revenue, payments, and discounts.
""")

# Load data (with caching)
@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_shopping_data.csv')
    return df

df = load_data()

# NEW: Data preparation function - Convert strings to numerics where needed
def prepare_data(df):
    df_prep = df.copy()
    
    # Convert categorical to numeric for calculations (e.g., Yes/No to 1/0)
    if 'Discount Applied' in df_prep.columns:
        df_prep['Discount Applied Numeric'] = df_prep['Discount Applied'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    if 'Subscription Status' in df_prep.columns:
        df_prep['Subscription Status Numeric'] = df_prep['Subscription Status'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    # Ensure other key columns are numeric (fallback if cleaning missed them)
    numeric_cols = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    for col in numeric_cols:
        if col in df_prep.columns:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce').fillna(df_prep[col].median() if col in df_prep.columns else 0)
    
    return df_prep

# Prepare the data
df = prepare_data(df)

# Sidebar for filters
st.sidebar.header("🔍 Filters")
if 'Category' in df.columns:
    category_filter = st.sidebar.multiselect("Select Category", options=df['Category'].unique(), default=df['Category'].unique())
else:
    category_filter = df['Category'].unique() if 'Category' in df.columns else []

gender_options = ['All'] + list(df['Gender'].unique()) if 'Gender' in df.columns else ['All']
gender_filter = st.sidebar.selectbox("Select Gender", options=gender_options, index=0)

if 'Age' in df.columns:
    age_range = st.sidebar.slider("Age Range", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(int(df['Age'].min()), int(df['Age'].max())))
else:
    age_range = (0, 100)

# Apply filters
filtered_df = df[df['Category'].isin(category_filter)] if 'Category' in df.columns else df
if gender_filter != 'All' and 'Gender' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
if 'Age' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]

if filtered_df.empty:
    st.warning("No data matches the selected filters. Adjust and try again.")
else:
    # Display raw data (collapsible)
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)

    # Key Metrics (KPIs) - FIXED: Use numeric version for Discount
    st.header("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_customers = filtered_df['Customer ID'].nunique() if 'Customer ID' in filtered_df.columns else len(filtered_df)
        st.metric("Total Customers", total_customers)
    with col2:
        avg_purchase = filtered_df['Purchase Amount (USD)'].mean() if 'Purchase Amount (USD)' in filtered_df.columns else 0
        st.metric("Avg Purchase Amount", f"${avg_purchase:.2f}")
    with col3:
        # FIXED: Use the numeric column for sum
        discount_rate = (filtered_df['Discount Applied Numeric'].sum() / len(filtered_df)) * 100 if 'Discount Applied Numeric' in filtered_df.columns else 0
        st.metric("Discount Usage", f"{discount_rate:.1f}%")
    with col4:
        top_category = filtered_df.groupby('Category')['Purchase Amount (USD)'].sum().idxmax() if 'Category' in filtered_df.columns and 'Purchase Amount (USD)' in filtered_df.columns else "N/A"
        st.metric("Top Category", top_category)

    # Visualizations
    st.header("📊 Key Visualizations")

    # 1. Top Categories by Revenue (Bar Chart) - Safe for strings
    if 'Category' in filtered_df.columns and 'Purchase Amount (USD)' in filtered_df.columns:
        revenue_by_category = filtered_df.groupby('Category')['Purchase Amount (USD)'].sum().reset_index()
        fig1 = px.bar(revenue_by_category, x='Category', y='Purchase Amount (USD)', title="Top Categories by Revenue")
        st.plotly_chart(fig1, use_container_width=True)

    # 2. Payment Methods (Pie Chart) - Handles strings
    if 'Payment Method' in filtered_df.columns:
        payment_counts = filtered_df['Payment Method'].value_counts()
        fig2 = px.pie(values=payment_counts.values, names=payment_counts.index, title="Payment Methods Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    # 3. Discount Impact (Box Plot) - FIXED: Use numeric for x-axis if needed, but original strings work for grouping
    if 'Discount Applied' in filtered_df.columns and 'Purchase Amount (USD)' in filtered_df.columns:
        fig3 = px.box(filtered_df, x='Discount Applied', y='Purchase Amount (USD)', title="Impact of Discounts on Purchase Amount")
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Purchase Frequency (Histogram) - Handles strings
    if 'Frequency of Purchases' in filtered_df.columns:
        fig4 = px.histogram(filtered_df, x='Frequency of Purchases', title="Purchase Frequency Distribution")
        st.plotly_chart(fig4, use_container_width=True)

# ENHANCED: Dynamic Insights Section - Load from Jupyter JSON + Live SQL Queries
st.header("💡 Key Insights & Project Integration")

# Define absolute project root (your path)
project_root = r'D:\Customer-Shopping-Trends-Analysis'

# DYNAMIC: Load Jupyter Insights from JSON
try:
    json_path = os.path.join(project_root, 'insights.json')
    print(f"App looking for JSON at: {json_path}")  # Debug in terminal
    
    with open(json_path, 'r') as f:
        jupyter_insights = json.load(f)
    st.success("✅ Loaded real Jupyter insights (EDA from analysis)")
    
    # Improved Display: Handles low/negative values from your data
    highest_cat = max(jupyter_insights['avg_by_category'], key=jupyter_insights['avg_by_category'].get)
    lowest_cat = min(jupyter_insights['avg_by_category'], key=jupyter_insights['avg_by_category'].get)
    highest_avg = max(jupyter_insights['avg_by_category'].values())
    lowest_avg = min(jupyter_insights['avg_by_category'].values())
    boost_desc = "higher" if jupyter_insights['discount_boost'] > 0 else "lower"
    boost_sign = "+" if jupyter_insights['discount_boost'] > 0 else ""
    
    st.markdown(f"""
    **From Python Analysis :**
    - Correlation ({jupyter_insights['key_corr_prev']:.3f}) between Previous Purchases and Purchase Amount – weak relationship; loyalty has minimal impact on spend.
    - Correlation ({jupyter_insights['key_corr_age']:.3f}) between Age and Purchase Amount; spending is similar across ages.
    - Trend: Highest average purchase in '{highest_cat}' category (${highest_avg:.2f}), lowest in '{lowest_cat}' (${lowest_avg:.2f}).
    - Discount Boost: Purchases with discounts average {boost_sign}{jupyter_insights['discount_boost']:.1f}% {boost_desc} (${jupyter_insights['yes_avg']:.2f} vs. ${jupyter_insights['no_avg']:.2f} without).
    Explore the notebook for full EDA, including scatter plots and statistical tests. [Visual: Correlation Heatmap](../visuals/correlation_heatmap.png)
    """)
    
    # Optional: Bar chart from JSON
    import pandas as pd  # Already imported, but safe
    cat_df = pd.DataFrame(list(jupyter_insights['avg_by_category'].items()), columns=['Category', 'Avg Purchase'])
    st.subheader("Average Purchase by Category")
    st.bar_chart(cat_df.set_index('Category'))
    
except FileNotFoundError:
    st.warning(f"⚠️ {json_path} not found – Run Jupyter Cell 6 to generate.")
    st.markdown("""
    **From Python Analysis (Sample):**
    - Strong positive correlation (0.15) between Previous Purchases and Purchase Amount – loyal customers spend more.
    - Weak negative correlation (-0.05) between Age and Purchase Amount; spending is similar across ages, but younger shoppers (18-30) use discounts 30% more.
    - Trend: Highest average purchase in 'Accessories' category ($53.45), lowest in 'Books' ($49.12).
    - Discount Boost: Purchases with discounts average 25.3% higher ($55.20 vs. $44.10 without).
    Explore the notebook for full EDA, including scatter plots and statistical tests. [Visual: Correlation Heatmap](../visuals/correlation_heatmap.png)
    """)
except (KeyError, ValueError) as e:
    st.error(f"⚠️ JSON format issue: {e}. Rerun Jupyter Cell 6 for complete data.")
    st.markdown("**From Python Analysis (Sample):** ...")  # Fallback

# DYNAMIC: Live SQL Queries from shopping.db
db_path = os.path.join(project_root, 'sql_queries', 'shopping.db')
print(f"App looking for DB at: {db_path}")  # Debug in terminal

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    st.success("✅ Connected to SQLite DB – Loading real query results")
    
    # Query 1: Top 5 Locations
    top_locations = conn.execute("""
        SELECT Location, SUM("Purchase Amount (USD)") AS Total_Revenue, 
            COUNT(*) AS Num_Purchases, AVG("Purchase Amount (USD)") AS Avg_Purchase
        FROM shopping_data GROUP BY Location ORDER BY Total_Revenue DESC LIMIT 5;
    """).fetchall()
    locations_str = "; ".join([f"{row[0]} (${row[1]:,.0f} | {row[2]} purchases)" for row in top_locations])
    
    # Query 2: Highest Discount Usage
    discount_usage = conn.execute("""
        SELECT Location, SUM(CASE WHEN "Discount Applied" = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS Discount_Percentage
        FROM shopping_data GROUP BY Location ORDER BY Discount_Percentage DESC LIMIT 5;
    """).fetchall()
    discount_str = "; ".join([f"{row[0]} ({row[1]:.0f}%)" for row in discount_usage])
    
    # Query 3: Top High-Value Customer (First One)
    top_customer = conn.execute("""
        SELECT "Customer ID", SUM("Purchase Amount (USD)") AS Total_Spent, COUNT(*) AS Num_Purchases
        FROM shopping_data GROUP BY "Customer ID" ORDER BY Total_Spent DESC LIMIT 1;
    """).fetchall()
    customer_str = f"Top spender (Customer ID: {top_customer[0][0]}) total ${top_customer[0][1]:,.0f} across {top_customer[0][2]} purchases" if top_customer else "N/A"
    
    conn.close()
    
    # Display SQL Insights
    st.markdown(f"""
    **From SQL Analysis :**
    - Top 5 Locations by Revenue: {locations_str}.
    - Highest Discount Usage: {discount_str} – target promotions there.
    - High-Value Customers: {customer_str}.
    - Query Example: `SELECT Location, SUM("Purchase Amount (USD)") AS Total_Revenue FROM shopping_data GROUP BY Location ORDER BY Total_Revenue DESC LIMIT 5;`
    Run the SQL file in SQLite for more queries on subscriptions and seasons.
    """)
else:
    st.warning(f"⚠️ {db_path} not found – Run SQLite import to generate.")
    st.markdown("""
    **From SQL Analysis (Sample):**
    - Top 5 Locations by Revenue: Los Angeles ($45,200 | 780 purchases), San Diego ($38,900 | 720), New York ($35,600 | 680), Boston ($32,100 | 650), Miami ($29,800 | 620).
    - Highest Discount Usage: Miami (55%), Los Angeles (52%) – target promotions there.
    - High-Value Customers: Top spender (Customer ID: 3783) total $1,250 across 25 purchases.
    - Query Example: `SELECT Location, SUM("Purchase Amount (USD)") AS Total_Revenue FROM shopping_data GROUP BY Location ORDER BY Total_Revenue DESC LIMIT 5;`
    Run the SQL file in SQLite for more queries on subscriptions and seasons.
    """)


# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data from Kaggle | [GitHub Repo](https://github.com/ShreeShreeRavi/Customer-Shopping-Trends-Analysis)")
st.markdown("© 2025 Shree Shree Rajeshwari Ravi")