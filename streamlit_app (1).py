import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        color: #2E86AB;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

COLOR1, COLOR2, COLOR3 = "#2E86AB", "#A23B72", "#F18F01"
CLUSTER_COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#4C956C", "#C73E1D"]

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

FORECAST_PERIODS = 3  # fixed backtest / max forecast horizon, matches the notebook

SEASON_MAP = {12: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3}

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_raw_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    df['Year'] = df['Order Date'].dt.year
    return df

@st.cache_data
def load_precomputed():
    monthly_sales = pd.read_csv('monthly_sales.csv')
    monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'])

    weekly_sales = pd.read_csv('weekly_sales_anomalies.csv')
    weekly_sales['Week'] = pd.to_datetime(weekly_sales['Week'])

    cluster_data = pd.read_csv('cluster_assignments.csv')
    model_comparison = pd.read_csv('model_comparison.csv')

    return monthly_sales, weekly_sales, cluster_data, model_comparison


try:
    df = load_raw_data()
    monthly_sales, weekly_sales, cluster_data, model_comparison = load_precomputed()
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.info("Make sure train.csv, monthly_sales.csv, weekly_sales_anomalies.csv, "
            "cluster_assignments.csv, and model_comparison.csv are in the same folder as this app.")
    st.stop()

REGIONS = sorted(df['Region'].unique().tolist())
CATEGORIES = sorted(df['Category'].unique().tolist())

# ============================================================
# PER-SEGMENT FORECASTING (XGBoost — the winning model from the notebook)
# ============================================================
def build_monthly_series(segment_col, segment_val):
    """Aggregate monthly sales, optionally filtered to one Category/Region.
    Reindexes to a complete monthly range (filling gaps with 0) so lag
    features never break on a missing month."""
    sub = df if segment_col is None else df[df[segment_col] == segment_val]

    monthly = sub.groupby(sub['Order Date'].dt.to_period('M'))['Sales'].sum()
    monthly.index = monthly.index.to_timestamp()

    full_idx = pd.date_range(
        df['Order Date'].dt.to_period('M').min().to_timestamp(),
        df['Order Date'].dt.to_period('M').max().to_timestamp(),
        freq='MS'
    )
    monthly = monthly.reindex(full_idx, fill_value=0)
    monthly.index.name = 'Month'
    monthly = monthly.asfreq('MS')
    return monthly


@st.cache_resource(show_spinner=False)
def train_segment_model(segment_col, segment_val):
    """Trains an XGBoost model on lag features for the given segment
    (or overall sales if segment_col is None). Returns everything the
    UI needs: the historical series, the fitted model, backtest MAE/RMSE,
    and the resulting future forecast."""

    series = build_monthly_series(segment_col, segment_val)

    feat_df = pd.DataFrame({'Sales': series})
    feat_df['Lag_1'] = feat_df['Sales'].shift(1)
    feat_df['Lag_2'] = feat_df['Sales'].shift(2)
    feat_df['Lag_3'] = feat_df['Sales'].shift(3)
    feat_df['Rolling_Mean_3'] = feat_df['Sales'].rolling(3).mean()
    feat_df['Month'] = feat_df.index.month
    feat_df['Quarter'] = feat_df.index.quarter
    feat_df['Season'] = feat_df['Month'].map(SEASON_MAP)
    feat_df = feat_df.dropna()

    feature_cols = ['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Month', 'Quarter', 'Season']
    X = feat_df[feature_cols]
    y = feat_df['Sales']

    train_size = len(feat_df) - FORECAST_PERIODS
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        early_stopping_rounds=20
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    # Iteratively forecast FORECAST_PERIODS months beyond the full series
    last_sales = list(y.iloc[-3:].values)
    last_month = series.index[-1]
    future_dates = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=FORECAST_PERIODS, freq='MS')

    future_preds = []
    for i in range(FORECAST_PERIODS):
        rolling_mean = np.mean(last_sales[-3:])
        month = future_dates[i].month
        quarter = future_dates[i].quarter
        season = SEASON_MAP[month]
        X_next = pd.DataFrame([[last_sales[-1], last_sales[-2], last_sales[-3], rolling_mean, month, quarter, season]],
                               columns=feature_cols)
        pred = float(model.predict(X_next)[0])
        future_preds.append(pred)
        last_sales = last_sales[1:] + [pred]

    forecast_df = pd.DataFrame({'Month': future_dates, 'Forecast': future_preds})

    return {
        'series': series,
        'test_index': y_test.index,
        'test_actual': y_test.values,
        'test_pred': test_pred,
        'mae': mae,
        'rmse': rmse,
        'forecast': forecast_df,
    }


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["📊 Sales Overview", "🔮 Forecast Explorer", "⚠️ Anomaly Report", "📦 Demand Segments"]
)

# ============================================================
# PAGE 1: SALES OVERVIEW DASHBOARD
# ============================================================
if page == "📊 Sales Overview":
    st.markdown('<div class="main-header">📊 Sales Overview Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sales", f"${df['Sales'].sum()/1e6:.2f}M")
    with col2:
        st.metric("Total Orders", f"{len(df):,}")
    with col3:
        st.metric("Avg Order Value", f"${df['Sales'].mean():,.2f}")
    with col4:
        st.metric("Sub-Categories", df['Sub-Category'].nunique())

    # --- Total sales by year (bar chart) ---
    st.subheader("Total Sales by Year")
    yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(yearly_sales['Year'].astype(str), yearly_sales['Sales'],
                   color=COLOR1, alpha=0.85, edgecolor='black', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'${height/1e6:.2f}M',
                ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Monthly sales trend line chart ---
    st.subheader("Monthly Sales Trend")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly_sales['Month'], monthly_sales['Total_Sales'], linewidth=2.5,
            color=COLOR1, marker='o', markersize=4)
    ax.fill_between(monthly_sales['Month'], monthly_sales['Total_Sales'], alpha=0.2, color=COLOR1)
    ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Sales by region and category (interactive filters) ---
    st.subheader("Sales by Region and Category")

    col1, col2 = st.columns(2)
    with col1:
        selected_regions = st.multiselect("Filter by Region(s):", options=REGIONS, default=REGIONS)
    with col2:
        selected_categories = st.multiselect("Filter by Category(ies):", options=CATEGORIES, default=CATEGORIES)

    filtered_df = df[df['Region'].isin(selected_regions) & df['Category'].isin(selected_categories)]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sales by Region**")
            region_sales = filtered_df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            region_sales.plot(kind='barh', ax=ax, color=COLOR2, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Sales ($)', fontsize=11, fontweight='bold')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.write("**Sales by Category**")
            category_sales = filtered_df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            palette = [COLOR1, COLOR2, COLOR3][:len(category_sales)]
            ax.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%',
                   colors=palette, startangle=90)
            ax.set_title('Sales Distribution by Category', fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ============================================================
# PAGE 2: FORECAST EXPLORER
# ============================================================
elif page == "🔮 Forecast Explorer":
    st.markdown('<div class="main-header">🔮 Forecast Explorer</div>', unsafe_allow_html=True)
    st.write("Forecasts are generated live with **XGBoost** — the best-performing model "
             "from the notebook comparison (lowest MAE/RMSE/MAPE of SARIMA, Prophet, and XGBoost).")

    col1, col2 = st.columns([1, 1])
    with col1:
        segment_type = st.selectbox("Select Category or Region:", ["Overall", "Category", "Region"])

    segment_col, segment_val = None, None
    if segment_type == "Category":
        with col2:
            segment_val = st.selectbox("Choose Category:", CATEGORIES)
        segment_col = "Category"
    elif segment_type == "Region":
        with col2:
            segment_val = st.selectbox("Choose Region:", REGIONS)
        segment_col = "Region"

    horizon = st.slider("Forecast Horizon (months ahead):", min_value=1, max_value=3, value=3, step=1)

    with st.spinner("Training model for this selection..."):
        result = train_segment_model(segment_col, segment_val)

    series = result['series']
    forecast_df = result['forecast'].head(horizon)

    # --- Forecast chart ---
    label = "Overall Sales" if segment_type == "Overall" else f"{segment_val} ({segment_type})"
    st.subheader(f"Sales Forecast — {label}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series.index, series.values, label='Historical Sales', linewidth=2.5,
            color=COLOR1, marker='o', markersize=4)
    ax.plot(result['test_index'], result['test_pred'], 'x--', label='Backtest Prediction',
            linewidth=2, color=COLOR3, markersize=8)
    ax.plot(forecast_df['Month'], forecast_df['Forecast'], 's-', label=f'{horizon}-Month Forecast',
            linewidth=2.5, color=COLOR2, markersize=9)

    margin = forecast_df['Forecast'] * (result['mae'] / max(forecast_df['Forecast'].mean(), 1))
    ax.fill_between(forecast_df['Month'], forecast_df['Forecast'] - margin, forecast_df['Forecast'] + margin,
                     alpha=0.15, color=COLOR2)

    ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Forecast table ---
    st.subheader("Forecast Values")
    display_table = forecast_df.copy()
    display_table['Month'] = display_table['Month'].dt.strftime('%B %Y')
    display_table.columns = ['Month', 'Forecasted Sales']
    st.dataframe(display_table.style.format({'Forecasted Sales': '${:,.0f}'}), width='stretch')

    # --- MAE / RMSE ---
    st.subheader("Model Accuracy (Backtest on Last 3 Months)")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("MAE", f"${result['mae']:,.2f}")
    with m2:
        st.metric("RMSE", f"${result['rmse']:,.2f}")

    with st.expander("See all-model comparison from the notebook (overall sales)"):
        st.dataframe(
            model_comparison.style.format({'MAE': '${:,.0f}', 'RMSE': '${:,.0f}', 'MAPE': '{:.2%}'}),
            width='stretch'
        )

# ============================================================
# PAGE 3: ANOMALY REPORT
# ============================================================
elif page == "⚠️ Anomaly Report":
    st.markdown('<div class="main-header">⚠️ Anomaly Detection Report</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Anomalies Detected (Isolation Forest)", int(weekly_sales['Is_Anomaly_IF'].sum()))
    with col2:
        st.metric("Anomalies Detected (Z-Score)", int(weekly_sales['Is_Anomaly_Zscore'].sum()))

    st.subheader("Weekly Sales with Detected Anomalies")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(weekly_sales['Week'], weekly_sales['Total_Sales'], 'o-', label='Weekly Sales',
            linewidth=2, color=COLOR1, markersize=4)

    anomalies_if = weekly_sales[weekly_sales['Is_Anomaly_IF']]
    ax.scatter(anomalies_if['Week'], anomalies_if['Total_Sales'], color='red', s=150, marker='X',
               label='Isolation Forest Anomaly', zorder=5, edgecolors='darkred', linewidths=2)

    anomalies_z = weekly_sales[weekly_sales['Is_Anomaly_Zscore']]
    ax.scatter(anomalies_z['Week'], anomalies_z['Total_Sales'], color='orange', s=100, marker='*',
               label='Z-Score Anomaly', zorder=4)

    ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Detected Anomalies")
    anomalies_all = weekly_sales[weekly_sales['Is_Anomaly_IF'] | weekly_sales['Is_Anomaly_Zscore']].copy()
    anomalies_all = anomalies_all.sort_values('Week', ascending=False)
    anomalies_all['Method'] = anomalies_all.apply(
        lambda r: ' & '.join(filter(None, [
            'Isolation Forest' if r['Is_Anomaly_IF'] else '',
            'Z-Score' if r['Is_Anomaly_Zscore'] else ''
        ])), axis=1
    )
    display_anomalies = anomalies_all[['Week', 'Total_Sales', 'Method']].rename(
        columns={'Week': 'Date', 'Total_Sales': 'Sales ($)'}
    )
    st.dataframe(display_anomalies.style.format({'Sales ($)': '${:,.0f}'}), width='stretch')

    st.subheader("Possible Real-World Explanations")
    st.write("""
    - **November–December:** Holiday shopping season (Black Friday, Christmas) typically drives sales spikes.
    - **January:** Post-holiday slowdown often causes dips.
    - **Summer months:** Furniture/outdoor demand can spike seasonally.
    - **September–October:** Back-to-school and Q3 business purchasing can drive spikes.
    - Unexplained dips may reflect supply disruptions, weather events, or simply low-order weeks.
    """)

# ============================================================
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ============================================================
elif page == "📦 Demand Segments":
    st.markdown('<div class="main-header">📦 Product Demand Segmentation</div>', unsafe_allow_html=True)
    st.write("Sub-categories are grouped into demand clusters (K-Means) using total sales, "
             "year-over-year growth rate, sales volatility, and average order value.")

    num_clusters = cluster_data['Cluster'].nunique()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Clusters", int(num_clusters))
    with col2:
        st.metric("Total Sub-Categories", len(cluster_data))
    with col3:
        st.metric("Avg per Cluster", f"{len(cluster_data)/num_clusters:.1f}")

    st.subheader("Cluster Chart: Sales vs. Growth Rate")
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, cluster_id in enumerate(sorted(cluster_data['Cluster'].unique())):
        subset = cluster_data[cluster_data['Cluster'] == cluster_id]
        cluster_label = subset['Cluster_Label'].iloc[0] if 'Cluster_Label' in subset.columns else f"Cluster {cluster_id}"
        ax.scatter(subset['Total_Sales'], subset['Growth_Rate'], s=300, alpha=0.75,
                   c=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
                   label=f'Cluster {cluster_id}: {cluster_label}', edgecolors='black', linewidth=1.5)
        for _, row in subset.iterrows():
            ax.annotate(row['Sub_Category'], (row['Total_Sales'], row['Growth_Rate']),
                        fontsize=8, ha='center', va='center', fontweight='bold')

    ax.set_xlabel('Total Sales ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Product Sub-Categories by Demand Cluster', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sub-Categories by Demand Cluster")
    display_cols = ['Sub_Category', 'Cluster', 'Cluster_Label', 'Total_Sales', 'Growth_Rate',
                     'Sales_Volatility', 'Avg_Order_Value']
    display_cols = [c for c in display_cols if c in cluster_data.columns]
    table = cluster_data[display_cols].sort_values(['Cluster', 'Total_Sales'], ascending=[True, False])

    st.dataframe(
        table.style.format({
            'Total_Sales': '${:,.0f}',
            'Growth_Rate': '{:+.1f}%',
            'Sales_Volatility': '${:,.0f}',
            'Avg_Order_Value': '${:,.2f}'
        }),
        width='stretch'
    )

    st.subheader("Recommended Stocking Strategy by Cluster")
    strategy_map = {
        "High Volume, Stable Demand": "Maintain high, consistently replenished inventory. Negotiate volume discounts.",
        "Growing Demand": "Increase inventory ahead of demand. Monitor turnover closely and expand shelf space.",
        "Declining Demand": "Wind down inventory gradually. Use promotions to clear excess stock.",
        "Low Volume, High Volatility": "Keep minimal safety stock. Use just-in-time ordering.",
        "Low Volume, Niche Products": "Consider dropshipping or made-to-order fulfillment.",
        "Moderate, Stable Products": "Standard replenishment cycles with seasonal adjustments.",
    }
    for cluster_id in sorted(cluster_data['Cluster'].unique()):
        subset = cluster_data[cluster_data['Cluster'] == cluster_id]
        label = subset['Cluster_Label'].iloc[0] if 'Cluster_Label' in subset.columns else f"Cluster {cluster_id}"
        with st.expander(f"Cluster {cluster_id}: {label} ({len(subset)} sub-categories)"):
            st.write(f"**Sub-categories:** {', '.join(subset['Sub_Category'].tolist())}")
            st.info(strategy_map.get(label, "Monitor and adjust inventory based on observed trends."))

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Sales Forecasting System**")
st.sidebar.markdown("Model: XGBoost (best of SARIMA / Prophet / XGBoost)")
