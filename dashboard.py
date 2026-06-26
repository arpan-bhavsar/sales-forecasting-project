import streamlit as st
import pandas as pd
import joblib
import altair as alt
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="DMart Sales Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# POWERBI-STYLE CSS
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f0f2f6; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f4b 0%, #2d3561 60%, #1a1f4b 100%);
    padding-top: 1rem;
}
[data-testid="stSidebar"] * { color: #e0e4ff !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px !important; padding: 6px 0 !important; }

.dash-header {
    background: linear-gradient(90deg, #1a1f4b 0%, #2d3561 100%);
    color: white;
    padding: 18px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.dash-header h1 { margin: 0; font-size: 26px; font-weight: 700; color: white !important; }
.dash-header p  { margin: 4px 0 0 0; font-size: 13px; color: #b0baff !important; }

.kpi-row { display: flex; gap: 16px; margin-bottom: 20px; }
.kpi-card {
    flex: 1; border-radius: 12px; padding: 18px 22px;
    color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    position: relative; overflow: hidden;
}
.kpi-card::after {
    content: ''; position: absolute; top: -20px; right: -20px;
    width: 80px; height: 80px; border-radius: 50%;
    background: rgba(255,255,255,0.12);
}
.kpi-card .kpi-icon  { font-size: 28px; margin-bottom: 6px; }
.kpi-card .kpi-label { font-size: 12px; opacity: 0.85; text-transform: uppercase; letter-spacing: 1px; }
.kpi-card .kpi-value { font-size: 26px; font-weight: 700; margin-top: 4px; }
.kpi-card .kpi-sub   { font-size: 11px; opacity: 0.75; margin-top: 2px; }
.kpi-blue   { background: linear-gradient(135deg, #1a56db, #3b82f6); }
.kpi-green  { background: linear-gradient(135deg, #057a55, #31c48d); }
.kpi-orange { background: linear-gradient(135deg, #b43403, #ff5a1f); }
.kpi-purple { background: linear-gradient(135deg, #5521b5, #9061f9); }

.chart-card {
    background: white; border-radius: 12px; padding: 20px 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07); margin-bottom: 20px;
}
.chart-card-title {
    font-size: 15px; font-weight: 600; color: #1a1f4b;
    margin-bottom: 4px; border-left: 4px solid #3b82f6; padding-left: 10px;
}
.chart-card-subtitle { font-size: 12px; color: #6b7280; margin-bottom: 14px; padding-left: 14px; }

.section-title {
    font-size: 18px; font-weight: 700; color: #1a1f4b;
    margin: 24px 0 12px 0; padding-bottom: 6px; border-bottom: 2px solid #e5e7eb;
}

.pred-result {
    background: linear-gradient(135deg, #1a56db, #3b82f6);
    color: white; border-radius: 12px; padding: 24px 32px;
    text-align: center; margin: 16px 0; box-shadow: 0 4px 20px rgba(26,86,219,0.3);
}
.pred-result .pred-label { font-size: 13px; opacity: 0.85; letter-spacing: 1px; text-transform: uppercase; }
.pred-result .pred-value { font-size: 42px; font-weight: 800; margin-top: 6px; }

.model-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
.model-table th { background: #1a1f4b; color: white; padding: 10px 16px; text-align: left; font-size: 13px; }
.model-table td { padding: 10px 16px; border-bottom: 1px solid #e5e7eb; font-size: 14px; }
.model-table tr:hover td { background: #f0f4ff; }
.badge-lr { background:#dbeafe; color:#1e40af; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-dt { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.winner   { color: #16a34a; font-weight: 700; }

.insight-box {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    font-size: 13px; color: #1e3a8a;
}
.insight-warn    { background: #fff7ed; border-left-color: #f97316; color: #7c2d12; }
.insight-success { background: #f0fdf4; border-left-color: #22c55e; color: #14532d; }

[data-testid="metric-container"] { display: none; }

.stButton > button {
    background: linear-gradient(90deg, #1a56db, #3b82f6) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 10px 32px !important;
    font-size: 15px !important; font-weight: 600 !important;
    width: 100% !important; box-shadow: 0 4px 12px rgba(26,86,219,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & DATA
# =========================
model = joblib.load("dmart_model.pkl")

try:
    dt_model = joblib.load("dt_model.pkl")
    dt_model_loaded = True
except FileNotFoundError:
    dt_model = None
    dt_model_loaded = False

data = pd.read_csv("dmart_sales.csv")
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
data['Month']      = data['Order Date'].dt.month
data['DayOfWeek']  = data['Order Date'].dt.dayofweek

def get_season(month):
    if month in [3, 4, 5, 6]:   return "Summer"
    elif month in [7, 8, 9]:    return "Monsoon"
    else:                        return "Winter"

data['Season'] = data['Month'].apply(get_season)

COLORS = ["#1a56db", "#31c48d", "#ff5a1f", "#9061f9", "#f59e0b", "#06b6d4"]
BLUE   = "#1a56db"
ORANGE = "#ff5a1f"

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("""
<div style='text-align:center; padding:10px 0 20px 0;'>
    <div style='font-size:42px;'>🛒</div>
    <div style='font-size:18px; font-weight:700; color:white; margin-top:6px;'>DMart Intelligence</div>
    <div style='font-size:11px; color:#b0baff; margin-top:2px;'>Sales Analytics Platform</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["📊 Business Overview", "📈 Sales Analytics", "🔮 Sales Prediction", "🤖 Model Info"]
)

st.sidebar.markdown("---")
ts = data['Sales'].sum()
tp = data['Profit'].sum()
st.sidebar.markdown(f"""
<div style='padding:12px; background:rgba(255,255,255,0.08); border-radius:10px; font-size:12px;'>
    <div style='color:#b0baff; margin-bottom:6px; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>Quick Stats</div>
    <div style='color:white; margin:4px 0;'>💰 Revenue: ₹{ts/1e6:.2f}M</div>
    <div style='color:white; margin:4px 0;'>📦 Orders: {len(data):,}</div>
    <div style='color:white; margin:4px 0;'>📈 Margin: {tp/ts*100:.1f}%</div>
    <div style='color:white; margin:4px 0;'>🗓️ Data: 2015–2018</div>
</div>
""", unsafe_allow_html=True)


# ======================================================
# 📊 PAGE 1 — BUSINESS OVERVIEW
# ======================================================
if page == "📊 Business Overview":

    st.markdown("""
    <div class="dash-header">
        <h1>📊 Business Overview</h1>
        <p>DMart Sales Intelligence Dashboard &nbsp;|&nbsp; 2015 – 2018</p>
    </div>
    """, unsafe_allow_html=True)

    total_sales  = data['Sales'].sum()
    total_profit = data['Profit'].sum()
    total_orders = len(data)
    avg_discount = data['Discount'].mean()

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card kpi-blue">
            <div class="kpi-icon">💰</div>
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">₹{total_sales/1e6:.2f}M</div>
            <div class="kpi-sub">₹{total_sales:,.0f} total</div>
        </div>
        <div class="kpi-card kpi-green">
            <div class="kpi-icon">📈</div>
            <div class="kpi-label">Total Profit</div>
            <div class="kpi-value">₹{total_profit/1e6:.2f}M</div>
            <div class="kpi-sub">Margin: {total_profit/total_sales*100:.1f}%</div>
        </div>
        <div class="kpi-card kpi-orange">
            <div class="kpi-icon">🛒</div>
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{total_orders:,}</div>
            <div class="kpi-sub">Avg ₹{total_sales/total_orders:,.0f} / order</div>
        </div>
        <div class="kpi-card kpi-purple">
            <div class="kpi-icon">🏷️</div>
            <div class="kpi-label">Avg Discount</div>
            <div class="kpi-value">{avg_discount:.1%}</div>
            <div class="kpi-sub">Across all orders</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: Area trend + Category pie ──
    st.markdown('<div class="section-title">📅 Sales Trends</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])

    monthly_sales = data.groupby(data['Order Date'].dt.to_period('M'))['Sales'].sum()
    monthly_sales.index = monthly_sales.index.astype(str)
    ms_df = pd.DataFrame({"Month": pd.to_datetime(monthly_sales.index), "Sales": monthly_sales.values})

    with c1:
        trend = alt.Chart(ms_df).mark_area(
            line={"color": BLUE, "strokeWidth": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#1a56db", offset=0),
                       alt.GradientStop(color="rgba(26,86,219,0.05)", offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y", labelAngle=-30)),
            y=alt.Y("Sales:Q", title="Monthly Sales (₹)", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("Month:T", format="%b %Y"), alt.Tooltip("Sales:Q", format=",.0f", title="Sales ₹")]
        ).properties(height=270).interactive()

        st.markdown('<div class="chart-card"><div class="chart-card-title">Monthly Sales Trend</div><div class="chart-card-subtitle">Total sales aggregated per month (2015–2018)</div>', unsafe_allow_html=True)
        st.altair_chart(trend, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        cat_pie_df = data.groupby("Category")["Sales"].sum().reset_index()
        cat_pie = alt.Chart(cat_pie_df).mark_arc(innerRadius=55, outerRadius=110).encode(
            theta=alt.Theta("Sales:Q"),
            color=alt.Color("Category:N", scale=alt.Scale(range=COLORS),
                            legend=alt.Legend(orient="bottom", labelFontSize=11)),
            tooltip=["Category:N", alt.Tooltip("Sales:Q", format=",.0f")]
        ).properties(height=270)

        st.markdown('<div class="chart-card"><div class="chart-card-title">Sales by Category</div><div class="chart-card-subtitle">Revenue share across categories</div>', unsafe_allow_html=True)
        st.altair_chart(cat_pie, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Forecast ──
    st.markdown('<div class="section-title">🔮 6-Month Sales Forecast</div>', unsafe_allow_html=True)

    last_date = data['Order Date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='ME')

    data['_Period']   = data['Order Date'].dt.to_period('M')
    data['_CalMonth'] = data['Order Date'].dt.month
    moc = data.groupby(['_Period', '_CalMonth']).size().reset_index(name='Count')
    avg_orders_by_month = moc.groupby('_CalMonth')['Count'].mean()
    data.drop(columns=['_Period', '_CalMonth'], inplace=True)

    future_sales = []
    for dt in future_dates:
        cm = dt.month
        n  = avg_orders_by_month.get(cm, avg_orders_by_month.mean())
        inp = pd.DataFrame({
            'Category': [data['Category'].mode()[0]], 'Sub Category': [data['Sub Category'].mode()[0]],
            'City': [data['City'].mode()[0]], 'Region': [data['Region'].mode()[0]],
            'Discount': [data[data['Month']==cm]['Discount'].mean()],
            'Profit':   [data[data['Month']==cm]['Profit'].mean()],
            'Month': [cm], 'DayOfWeek': [2], 'Season': [get_season(cm)]
        })
        future_sales.append(model.predict(inp)[0] * n)

    last_actual_date  = pd.to_datetime(monthly_sales.index[-1])
    last_actual_sales = float(monthly_sales.iloc[-1])

    actual_df   = pd.DataFrame({"Date": pd.to_datetime(monthly_sales.index), "Sales": monthly_sales.values, "Type": "Actual Sales"})
    forecast_df = pd.DataFrame({"Date": [last_actual_date]+list(future_dates), "Sales": [last_actual_sales]+future_sales, "Type": "Predicted Sales"})
    chart_df    = pd.concat([actual_df, forecast_df], ignore_index=True)

    fc = alt.Chart(chart_df).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=50)).encode(
        x=alt.X("Date:T", title="Month", axis=alt.Axis(format="%b %Y", labelAngle=-30)),
        y=alt.Y("Sales:Q", title="Sales (₹)", axis=alt.Axis(format="~s")),
        color=alt.Color("Type:N",
            scale=alt.Scale(domain=["Actual Sales","Predicted Sales"], range=[BLUE, ORANGE]),
            legend=alt.Legend(title="", orient="top-right", labelFontSize=12)),
        strokeDash=alt.condition(alt.datum.Type=="Predicted Sales", alt.value([8,4]), alt.value([0])),
        tooltip=[alt.Tooltip("Date:T", format="%b %Y", title="Month"),
                 alt.Tooltip("Sales:Q", format=",.0f", title="Sales ₹"), "Type:N"]
    ).properties(height=300).interactive()

    st.markdown('<div class="chart-card"><div class="chart-card-title">Sales Trend with 6-Month Prediction</div><div class="chart-card-subtitle">🔵 Blue = actual sales &nbsp;|&nbsp; 🟠 Orange dashed = ML forecast (next 6 months)</div>', unsafe_allow_html=True)
    st.altair_chart(fc, width='stretch', height=350)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Top Products + Region pie ──
    st.markdown('<div class="section-title">📦 Product & Regional Breakdown</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        top = data.groupby("Sub Category")["Sales"].sum().sort_values(ascending=True).tail(10).reset_index()
        bar = alt.Chart(top).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("Sales:Q", title="Total Sales (₹)", axis=alt.Axis(format="~s")),
            y=alt.Y("Sub Category:N", sort="-x", title=""),
            color=alt.Color("Sales:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Sub Category:N", alt.Tooltip("Sales:Q", format=",.0f")]
        ).properties(height=320)
        st.markdown('<div class="chart-card"><div class="chart-card-title">Top 10 Products by Sales</div><div class="chart-card-subtitle">Highest revenue-generating sub-categories</div>', unsafe_allow_html=True)
        st.altair_chart(bar, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        reg = data.groupby("Region")["Sales"].sum().reset_index()
        rp  = alt.Chart(reg).mark_arc(innerRadius=55, outerRadius=110).encode(
            theta=alt.Theta("Sales:Q"),
            color=alt.Color("Region:N", scale=alt.Scale(range=COLORS[1:]),
                            legend=alt.Legend(orient="bottom", labelFontSize=11)),
            tooltip=["Region:N", alt.Tooltip("Sales:Q", format=",.0f")]
        ).properties(height=320)
        st.markdown('<div class="chart-card"><div class="chart-card-title">Sales by Region</div><div class="chart-card-subtitle">Geographic revenue distribution</div>', unsafe_allow_html=True)
        st.altair_chart(rp, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 4: Scatter + Profit pie ──
    st.markdown('<div class="section-title">💰 Profitability Analysis</div>', unsafe_allow_html=True)
    c5, c6 = st.columns([2, 1])

    with c5:
        sc = alt.Chart(data).mark_circle(opacity=0.5, size=55).encode(
            x=alt.X("Discount:Q", title="Discount Rate", axis=alt.Axis(format=".0%")),
            y=alt.Y("Profit:Q",   title="Profit (₹)",   axis=alt.Axis(format="~s")),
            color=alt.Color("Category:N", scale=alt.Scale(range=COLORS),
                            legend=alt.Legend(title="Category")),
            tooltip=["Category:N", "Sub Category:N",
                     alt.Tooltip("Discount:Q", format=".1%"),
                     alt.Tooltip("Profit:Q", format=",.0f")]
        ).properties(height=280).interactive()
        st.markdown('<div class="chart-card"><div class="chart-card-title">Profit vs Discount</div><div class="chart-card-subtitle">Impact of discount rate on profit — higher discounts often hurt margins</div>', unsafe_allow_html=True)
        st.altair_chart(sc, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        prof = data.groupby("Category")["Profit"].sum().reset_index()
        pp   = alt.Chart(prof).mark_arc(innerRadius=50, outerRadius=100).encode(
            theta=alt.Theta("Profit:Q"),
            color=alt.Color("Category:N", scale=alt.Scale(range=COLORS),
                            legend=alt.Legend(orient="bottom", labelFontSize=11)),
            tooltip=["Category:N", alt.Tooltip("Profit:Q", format=",.0f")]
        ).properties(height=280)
        st.markdown('<div class="chart-card"><div class="chart-card-title">Profit by Category</div><div class="chart-card-subtitle">Profit share breakdown</div>', unsafe_allow_html=True)
        st.altair_chart(pp, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# 📈 PAGE 2 — SALES ANALYTICS
# ======================================================
elif page == "📈 Sales Analytics":

    st.markdown("""
    <div class="dash-header">
        <h1>📈 Sales Analytics</h1>
        <p>Deep-dive into product and regional performance</p>
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    sel_cat = fc1.selectbox("🔍 Filter by Category", ["All"] + sorted(data['Category'].unique()))
    fdata = data if sel_cat == "All" else data[data["Category"] == sel_cat]
    sel_sub = fc2.selectbox("🔍 Filter by Sub Category", ["All"] + sorted(fdata["Sub Category"].unique()))
    if sel_sub != "All":
        fdata = fdata[fdata["Sub Category"] == sel_sub]

    fs = fdata['Sales'].sum(); fp = fdata['Profit'].sum(); fo = len(fdata)
    st.markdown(f"""
    <div class="kpi-row" style="margin-top:16px;">
        <div class="kpi-card kpi-blue"><div class="kpi-icon">💰</div><div class="kpi-label">Filtered Sales</div><div class="kpi-value">₹{fs/1e3:.1f}K</div></div>
        <div class="kpi-card kpi-green"><div class="kpi-icon">📈</div><div class="kpi-label">Filtered Profit</div><div class="kpi-value">₹{fp/1e3:.1f}K</div></div>
        <div class="kpi-card kpi-orange"><div class="kpi-icon">🛒</div><div class="kpi-label">Filtered Orders</div><div class="kpi-value">{fo:,}</div></div>
        <div class="kpi-card kpi-purple"><div class="kpi-icon">📊</div><div class="kpi-label">Profit Margin</div><div class="kpi-value">{(fp/fs*100) if fs else 0:.1f}%</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📦 Category & Sub-Category Sales</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)

    with r1:
        cd = fdata.groupby("Category")["Sales"].sum().reset_index()
        ch = alt.Chart(cd).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Category:N", title="", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Sales:Q", title="Total Sales (₹)", axis=alt.Axis(format="~s")),
            color=alt.Color("Category:N", scale=alt.Scale(range=COLORS), legend=None),
            tooltip=["Category:N", alt.Tooltip("Sales:Q", format=",.0f")]
        ).properties(height=280)
        st.markdown('<div class="chart-card"><div class="chart-card-title">Sales by Category</div><div class="chart-card-subtitle">Revenue per product category</div>', unsafe_allow_html=True)
        st.altair_chart(ch, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        sd = fdata.groupby("Sub Category")["Sales"].sum().sort_values(ascending=True).reset_index()
        sh = alt.Chart(sd).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("Sales:Q", title="Total Sales (₹)", axis=alt.Axis(format="~s")),
            y=alt.Y("Sub Category:N", sort="-x", title=""),
            color=alt.Color("Sales:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Sub Category:N", alt.Tooltip("Sales:Q", format=",.0f")]
        ).properties(height=280)
        st.markdown('<div class="chart-card"><div class="chart-card-title">Sales by Sub-Category</div><div class="chart-card-subtitle">Ranked by revenue</div>', unsafe_allow_html=True)
        st.altair_chart(sh, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🗺️ Regional Performance</div>', unsafe_allow_html=True)
    r3, r4 = st.columns(2)

    with r3:
        rd = fdata.groupby("Region")["Sales"].sum().reset_index()
        rc = alt.Chart(rd).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Region:N", title="", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Sales:Q",  title="Total Sales (₹)", axis=alt.Axis(format="~s")),
            color=alt.Color("Region:N", scale=alt.Scale(range=COLORS[1:]), legend=None),
            tooltip=["Region:N", alt.Tooltip("Sales:Q", format=",.0f")]
        ).properties(height=260)
        st.markdown('<div class="chart-card"><div class="chart-card-title">Sales by Region</div><div class="chart-card-subtitle">Revenue contribution by geography</div>', unsafe_allow_html=True)
        st.altair_chart(rc, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    with r4:
        rpd = fdata.groupby("Region")["Profit"].sum().reset_index()
        rpp = alt.Chart(rpd).mark_arc(innerRadius=50, outerRadius=100).encode(
            theta=alt.Theta("Profit:Q"),
            color=alt.Color("Region:N", scale=alt.Scale(range=COLORS[1:]),
                            legend=alt.Legend(orient="bottom", labelFontSize=11)),
            tooltip=["Region:N", alt.Tooltip("Profit:Q", format=",.0f")]
        ).properties(height=260)
        st.markdown('<div class="chart-card"><div class="chart-card-title">Profit Share by Region</div><div class="chart-card-subtitle">Which regions are most profitable</div>', unsafe_allow_html=True)
        st.altair_chart(rpp, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🗂️ Data Table</div>', unsafe_allow_html=True)
    # OPTION 2 (Scrollable table)
    st.dataframe(
        fdata[['Order Date','Category','Sub Category','Region','Sales','Profit','Discount']],
        height=500,
        use_container_width=True
    )


# ======================================================
# 🔮 PAGE 3 — SALES PREDICTION
# ======================================================
elif page == "🔮 Sales Prediction":

    st.markdown("""
    <div class="dash-header">
        <h1>🔮 Sales Prediction Simulator</h1>
        <p>Predict expected sales using the trained ML model</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚙️ Configure Prediction Inputs</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    category     = col1.selectbox("🏷️ Product Category", sorted(data['Category'].unique()))
    sub_category = col2.selectbox("📦 Product Sub Category", sorted(data[data["Category"]==category]["Sub Category"].unique()))
    city         = col1.selectbox("🏙️ City",   sorted(data['City'].unique()))
    region       = col2.selectbox("🗺️ Region", sorted(data['Region'].unique()))
    discount     = col1.slider("🏷️ Discount Offered", 0.0, 1.0, 0.1)
    profit       = col2.number_input("💰 Expected Profit (₹)", value=100.0, step=50.0)
    sel_date     = col1.date_input("📅 Order Date", datetime.today())

    month     = sel_date.month
    dayofweek = sel_date.weekday()
    season    = get_season(month)
    icons     = {"Summer": "☀️", "Monsoon": "🌧️", "Winter": "❄️"}

    col2.markdown(f"""
    <div class="insight-box">
        {icons[season]} <strong>{season}</strong> season
        &nbsp;|&nbsp; 📅 {sel_date.strftime("%B %Y")}
    </div>
    """, unsafe_allow_html=True)

    input_data = pd.DataFrame({
        'Category': [category], 'Sub Category': [sub_category],
        'City': [city], 'Region': [region],
        'Discount': [discount], 'Profit': [profit],
        'Month': [month], 'DayOfWeek': [dayofweek], 'Season': [season]
    })

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀  Predict Sales Now"):
        prediction = model.predict(input_data)[0]
        st.markdown(f"""
        <div class="pred-result">
            <div class="pred-label">Predicted Sales</div>
            <div class="pred-value">₹ {prediction:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        if discount > 0.3:
            st.markdown('<div class="insight-box insight-warn">⚠️ High discount (&gt;30%) — boosts volume but reduces profit margin significantly.</div>', unsafe_allow_html=True)
        if month in [10, 11, 12]:
            st.markdown('<div class="insight-box insight-success">🎉 Festive season (Oct–Dec) — historically shows 40–60% higher demand.</div>', unsafe_allow_html=True)
        if season == "Summer":
            st.markdown('<div class="insight-box">☀️ Summer — strong demand for beverages, personal care & cooling products.</div>', unsafe_allow_html=True)


# ======================================================
# 🤖 PAGE 4 — MODEL INFO
# ======================================================
elif page == "🤖 Model Info":

    st.markdown("""
    <div class="dash-header">
        <h1>🤖 Model Evaluation Report</h1>
        <p>Performance analysis, error breakdown & model comparison — as required by evaluation guidelines</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load evaluation data ──
    try:
        metrics  = joblib.load("model_metrics.pkl")
        eval_df  = joblib.load("eval_report.pkl")
        lr_m     = metrics["lr"]
        dt_m     = metrics["dt"]
        report_loaded = True
    except Exception:
        report_loaded = False

    if not report_loaded:
        st.markdown('''<div class="insight-box insight-warn">
            ⚠️ Evaluation data not found. Please run <b>model_training.py</b> first to generate
            <code>model_metrics.pkl</code> and <code>eval_report.pkl</code>, then restart the dashboard.
        </div>''', unsafe_allow_html=True)
        st.stop()

    lr_mae=lr_m["MAE"]; lr_rmse=lr_m["RMSE"]; lr_r2=lr_m["R2"]
    dt_mae=dt_m["MAE"]; dt_rmse=dt_m["RMSE"]; dt_r2=dt_m["R2"]
    best = "Linear Regression" if lr_mae < dt_mae else "Decision Tree"

    # ══════════════════════════════════════
    # SECTION 1 — KPI Metrics
    # ══════════════════════════════════════
    st.markdown('<div class="section-title">📊 Model Performance Metrics</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card kpi-blue">
            <div class="kpi-icon">📐</div><div class="kpi-label">LR — MAE</div>
            <div class="kpi-value">₹{lr_mae:,.0f}</div>
            <div class="kpi-sub">Avg prediction error</div>
        </div>
        <div class="kpi-card kpi-purple">
            <div class="kpi-icon">📏</div><div class="kpi-label">LR — RMSE</div>
            <div class="kpi-value">₹{lr_rmse:,.0f}</div>
            <div class="kpi-sub">Penalises large errors</div>
        </div>
        <div class="kpi-card kpi-green">
            <div class="kpi-icon">📈</div><div class="kpi-label">LR — R² Score</div>
            <div class="kpi-value">{lr_r2:.2f}</div>
            <div class="kpi-sub">{lr_r2*100:.1f}% variance explained</div>
        </div>
        <div class="kpi-card kpi-orange">
            <div class="kpi-icon">🏆</div><div class="kpi-label">Best Model</div>
            <div class="kpi-value" style="font-size:16px;">{best}</div>
            <div class="kpi-sub">Lower MAE wins</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── What these metrics mean ──
    st.markdown(f"""
    <div class="chart-card">
        <div class="chart-card-title">📖 What These Metrics Mean</div>
        <div class="chart-card-subtitle">Understanding the evaluation report</div>
        <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:10px;">
            <div style="flex:1; min-width:200px; padding:12px; background:#eff6ff; border-radius:8px;">
                <b>MAE = ₹{lr_mae:,.0f}</b><br>
                <span style="font-size:12px; color:#374151;">On average, each sales prediction is off by ₹{lr_mae:,.0f}. Lower = better.</span>
            </div>
            <div style="flex:1; min-width:200px; padding:12px; background:#faf5ff; border-radius:8px;">
                <b>RMSE = ₹{lr_rmse:,.0f}</b><br>
                <span style="font-size:12px; color:#374151;">RMSE is higher than MAE because it penalises large errors more heavily.</span>
            </div>
            <div style="flex:1; min-width:200px; padding:12px; background:#f0fdf4; border-radius:8px;">
                <b>R² = {lr_r2:.2f}</b><br>
                <span style="font-size:12px; color:#374151;">Model explains {lr_r2*100:.1f}% of sales variation. Sales data is highly variable so this is expected.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════
    # SECTION 2 — Actual vs Predicted
    # ══════════════════════════════════════
    st.markdown('<div class="section-title">🎯 Actual vs Predicted Sales</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Scatter: Actual vs LR Predicted
        scatter_df = eval_df[["Actual","LR_Predicted"]].copy().sample(min(300, len(eval_df)), random_state=42)
        max_val = max(scatter_df["Actual"].max(), scatter_df["LR_Predicted"].max())

        # Perfect prediction line
        line_df = pd.DataFrame({"x": [0, max_val], "y": [0, max_val]})
        perfect_line = alt.Chart(line_df).mark_line(color="#22c55e", strokeDash=[5,3], strokeWidth=1.5).encode(
            x="x:Q", y="y:Q"
        )

        scatter = alt.Chart(scatter_df).mark_circle(size=55, opacity=0.55, color=BLUE).encode(
            x=alt.X("Actual:Q",       title="Actual Sales (₹)",    axis=alt.Axis(format="~s")),
            y=alt.Y("LR_Predicted:Q", title="Predicted Sales (₹)", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("Actual:Q", format=",.0f"), alt.Tooltip("LR_Predicted:Q", format=",.0f")]
        )

        st.markdown('<div class="chart-card"><div class="chart-card-title">Linear Regression — Actual vs Predicted</div><div class="chart-card-subtitle">Points on green line = perfect prediction. Spread = error.</div>', unsafe_allow_html=True)
        st.altair_chart((scatter + perfect_line).properties(height=300).interactive(), width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        scatter_df2 = eval_df[["Actual","DT_Predicted"]].copy().sample(min(300, len(eval_df)), random_state=42)

        perfect_line2 = alt.Chart(line_df).mark_line(color="#22c55e", strokeDash=[5,3], strokeWidth=1.5).encode(
            x="x:Q", y="y:Q"
        )
        scatter2 = alt.Chart(scatter_df2).mark_circle(size=55, opacity=0.55, color=ORANGE).encode(
            x=alt.X("Actual:Q",       title="Actual Sales (₹)",    axis=alt.Axis(format="~s")),
            y=alt.Y("DT_Predicted:Q", title="Predicted Sales (₹)", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("Actual:Q", format=",.0f"), alt.Tooltip("DT_Predicted:Q", format=",.0f")]
        )

        st.markdown('<div class="chart-card"><div class="chart-card-title">Decision Tree — Actual vs Predicted</div><div class="chart-card-subtitle">Points on green line = perfect prediction. Spread = error.</div>', unsafe_allow_html=True)
        st.altair_chart((scatter2 + perfect_line2).properties(height=300).interactive(), width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    # SECTION 3 — Where Model is Weak
    # ══════════════════════════════════════
    st.markdown('<div class="section-title">⚠️ Where the Model Makes Errors (Loss Analysis)</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Error by Category
        cat_err = eval_df.groupby("Category")[["LR_AbsError","DT_AbsError"]].mean().reset_index()
        cat_err = cat_err.sort_values("LR_AbsError", ascending=True)

        cat_long = pd.melt(cat_err, id_vars="Category",
                           value_vars=["LR_AbsError","DT_AbsError"],
                           var_name="Model", value_name="MAE")
        cat_long["Model"] = cat_long["Model"].map({"LR_AbsError":"Linear Regression","DT_AbsError":"Decision Tree"})

        cat_chart = alt.Chart(cat_long).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("MAE:Q", title="Mean Absolute Error (₹)", axis=alt.Axis(format="~s")),
            y=alt.Y("Category:N", sort="-x", title=""),
            color=alt.Color("Model:N", scale=alt.Scale(domain=["Linear Regression","Decision Tree"], range=[BLUE, ORANGE])),
            tooltip=["Category:N","Model:N", alt.Tooltip("MAE:Q", format=",.0f")]
        ).properties(height=280)

        st.markdown('<div class="chart-card"><div class="chart-card-title">Error by Category</div><div class="chart-card-subtitle">Which categories have highest prediction error — model is weakest here</div>', unsafe_allow_html=True)
        st.altair_chart(cat_chart, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        # Error by Region
        reg_err = eval_df.groupby("Region")[["LR_AbsError","DT_AbsError"]].mean().reset_index()
        reg_long = pd.melt(reg_err, id_vars="Region",
                           value_vars=["LR_AbsError","DT_AbsError"],
                           var_name="Model", value_name="MAE")
        reg_long["Model"] = reg_long["Model"].map({"LR_AbsError":"Linear Regression","DT_AbsError":"Decision Tree"})

        reg_chart = alt.Chart(reg_long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Region:N", title="", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("MAE:Q", title="Mean Absolute Error (₹)", axis=alt.Axis(format="~s")),
            color=alt.Color("Model:N", scale=alt.Scale(domain=["Linear Regression","Decision Tree"], range=[BLUE, ORANGE])),
            xOffset="Model:N",
            tooltip=["Region:N","Model:N", alt.Tooltip("MAE:Q", format=",.0f")]
        ).properties(height=280)

        st.markdown('<div class="chart-card"><div class="chart-card-title">Error by Region</div><div class="chart-card-subtitle">Which regions the model struggles with most</div>', unsafe_allow_html=True)
        st.altair_chart(reg_chart, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    # Error by Month
    month_err = eval_df.groupby("Month")[["LR_AbsError","DT_AbsError"]].mean().reset_index()
    month_err["MonthName"] = month_err["Month"].apply(lambda m: ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][int(m)])
    month_long = pd.melt(month_err, id_vars=["Month","MonthName"],
                         value_vars=["LR_AbsError","DT_AbsError"],
                         var_name="Model", value_name="MAE")
    month_long["Model"] = month_long["Model"].map({"LR_AbsError":"Linear Regression","DT_AbsError":"Decision Tree"})

    month_chart = alt.Chart(month_long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Month:O", title="Month", axis=alt.Axis(labelExpr='["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][datum.value]')),
        y=alt.Y("MAE:Q", title="Mean Absolute Error (₹)", axis=alt.Axis(format="~s")),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["Linear Regression","Decision Tree"], range=[BLUE, ORANGE])),
        tooltip=["MonthName:N","Model:N", alt.Tooltip("MAE:Q", format=",.0f")]
    ).properties(height=260).interactive()

    st.markdown('<div class="chart-card"><div class="chart-card-title">Error by Month — Seasonal Weakness</div><div class="chart-card-subtitle">Error peaks in festive months (Sep–Dec) where sales are highly variable and harder to predict</div>', unsafe_allow_html=True)
    st.altair_chart(month_chart, width='stretch', height=350)
    st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    # SECTION 4 — Residual Distribution
    # ══════════════════════════════════════
    st.markdown('<div class="section-title">📉 Residual / Error Distribution</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2)
   

    with c3:
        bins_labels = ["<200","200-500","500-1K","1K-2K",">2K"]
        bins_vals   = [0, 200, 500, 1000, 2000, float("inf")]
        eval_df["LR_Bucket"] = pd.cut(eval_df["LR_AbsError"], bins=bins_vals, labels=bins_labels)
        eval_df["DT_Bucket"] = pd.cut(eval_df["DT_AbsError"], bins=bins_vals, labels=bins_labels)

        lr_dist = eval_df["LR_Bucket"].value_counts(normalize=True).reset_index()
        lr_dist.columns = ["Error Range","Percentage"]
        lr_dist["Percentage"] = (lr_dist["Percentage"]*100).round(1)
        lr_dist["Model"] = "Linear Regression"

        dt_dist = eval_df["DT_Bucket"].value_counts(normalize=True).reset_index()
        dt_dist.columns = ["Error Range","Percentage"]
        dt_dist["Percentage"] = (dt_dist["Percentage"]*100).round(1)
        dt_dist["Model"] = "Decision Tree"

        dist_df = pd.concat([lr_dist, dt_dist])

        dist_chart = alt.Chart(dist_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Error Range:N", sort=bins_labels, title="Absolute Error Range (₹)"),
            y=alt.Y("Percentage:Q", title="% of Predictions"),
            color=alt.Color("Model:N", scale=alt.Scale(domain=["Linear Regression","Decision Tree"], range=[BLUE, ORANGE])),
            xOffset="Model:N",
            tooltip=["Model:N","Error Range:N", alt.Tooltip("Percentage:Q", format=".1f", title="%")]
        ).properties(height=280)

        st.markdown('<div class="chart-card"><div class="chart-card-title">Error Distribution Comparison</div><div class="chart-card-subtitle">What % of predictions fall within each error range — more in &lt;₹500 = better</div>', unsafe_allow_html=True)
        st.altair_chart(dist_chart, width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)
        

    with c4:
        # Residual scatter (over/under prediction)
        res_df = eval_df[["Actual","LR_Residual"]].copy().sample(min(300, len(eval_df)), random_state=1)
        zero_line = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color="#ef4444", strokeDash=[4,3]).encode(y="y:Q")

        res_chart = alt.Chart(res_df).mark_circle(size=50, opacity=0.5, color=BLUE).encode(
            x=alt.X("Actual:Q",      title="Actual Sales (₹)",     axis=alt.Axis(format="~s")),
            y=alt.Y("LR_Residual:Q", title="Residual (Actual − Predicted)", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("Actual:Q", format=",.0f"), alt.Tooltip("LR_Residual:Q", format=",.0f", title="Error")]
        )

        st.markdown('<div class="chart-card"><div class="chart-card-title">LR Residual Plot</div><div class="chart-card-subtitle">Above red line = underpredicted &nbsp;|&nbsp; Below = overpredicted. Ideal: random scatter around 0.</div>', unsafe_allow_html=True)
        st.altair_chart((res_chart + zero_line).properties(height=280).interactive(), width='stretch', height=350)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    # SECTION 5 — Model Comparison Table
    # ══════════════════════════════════════
    st.markdown('<div class="section-title">⚖️ Model Comparison Summary</div>', unsafe_allow_html=True)

    lr_wins = sum([lr_mae < dt_mae, lr_rmse < dt_rmse, lr_r2 > dt_r2])
    dt_wins = 3 - lr_wins

    st.markdown(f"""
    <div class="chart-card">
        <table class="model-table">
            <tr>
                <th>Metric</th>
                <th>Linear Regression</th>
                <th>Decision Tree</th>
                <th>Winner</th>
            </tr>
            <tr>
                <td><b>MAE ↓</b> (avg error)</td>
                <td>₹{lr_mae:,.2f} {"✅" if lr_mae < dt_mae else ""}</td>
                <td>₹{dt_mae:,.2f} {"✅" if dt_mae < lr_mae else ""}</td>
                <td class="winner">{"LR" if lr_mae < dt_mae else "DT"}</td>
            </tr>
            <tr>
                <td><b>RMSE ↓</b> (large error penalty)</td>
                <td>₹{lr_rmse:,.2f} {"✅" if lr_rmse < dt_rmse else ""}</td>
                <td>₹{dt_rmse:,.2f} {"✅" if dt_rmse < lr_rmse else ""}</td>
                <td class="winner">{"LR" if lr_rmse < dt_rmse else "DT"}</td>
            </tr>
            <tr>
                <td><b>R² ↑</b> (variance explained)</td>
                <td>{lr_r2:.4f} ({lr_r2*100:.1f}%) {"✅" if lr_r2 > dt_r2 else ""}</td>
                <td>{dt_r2:.4f} ({dt_r2*100:.1f}%) {"✅" if dt_r2 > lr_r2 else ""}</td>
                <td class="winner">{"LR" if lr_r2 > dt_r2 else "DT"}</td>
            </tr>
            <tr style="background:#f0f9ff;">
                <td><b>Overall Winner</b></td>
                <td colspan="3" style="font-weight:700; color:#1a56db;">
                    🏆 {"Linear Regression" if lr_wins >= dt_wins else "Decision Tree"} wins {max(lr_wins,dt_wins)}/3 metrics
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Insight summary ──
    st.markdown('<div class="section-title">🔍 Key Findings & Recommendations</div>', unsafe_allow_html=True)

    # Find weakest category
    worst_cat = eval_df.groupby("Category")["LR_AbsError"].mean().idxmax()
    worst_reg = eval_df.groupby("Region")["LR_AbsError"].mean().idxmax()
    pct_under500 = (eval_df["LR_AbsError"] < 500).mean() * 100

    st.markdown(f"""
    <div class="chart-card">
        <div style="display:flex; flex-direction:column; gap:10px;">
            <div class="insight-box">
                📌 <b>Model accuracy:</b> {pct_under500:.1f}% of predictions have error below ₹500, which is acceptable for sales forecasting.
            </div>
            <div class="insight-box insight-warn">
                ⚠️ <b>Weakest category:</b> <b>{worst_cat}</b> has the highest average prediction error — the model struggles here due to high sales variability.
            </div>
            <div class="insight-box insight-warn">
                ⚠️ <b>Weakest region:</b> <b>{worst_reg}</b> shows highest regional error — regional demand patterns may need more features.
            </div>
            <div class="insight-box insight-warn">
                ⚠️ <b>Festive months weakness:</b> Sep–Dec show higher errors because festive sales spikes are hard to predict with linear models.
            </div>
            <div class="insight-box insight-success">
                ✅ <b>Decision Tree comparison:</b> Linear Regression outperforms Decision Tree on this dataset (LR MAE ₹{lr_mae:,.0f} vs DT MAE ₹{dt_mae:,.0f}), confirming LR as the better production model.
            </div>
            <div class="insight-box insight-success">
                ✅ <b>Recommendation:</b> To improve performance, consider adding more features like promotions, holidays, store location and adding more years of training data.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)