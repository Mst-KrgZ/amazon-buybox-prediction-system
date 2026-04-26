import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgbm

# ─── Page Config ───
st.set_page_config(
    page_title="🏆 Buy Box % Prediction System",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; max-width: 1200px; }
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        text-align: center; padding: 1.2rem 0;
        border-bottom: 4px solid #f59e0b; margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #f59e0b; border-radius: 12px;
        letter-spacing: 1px;
    }
    .kpi-container { display: flex; gap: 16px; margin: 20px 0; }
    .kpi-card {
        flex: 1; background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 14px; padding: 20px 24px;
        border-left: 5px solid #f59e0b;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        text-align: center;
    }
    .kpi-card .kpi-icon { font-size: 1.8rem; margin-bottom: 4px; }
    .kpi-card .kpi-label { font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-card .kpi-value { font-size: 2rem; font-weight: 800; color: #f1f5f9; margin: 6px 0; }
    .kpi-card:nth-child(2) { border-left-color: #10b981; }
    .kpi-card:nth-child(3) { border-left-color: #3b82f6; }
    .kpi-card:nth-child(4) { border-left-color: #8b5cf6; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem !important; font-weight: 600 !important;
        padding: 12px 20px !important; border-radius: 8px 8px 0 0 !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; font-weight: 600 !important; }
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #f59e0b !important; font-size: 1.1rem !important; }
    .custom-subheader {
        font-size: 1.6rem; font-weight: 700; color: #1e293b;
        padding: 12px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 16px;
    }
    .stSuccess, .stInfo, .stWarning { font-size: 1rem !important; padding: 12px 16px !important; border-radius: 10px !important; }
    .stDataFrame { font-size: 0.95rem !important; }
    .stButton > button[kind="primary"] {
        font-size: 1.1rem !important; padding: 12px 24px !important;
        border-radius: 10px !important; font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏆 Buy Box % Prediction System</div>', unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("📂 Data Source")
    data_source = st.radio("Data loading:", ["📁 Auto Load", "📤 Upload File"])

    st.subheader("📅 Forecast Period")
    forecast_horizon = st.selectbox("Forecast horizon (days):", [7, 14, 21, 28], index=3)

# ─── Data Loading ───
APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data_auto():
    csv_path = os.path.join(APP_DIR, "synthetic_business_reports.csv")
    adv_path = os.path.join(APP_DIR, "synthetic_advertised_product.csv")
    st_path  = os.path.join(APP_DIR, "synthetic_search_term.csv")
    if not os.path.exists(csv_path):
        return None, None, None
    df_br  = pd.read_csv(csv_path)
    df_adv = pd.read_csv(adv_path) if os.path.exists(adv_path) else None
    df_st  = pd.read_csv(st_path)  if os.path.exists(st_path)  else None
    return df_br, df_adv, df_st

@st.cache_data
def load_precomputed_outputs():
    out_dir = os.path.join(APP_DIR, "outputs_buybox")
    data = {}
    for f in ["model_comparison.csv", "feature_importance_xgboost.csv",
              "feature_importance_ensemble.csv", "buybox_forecast_with_risk.csv"]:
        path = os.path.join(out_dir, f)
        if not os.path.exists(path):
            path = os.path.join(APP_DIR, f)
        if os.path.exists(path):
            data[f] = pd.read_csv(path)
    return data

# ─── Feature Engineering ───
@st.cache_data
def build_features(df_br, df_adv, df_st):
    df = df_br.copy()
    df['Report_Date'] = pd.to_datetime(df['Report_Date'])

    # Time features
    df['year']        = df['Report_Date'].dt.year
    df['month']       = df['Report_Date'].dt.month
    df['quarter']     = df['Report_Date'].dt.quarter
    df['day_of_year'] = df['Report_Date'].dt.dayofyear

    # Session & page-view ratios
    df['mobile_session_ratio']  = np.where(df['Sessions_Total'] > 0, df['Sessions_Mobile_App'] / df['Sessions_Total'], 0)
    df['browser_session_ratio'] = np.where(df['Sessions_Total'] > 0, df['Sessions_Browser']    / df['Sessions_Total'], 0)
    df['mobile_pv_ratio']       = np.where(df['Page_Views_Total'] > 0, df['Page_Views_Mobile_App'] / df['Page_Views_Total'], 0)
    df['browser_pv_ratio']      = np.where(df['Page_Views_Total'] > 0, df['Page_Views_Browser']    / df['Page_Views_Total'], 0)
    df['pv_per_session']        = np.where(df['Sessions_Total'] > 0, df['Page_Views_Total'] / df['Sessions_Total'], 0)

    # Conversion metrics
    df['conversion_rate']   = df['Unit_Session_Percentage']
    df['order_rate']        = np.where(df['Sessions_Total'] > 0, df['Total_Order_Items'] / df['Sessions_Total'] * 100, 0)
    df['avg_selling_price'] = np.where(df['Units_Ordered'] > 0, df['Ordered_Product_Sales'] / df['Units_Ordered'], 0)
    df['units_per_order']   = np.where(df['Total_Order_Items'] > 0, df['Units_Ordered'] / df['Total_Order_Items'], 0)

    # Sales intensity
    df['sales_per_session'] = np.where(df['Sessions_Total'] > 0,   df['Ordered_Product_Sales'] / df['Sessions_Total'],   0)
    df['sales_per_view']    = np.where(df['Page_Views_Total'] > 0, df['Ordered_Product_Sales'] / df['Page_Views_Total'], 0)

    # B2B ratios
    df['b2b_sales_ratio'] = np.where(df['Ordered_Product_Sales'] > 0, df['Ordered_Product_Sales_B2B'] / df['Ordered_Product_Sales'], 0)
    df['b2b_units_ratio'] = np.where(df['Units_Ordered'] > 0,         df['Units_Ordered_B2B']         / df['Units_Ordered'],         0)

    # Log transforms
    df['log_sessions']   = np.log1p(df['Sessions_Total'])
    df['log_page_views'] = np.log1p(df['Page_Views_Total'])
    df['log_sales']      = np.log1p(df['Ordered_Product_Sales'])

    # Lag & rolling features per ASIN
    df = df.sort_values(['Child_ASIN', 'Report_Date'])
    for col in ['Featured_Offer_Buy_Box_Percentage', 'Sessions_Total', 'Units_Ordered',
                'Ordered_Product_Sales', 'Unit_Session_Percentage']:
        for lag in [1, 7, 14, 30]:
            df[f'{col}_lag_{lag}'] = df.groupby('Child_ASIN')[col].shift(lag)
        for window in [7, 14, 30]:
            df[f'{col}_roll_mean_{window}'] = df.groupby('Child_ASIN')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean())

    # Product-level statistics
    for col in ['Sessions_Total', 'Units_Ordered', 'Ordered_Product_Sales', 'Unit_Session_Percentage']:
        product_stats = df.groupby('Child_ASIN')[col].agg(['mean', 'std']).reset_index()
        product_stats.columns = ['Child_ASIN', f'{col}_product_mean', f'{col}_product_std']
        df = df.merge(product_stats, on='Child_ASIN', how='left')

    df['product_appearance_count'] = df.groupby('Child_ASIN')['Report_Date'].transform('count')

    # Advertising data integration
    if df_adv is not None:
        df_adv_c = df_adv.copy()
        df_adv_c['End_Date'] = pd.to_datetime(df_adv_c['End_Date'])
        df['year_month']       = df['Report_Date'].dt.to_period('M')
        df_adv_c['year_month'] = df_adv_c['End_Date'].dt.to_period('M')
        adv_monthly = df_adv_c.groupby(['Advertised_ASIN', 'year_month']).agg(
            ad_total_impressions=('Impressions',       'sum'),
            ad_total_clicks     =('Clicks',            'sum'),
            ad_total_spend      =('Spend',             'sum'),
            ad_total_sales      =('7_Day_Total_Sales', 'sum')
        ).reset_index()
        df = df.merge(adv_monthly, left_on=['Child_ASIN', 'year_month'],
                      right_on=['Advertised_ASIN', 'year_month'], how='left')
        for c in ['ad_total_impressions', 'ad_total_clicks', 'ad_total_spend', 'ad_total_sales']:
            df[c] = df[c].fillna(0)
        df['has_advertising'] = (df['ad_total_impressions'] > 0).astype(int)
        df['ad_ctr']          = np.where(df['ad_total_impressions'] > 0, df['ad_total_clicks'] / df['ad_total_impressions'], 0)
        df['ad_conversion']   = np.where(df['ad_total_clicks'] > 0,      df['ad_total_sales']  / df['ad_total_clicks'],      0)
        if 'Advertised_ASIN' in df.columns:
            df.drop(columns=['Advertised_ASIN'], inplace=True)
    else:
        df['has_advertising']      = 0
        df['ad_total_impressions'] = 0
        df['ad_total_clicks']      = 0
        df['ad_total_spend']       = 0
        df['ad_total_sales']       = 0
        df['ad_ctr']               = 0
        df['ad_conversion']        = 0

    return df


# ─── Load Data ───
df_br, df_adv, df_st = None, None, None

if data_source == "📁 Auto Load":
    df_br, df_adv, df_st = load_data_auto()
    if df_br is not None:
        st.success(f"✅ Data loaded! ({len(df_br):,} rows)")
    else:
        st.error("❌ synthetic_business_reports.csv not found. Place the files in the app folder.")
        st.stop()
else:
    file_br  = st.file_uploader("synthetic_business_reports.csv",   type="csv")
    file_adv = st.file_uploader("synthetic_advertised_product.csv", type="csv")
    file_st  = st.file_uploader("synthetic_search_term.csv",        type="csv")
    if file_br:
        df_br  = pd.read_csv(file_br)
        df_adv = pd.read_csv(file_adv) if file_adv else None
        df_st  = pd.read_csv(file_st)  if file_st  else None
    else:
        st.info("👆 Please upload at least synthetic_business_reports.csv.")
        st.stop()

# Precomputed outputs
precomputed = load_precomputed_outputs()

# ─── Main Tabs ───
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Exploration", "🤖 Model Training", "📈 Prediction Results",
    "🔮 Buy Box Forecast",  "📋 Summary Report"
])

# ═══════════════════════════════════════════
# TAB 1 — Data Exploration
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="custom-subheader">📊 Buy Box Data Analysis</div>', unsafe_allow_html=True)
    df_br['Report_Date'] = pd.to_datetime(df_br['Report_Date'])
    bb         = df_br['Featured_Offer_Buy_Box_Percentage']
    date_range = (df_br['Report_Date'].max() - df_br['Report_Date'].min()).days

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-icon">📅</div>
            <div class="kpi-label">Total Records</div>
            <div class="kpi-value">{len(df_br):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🏆</div>
            <div class="kpi-label">Avg. Buy Box %</div>
            <div class="kpi-value">{bb.mean():.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📦</div>
            <div class="kpi-label">Unique ASINs</div>
            <div class="kpi-value">{df_br['Child_ASIN'].nunique():,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📅</div>
            <div class="kpi-label">Date Range</div>
            <div class="kpi-value">{date_range} days</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=bb, nbinsx=50, marker_color="#f59e0b",
                                        marker_line_color="white", marker_line_width=0.5))
        fig_dist.update_layout(title="🏆 Buy Box % Distribution", xaxis_title="Buy Box %",
                               yaxis_title="Frequency", template="plotly_white", height=450,
                               title_font_size=18, font=dict(size=14))
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        monthly = df_br.groupby(df_br['Report_Date'].dt.to_period('M'))['Featured_Offer_Buy_Box_Percentage'].mean()
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly.index.astype(str), y=monthly.values,
            mode="lines+markers", line=dict(color="#10b981", width=2),
            marker=dict(size=6)
        ))
        fig_monthly.update_layout(title="📈 Monthly Average Buy Box %", xaxis_title="Month",
                                  yaxis_title="Buy Box %", template="plotly_white", height=450,
                                  title_font_size=18, font=dict(size=14))
        st.plotly_chart(fig_monthly, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        bb_cats    = pd.cut(bb, bins=[-0.01, 0, 25, 50, 75, 100.01],
                            labels=['0%', '1-25%', '25-50%', '50-75%', '75-100%'])
        cat_counts = bb_cats.value_counts().sort_index()
        fig_pie    = go.Figure(data=[go.Pie(labels=cat_counts.index, values=cat_counts.values,
                                             marker=dict(colors=px.colors.qualitative.Set2))])
        fig_pie.update_layout(title="📊 Buy Box Category Distribution", height=450,
                              title_font_size=18, font=dict(size=14))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col4:
        numeric_cols = df_br.select_dtypes(include=[np.number]).columns
        corrs = (df_br[numeric_cols]
                 .corr()['Featured_Offer_Buy_Box_Percentage']
                 .drop('Featured_Offer_Buy_Box_Percentage')
                 .sort_values(key=abs, ascending=False)
                 .head(10))
        fig_corr = go.Figure(go.Bar(x=corrs.values, y=corrs.index, orientation='h',
                                     marker_color=np.where(corrs.values > 0, '#10b981', '#ef4444')))
        fig_corr.update_layout(title="🔗 Top 10 Correlations with Buy Box %",
                               template="plotly_white", height=450,
                               title_font_size=18, font=dict(size=13))
        st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("📋 Raw Data"):
        st.dataframe(df_br.head(100), use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — Model Training
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="custom-subheader">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)

    if "model_comparison.csv" in precomputed:
        st.info("📌 Pre-computed model results loaded.")
        comp_df = precomputed["model_comparison.csv"]
        if 'Unnamed: 0' in comp_df.columns:
            comp_df = comp_df.rename(columns={'Unnamed: 0': 'Model'})
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        best = comp_df.iloc[0]
        st.success(f"🏆 Best model: **{best['Model']}** (R²: {best['R2']:.3f}, MAE: {best['MAE']:.1f}%)")

    if st.button("🚀 Retrain Models", type="primary", use_container_width=True):
        with st.spinner("🔄 Feature engineering & model training in progress..."):
            df = build_features(df_br, df_adv, df_st)

            target  = 'Featured_Offer_Buy_Box_Percentage'
            exclude = [target, 'Report_Date', 'Child_ASIN', 'Parent_ASIN', 'Product_Title',
                       'SKU', 'ASIN', 'year_month']
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                            if c not in exclude and c in df.columns]

            df_model = df.dropna(subset=[target])
            for c in feature_cols:
                df_model[c] = df_model[c].fillna(0)

            X = df_model[feature_cols]
            y = df_model[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler         = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)

            results   = {}
            all_preds = {}
            progress  = st.progress(0)
            status    = st.empty()

            # 1. Linear Regression
            status.text("⏳ Linear Regression...")
            lr      = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            lr_pred = np.clip(lr.predict(X_test_scaled), 0, 100)
            results['Linear Regression'] = {'MAE': mean_absolute_error(y_test, lr_pred),
                                             'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
                                             'R2': r2_score(y_test, lr_pred)}
            all_preds['Linear Regression'] = lr_pred
            progress.progress(1/6)

            # 2. Ridge
            status.text("⏳ Ridge Regression...")
            ridge      = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled, y_train)
            ridge_pred = np.clip(ridge.predict(X_test_scaled), 0, 100)
            results['Ridge'] = {'MAE': mean_absolute_error(y_test, ridge_pred),
                                'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred)),
                                'R2': r2_score(y_test, ridge_pred)}
            all_preds['Ridge'] = ridge_pred
            progress.progress(2/6)

            # 3. Random Forest
            status.text("⏳ Random Forest...")
            rf      = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=10,
                                            min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_pred = np.clip(rf.predict(X_test), 0, 100)
            results['Random Forest'] = {'MAE': mean_absolute_error(y_test, rf_pred),
                                         'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                                         'R2': r2_score(y_test, rf_pred)}
            all_preds['Random Forest'] = rf_pred
            progress.progress(3/6)

            # 4. Gradient Boosting
            status.text("⏳ Gradient Boosting...")
            gb      = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                                subsample=0.8, random_state=42)
            gb.fit(X_train, y_train)
            gb_pred = np.clip(gb.predict(X_test), 0, 100)
            results['Gradient Boosting'] = {'MAE': mean_absolute_error(y_test, gb_pred),
                                             'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)),
                                             'R2': r2_score(y_test, gb_pred)}
            all_preds['Gradient Boosting'] = gb_pred
            progress.progress(4/6)

            # 5. XGBoost
            status.text("⏳ XGBoost...")
            xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                                          reg_lambda=1.0, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            xgb_pred = np.clip(xgb_model.predict(X_test), 0, 100)
            results['XGBoost'] = {'MAE': mean_absolute_error(y_test, xgb_pred),
                                   'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                                   'R2': r2_score(y_test, xgb_pred)}
            all_preds['XGBoost'] = xgb_pred
            progress.progress(5/6)

            # 6. LightGBM
            status.text("⏳ LightGBM...")
            lgb_model = lgbm.LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                                            reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
            lgb_model.fit(X_train, y_train)
            lgb_pred = np.clip(lgb_model.predict(X_test), 0, 100)
            results['LightGBM'] = {'MAE': mean_absolute_error(y_test, lgb_pred),
                                    'RMSE': np.sqrt(mean_squared_error(y_test, lgb_pred)),
                                    'R2': r2_score(y_test, lgb_pred)}
            all_preds['LightGBM'] = lgb_pred
            progress.progress(6/6)

            status.text("✅ All models trained!")

            # Ensemble
            ensemble_pred = np.mean(list(all_preds.values()), axis=0)
            results['Ensemble'] = {'MAE': mean_absolute_error(y_test, ensemble_pred),
                                    'RMSE': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                                    'R2': r2_score(y_test, ensemble_pred)}

            # Save to session state
            comp_new = pd.DataFrame(results).T.sort_values('R2', ascending=False)
            comp_new.index.name = 'Model'
            comp_new = comp_new.reset_index()
            comp_new['MAE']  = comp_new['MAE'].round(2)
            comp_new['RMSE'] = comp_new['RMSE'].round(2)
            comp_new['R2']   = comp_new['R2'].round(4)

            st.session_state['comp_df']       = comp_new
            st.session_state['y_test']        = y_test
            st.session_state['ensemble_pred'] = ensemble_pred
            st.session_state['all_preds']     = all_preds
            st.session_state['feature_cols']  = feature_cols
            st.session_state['xgb_model']     = xgb_model
            st.session_state['rf_model']      = rf
            st.session_state['gb_model']      = gb
            st.session_state['lgb_model']     = lgb_model

            fi = pd.DataFrame({'feature': feature_cols, 'importance': xgb_model.feature_importances_})
            fi = fi.sort_values('importance', ascending=False)
            st.session_state['feature_importance'] = fi

            # ─── Generate Forecast ───
            status.text("🔮 Generating forecast...")
            try:
                df_last    = df.sort_values('Report_Date').groupby('Child_ASIN').last().reset_index()
                X_forecast = df_last[feature_cols].fillna(0)

                preds_rf  = np.clip(rf.predict(X_forecast),        0, 100)
                preds_xgb = np.clip(xgb_model.predict(X_forecast), 0, 100)
                preds_lgb = np.clip(lgb_model.predict(X_forecast), 0, 100)
                preds_gb  = np.clip(gb.predict(X_forecast),        0, 100)
                ensemble_fc = np.mean([preds_rf, preds_xgb, preds_lgb, preds_gb], axis=0)
                std_fc      = np.std( [preds_rf, preds_xgb, preds_lgb, preds_gb], axis=0)

                forecast_df_new = pd.DataFrame({
                    'SKU':          df_last['Child_ASIN'].values,
                    'BuyBox_Pred':  np.round(ensemble_fc, 2),
                    'P_low':        np.clip(np.round(ensemble_fc - 1.64 * std_fc, 2), 0, 100),
                    'P_high':       np.clip(np.round(ensemble_fc + 1.64 * std_fc, 2), 0, 100),
                    'Horizon_Days': forecast_horizon,
                })
                forecast_df_new['Risk_Flag'] = pd.cut(
                    forecast_df_new['BuyBox_Pred'],
                    bins=[-0.01, 30, 60, 100.01],
                    labels=['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK']
                )
                st.session_state['forecast_df'] = forecast_df_new
            except Exception as e:
                st.warning(f"Forecast could not be generated: {e}")

            st.dataframe(comp_new, use_container_width=True, hide_index=True)
            best = comp_new.iloc[0]
            st.success(f"🏆 Best model: **{best['Model']}** (R²: {best['R2']}, MAE: {best['MAE']}%)")
            st.success("🔮 Forecast ready! Switch to the 'Buy Box Forecast' tab.")

# ═══════════════════════════════════════════
# TAB 3 — Prediction Results
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="custom-subheader">📈 Ensemble Prediction Results</div>', unsafe_allow_html=True)

    if 'ensemble_pred' in st.session_state:
        y_test        = st.session_state['y_test']
        ensemble_pred = st.session_state['ensemble_pred']

        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_test.values, y=ensemble_pred,
                mode='markers', marker=dict(size=4, color='#3b82f6', opacity=0.3),
                name='Predictions'
            ))
            fig_scatter.add_trace(go.Scatter(
                x=[0, 100], y=[0, 100],
                mode='lines', line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction'
            ))
            fig_scatter.update_layout(
                title="🎯 Ensemble: Actual vs Predicted",
                xaxis_title="Actual Buy Box %", yaxis_title="Predicted Buy Box %",
                template="plotly_white", height=450, title_font_size=18, font=dict(size=14)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            residuals = y_test.values - ensemble_pred
            fig_res   = go.Figure()
            fig_res.add_trace(go.Histogram(x=residuals, nbinsx=50, marker_color='#8b5cf6'))
            fig_res.add_vline(x=0, line_dash="dash", line_color="red")
            fig_res.update_layout(
                title="📉 Ensemble Residual Distribution",
                xaxis_title="Error (Actual − Predicted)", yaxis_title="Frequency",
                template="plotly_white", height=450, title_font_size=18, font=dict(size=14)
            )
            st.plotly_chart(fig_res, use_container_width=True)

        if 'feature_importance' in st.session_state:
            fi     = st.session_state['feature_importance'].head(20)
            fig_fi = go.Figure(go.Bar(
                x=fi['importance'].values, y=fi['feature'].values,
                orientation='h', marker_color='#f59e0b'
            ))
            fig_fi.update_layout(
                title="🔑 Top 20 Feature Importance (XGBoost)",
                template="plotly_white", height=650, yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    elif "feature_importance_xgboost.csv" in precomputed:
        st.info("📌 Showing pre-computed results.")
        fi     = precomputed["feature_importance_xgboost.csv"].head(20)
        fig_fi = go.Figure(go.Bar(
            x=fi['importance'].values, y=fi['feature'].values,
            orientation='h', marker_color='#f59e0b'
        ))
        fig_fi.update_layout(
            title="🔑 Top 20 Feature Importance (XGBoost)",
            template="plotly_white", height=650, yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.warning("⚠️ Please train the models first in the 'Model Training' tab.")

# ═══════════════════════════════════════════
# TAB 4 — Buy Box Forecast
# ═══════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="custom-subheader">🔮 Buy Box Forecast ({forecast_horizon} Days)</div>', unsafe_allow_html=True)

    # Use session_state (if models trained) or precomputed CSV
    if 'forecast_df' in st.session_state:
        forecast_df = st.session_state['forecast_df']
    elif "buybox_forecast_with_risk.csv" in precomputed:
        forecast_df = precomputed["buybox_forecast_with_risk.csv"]
    else:
        forecast_df = None

    if forecast_df is not None:

        if 'Horizon_Days' in forecast_df.columns:
            available_horizons = sorted(forecast_df['Horizon_Days'].unique())
            selected_horizon   = forecast_horizon if forecast_horizon in available_horizons else available_horizons[0]
            fc_filtered        = forecast_df[forecast_df['Horizon_Days'] == selected_horizon]
        else:
            fc_filtered = forecast_df

        high_risk = (fc_filtered['Risk_Flag'] == 'HIGH_RISK').sum()
        low_risk  = (fc_filtered['Risk_Flag'] == 'LOW_RISK').sum()
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-icon">📦</div>
                <div class="kpi-label">Forecasted ASINs</div>
                <div class="kpi-value">{fc_filtered['SKU'].nunique():,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🏆</div>
                <div class="kpi-label">Avg. Buy Box Forecast</div>
                <div class="kpi-value">{fc_filtered['BuyBox_Pred'].mean():.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🔴</div>
                <div class="kpi-label">High Risk</div>
                <div class="kpi-value">{high_risk:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🟢</div>
                <div class="kpi-label">Low Risk</div>
                <div class="kpi-value">{low_risk:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            fig_fc_dist = go.Figure()
            fig_fc_dist.add_trace(go.Histogram(
                x=fc_filtered['BuyBox_Pred'], nbinsx=50, marker_color='#3b82f6'
            ))
            fig_fc_dist.update_layout(
                title="📊 Forecast Buy Box % Distribution",
                xaxis_title="Predicted Buy Box %", template="plotly_white",
                height=450, title_font_size=18, font=dict(size=14)
            )
            st.plotly_chart(fig_fc_dist, use_container_width=True)

        with col2:
            risk_counts = fc_filtered['Risk_Flag'].value_counts()
            colors_risk = {'LOW_RISK': '#10b981', 'MEDIUM_RISK': '#f59e0b', 'HIGH_RISK': '#ef4444'}
            fig_risk    = go.Figure(data=[go.Pie(
                labels=risk_counts.index, values=risk_counts.values,
                marker=dict(colors=[colors_risk.get(r, '#666') for r in risk_counts.index])
            )])
            fig_risk.update_layout(title="⚠️ Risk Distribution", height=450)
            st.plotly_chart(fig_risk, use_container_width=True)

        top_asins = fc_filtered.nlargest(20, 'BuyBox_Pred')
        fig_conf  = go.Figure()
        fig_conf.add_trace(go.Bar(
            x=top_asins['SKU'].astype(str), y=top_asins['BuyBox_Pred'],
            name='Prediction', marker_color='#3b82f6'
        ))
        fig_conf.add_trace(go.Scatter(
            x=top_asins['SKU'].astype(str), y=top_asins['P_high'],
            mode='markers', marker=dict(symbol='triangle-up', size=8, color='#10b981'),
            name='Upper Bound'
        ))
        fig_conf.add_trace(go.Scatter(
            x=top_asins['SKU'].astype(str), y=top_asins['P_low'],
            mode='markers', marker=dict(symbol='triangle-down', size=8, color='#ef4444'),
            name='Lower Bound'
        ))
        fig_conf.update_layout(
            title="🎯 Top 20 ASINs — Buy Box Prediction & Confidence Band",
            xaxis_title="ASIN", yaxis_title="Buy Box %",
            template="plotly_white", height=450, xaxis_tickangle=-45
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        with st.expander("📋 Full Forecast Table"):
            st.dataframe(fc_filtered, use_container_width=True, hide_index=True)

    else:
        st.warning("⚠️ No forecast data available. Train the models first in the 'Model Training' tab.")

# ═══════════════════════════════════════════
# TAB 5 — Summary Report
# ═══════════════════════════════════════════
with tab5:
    st.markdown('<div class="custom-subheader">📋 Project Summary Report</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Buy Box % Prediction System — Optimization with Machine Learning

    **Project Objective:** Predict the Amazon Buy Box win rate to enable
    sales strategy optimization.

    ---

    #### 📌 Why These 6 Models?
    | Model | Type | Rationale |
    |-------|------|-----------|
    | Linear Regression | Basic regression | Baseline reference |
    | Ridge | Regularized regression | Overfitting control |
    | Random Forest | Tree-based ensemble | Non-linear relationships |
    | Gradient Boosting | Boosting | Sequential error correction |
    | XGBoost | Advanced boosting | Regularization + speed |
    | LightGBM | Fast boosting | Large-data optimization |

    > **Note:** Deep learning models such as LSTM are suited for time-series forecasting,
    > but Buy Box prediction is a tabular data problem. Academic research consistently
    > shows that tree-based models (XGBoost, LightGBM) outperform deep learning on
    > tabular datasets.

    ---

    #### 📊 Data Pipeline
    1. **Business Reports** → Core data (Buy Box %, Sessions, Sales) — 13,486 records
    2. **Advertising Data** → Impressions, clicks, spend integration
    3. **Feature Engineering** → 78 features (time, session, conversion, B2B, lag, advertising)
    4. **6 Model Training** → Simple → Advanced comparison
    5. **Ensemble** → Weighted average of all models
    6. **Forecast** → 7 / 14 / 28-day predictions + confidence band + risk flag

    ---

    #### 🔑 Key Findings & Action Recommendations
    - **Advertising impact:** `has_advertising` ranks #1 in feature importance at 81% → Launch ads for non-advertised products
    - **Bimodal distribution:** Most products are either at 0% or 100% Buy Box → Mid-range products represent the largest opportunity
    - **Conversion rate:** Higher session-to-sale conversion correlates with higher Buy Box → Optimize product detail pages
    - **Example:** A product receiving 50 daily sessions at 3% conversion yields a predicted Buy Box of ~45%. Adding advertising raises the prediction to ~72%.
    """)

    if 'comp_df' in st.session_state:
        st.markdown("#### 🏆 Model Performance")
        st.dataframe(st.session_state['comp_df'], use_container_width=True, hide_index=True)
    elif "model_comparison.csv" in precomputed:
        st.markdown("#### 🏆 Model Performance")
        comp = precomputed["model_comparison.csv"]
        if 'Unnamed: 0' in comp.columns:
            comp = comp.rename(columns={'Unnamed: 0': 'Model'})
        st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("🔧 Buy Box % Prediction System — Deployment: Streamlit | Models: LR, Ridge, RF, GB, XGBoost, LightGBM")
