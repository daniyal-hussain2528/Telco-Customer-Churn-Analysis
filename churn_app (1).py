"""
Telco Customer Churn Analysis — Streamlit App
By Daniyal | OEL Project

Run:
    pip install streamlit plotly pandas numpy scikit-learn scipy
    streamlit run churn_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Main background */
.stApp { background-color: #0b0f1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2d45; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a2235;
    border: 1px solid #1f2d45;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: #64748b !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 28px !important;
}

/* Section headers */
h1 { color: #e2e8f0 !important; font-family: 'Space Grotesk', sans-serif !important; }
h2 { color: #93c5fd !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 18px !important; }
h3 { color: #94a3b8 !important; font-size: 14px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748b; border-radius: 6px;
    font-size: 13px; font-weight: 500; padding: 8px 16px;
}
.stTabs [aria-selected="true"] { background: #1a2235 !important; color: #e2e8f0 !important; }

/* Sidebar labels */
.sidebar-label {
    font-size: 11px; letter-spacing: 2px; color: #3b82f6;
    text-transform: uppercase; margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}

/* Hero band */
.hero-band {
    background: linear-gradient(135deg, #0f1e3a 0%, #111827 100%);
    border: 1px solid #1f2d45;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
}
.hero-title { font-size: 36px; font-weight: 700; color: #e2e8f0; line-height: 1.2; }
.hero-accent { color: #3b82f6; }
.hero-sub { color: #64748b; font-size: 15px; margin-top: 8px; }
.tag {
    display: inline-block; background: #1e3a5f; color: #93c5fd;
    font-size: 11px; padding: 4px 12px; border-radius: 20px;
    margin-right: 8px; margin-top: 12px; font-family: 'JetBrains Mono', monospace;
}

/* Info box */
.info-box {
    background: #1a2235; border: 1px solid #1f2d45;
    border-left: 3px solid #3b82f6;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    font-size: 13px; color: #94a3b8; line-height: 1.7;
}

/* Cluster cards */
.cluster-card {
    background: #1a2235; border: 1px solid #1f2d45;
    border-radius: 12px; padding: 20px; text-align: center;
}

/* Divider */
.divider { border-top: 1px solid #1f2d45; margin: 24px 0; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0b0f1a; }
::-webkit-scrollbar-thumb { background: #1f2d45; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY DARK TEMPLATE ───────────────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111827",
    font=dict(family="Space Grotesk, sans-serif", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1f2d45", linecolor="#1f2d45", tickcolor="#64748b"),
    yaxis=dict(gridcolor="#1f2d45", linecolor="#1f2d45", tickcolor="#64748b"),
    margin=dict(l=40, r=20, t=40, b=40),
)
COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

# ─── LOAD & PREPROCESS ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    return df

@st.cache_data
def preprocess(df):
    df2 = df.copy()
    le = LabelEncoder()

    # Encode ALL non-numeric columns including any leftover categories
    for col in df2.columns:
        if df2[col].dtype == "object" or str(df2[col].dtype) == "category":
            df2[col] = le.fit_transform(df2[col].astype(str))

    # Force every column to numeric, coerce any remaining issues
    for col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    df2.dropna(inplace=True)
    df2 = df2.reset_index(drop=True)

    X = df2.drop("Churn", axis=1)
    y = df2["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(float))
    return X_scaled, y, df2


def train_models(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
            "Precision": round(precision_score(y_test, y_pred) * 100, 1),
            "Recall":    round(recall_score(y_test, y_pred) * 100, 1),
            "F1 Score":  round(f1_score(y_test, y_pred) * 100, 1),
            "cm":        confusion_matrix(y_test, y_pred),
        }
    return results, X_test, y_test


def run_clustering(X_scaled):
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    db = DBSCAN(eps=1.5, min_samples=5)
    db_labels = db.fit_predict(X_scaled)
    return labels, db_labels

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-label">📡 Telco Churn</p>', unsafe_allow_html=True)
    st.markdown("### OEL Project")
    st.markdown('<div class="info-box">Upload your <b>WA_Fn-UseC_-Telco-Customer-Churn.csv</b> file or use demo data.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    use_demo = st.checkbox("Use demo data (simulated)", value=uploaded is None)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">View</p>', unsafe_allow_html=True)
    page = st.radio("", ["Overview", "EDA", "Models", "Clustering", "Predict"],
                    label_visibility="collapsed")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;color:#374151;text-align:center">By Daniyal · scikit-learn · Plotly</p>', unsafe_allow_html=True)

# ─── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def make_demo():
    np.random.seed(42)
    n = 1000
    tenure = np.random.randint(1, 72, n)
    monthly = np.random.uniform(20, 110, n)
    total = tenure * monthly + np.random.normal(0, 50, n)
    churn_prob = 1 / (1 + np.exp(0.05 * tenure - 0.03 * monthly + 1))
    churn = (np.random.rand(n) < churn_prob).astype(int)
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = np.random.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22])
    payment = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    gender = np.random.choice(["Male", "Female"], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    paperless = np.random.choice(["Yes", "No"], n)
    df = pd.DataFrame({
        "gender": gender, "SeniorCitizen": senior, "Partner": np.random.choice(["Yes","No"], n),
        "Dependents": np.random.choice(["Yes","No"], n), "tenure": tenure,
        "PhoneService": np.random.choice(["Yes","No"], n),
        "MultipleLines": np.random.choice(["Yes","No","No phone service"], n),
        "InternetService": internet,
        "OnlineSecurity": np.random.choice(["Yes","No","No internet service"], n),
        "OnlineBackup": np.random.choice(["Yes","No","No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes","No","No internet service"], n),
        "TechSupport": np.random.choice(["Yes","No","No internet service"], n),
        "StreamingTV": np.random.choice(["Yes","No","No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes","No","No internet service"], n),
        "Contract": contracts, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly.round(2),
        "TotalCharges": total.clip(0).round(2),
        "Churn": pd.Series(churn).map({0: "No", 1: "Yes"}),
    })
    return df

if uploaded:
    try:
        df_raw = load_data(uploaded)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    df_raw = make_demo()
    df_raw.drop_duplicates(inplace=True)
    df_raw.dropna(inplace=True)

with st.spinner("Processing data & training models..."):
    X_scaled, y, df_enc = preprocess(df_raw)

    cache_key = f"models_{len(df_raw)}_{df_raw.shape[1]}"
    if cache_key not in st.session_state:
        results, X_test, y_test = train_models(X_scaled, y)
        km_labels, db_labels = run_clustering(X_scaled)
        st.session_state[cache_key] = (results, X_test, y_test, km_labels, db_labels)
    else:
        results, X_test, y_test, km_labels, db_labels = st.session_state[cache_key]
best_model_name = max(results, key=lambda k: results[k]["Accuracy"])
best = results[best_model_name]

churn_rate = round(df_raw["Churn"].value_counts(normalize=True).get("Yes", 0) * 100, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f"""
    <div class="hero-band">
        <div class="hero-title">Telco Customer<br><span class="hero-accent">Churn Analysis</span></div>
        <div class="hero-sub">End-to-end ML pipeline — classification, evaluation, and customer segmentation.</div>
        <div>
            <span class="tag">WA Telco Dataset</span>
            <span class="tag">5 Classifiers</span>
            <span class="tag">K-Means · Hierarchical · DBSCAN</span>
            <span class="tag">OEL Project</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(df_raw):,}")
    c2.metric("Churn Rate", f"{churn_rate}%", delta="High risk" if churn_rate > 25 else "Moderate")
    c3.metric("Features", df_raw.shape[1] - 1)
    c4.metric("Best Accuracy", f"{best['Accuracy']}%", delta=best_model_name)
    c5.metric("Clusters", "3", delta="K-Means")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Pipeline")
        steps = ["Load CSV", "Clean & Fix Types", "Label Encode", "StandardScale",
                 "Train/Test Split", "Train 5 Models", "Evaluate & Cluster"]
        colors_pipe = ["#1e3a5f"] * len(steps)
        fig = go.Figure(go.Bar(
            x=[1] * len(steps), y=steps, orientation="h",
            marker_color=["#3b82f6" if i == 5 else "#1e3a5f" for i in range(len(steps))],
            text=steps, textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#e2e8f0", size=12),
        ))
        fig.update_layout(**PLOT_THEME, height=280, showlegend=False,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Model accuracy")
        names = list(results.keys())
        accs = [results[n]["Accuracy"] for n in names]
        bar_colors = ["#10b981" if n == best_model_name else "#3b82f6" for n in names]
        fig2 = go.Figure(go.Bar(
            x=accs, y=names, orientation="h",
            marker_color=bar_colors, text=[f"{a}%" for a in accs],
            textposition="outside", textfont=dict(color="#e2e8f0"),
        ))
        fig2.update_layout(**PLOT_THEME, height=280, showlegend=False,
                           xaxis=dict(range=[70, 92], gridcolor="#1f2d45"),
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Key insights")
    ins = [
        ("🔴", "High monthly charges strongly predict churn — churned customers pay significantly more."),
        ("🟠", "Short-tenure customers (< 12 months) are the highest-risk group."),
        ("🔵", "Month-to-month contracts show the highest churn rate vs annual plans."),
        ("🟣", "Fiber optic internet users churn more than DSL or no-internet users."),
    ]
    for icon, text in ins:
        st.markdown(f'<div class="info-box">{icon} &nbsp;{text}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.markdown("## Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        churn_counts = df_raw["Churn"].value_counts()
        fig = px.pie(
            values=churn_counts.values, names=churn_counts.index,
            title="Churn distribution",
            color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
            hole=0.55
        )
        fig.update_layout(**PLOT_THEME, showlegend=True)
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            df_raw, x="MonthlyCharges", color="Churn",
            title="Monthly charges by churn status",
            color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
            barmode="overlay", opacity=0.75, nbins=30
        )
        fig2.update_layout(**PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.box(
            df_raw, x="Churn", y="tenure",
            title="Tenure vs churn",
            color="Churn",
            color_discrete_map={"Yes": "#ef4444", "No": "#10b981"}
        )
        fig3.update_layout(**PLOT_THEME)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        contract_churn = df_raw.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
        fig4 = px.bar(
            contract_churn, x="Contract", y="Count", color="Churn",
            title="Churn by contract type", barmode="group",
            color_discrete_map={"Yes": "#ef4444", "No": "#10b981"}
        )
        fig4.update_layout(**PLOT_THEME)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Feature correlation heatmap")
    corr = df_enc.corr()
    fig5 = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature correlation matrix"
    )
    fig5.update_layout(**PLOT_THEME, height=550)
    fig5.update_traces(textfont_size=9)
    st.plotly_chart(fig5, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        internet_churn = df_raw.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
        fig6 = px.bar(internet_churn, x="InternetService", y="Count", color="Churn",
                      title="Churn by internet service", barmode="group",
                      color_discrete_map={"Yes": "#ef4444", "No": "#10b981"})
        fig6.update_layout(**PLOT_THEME)
        st.plotly_chart(fig6, use_container_width=True)

    with col6:
        pay_churn = df_raw.groupby(["PaymentMethod", "Churn"]).size().reset_index(name="Count")
        fig7 = px.bar(pay_churn, x="PaymentMethod", y="Count", color="Churn",
                      title="Churn by payment method", barmode="group",
                      color_discrete_map={"Yes": "#ef4444", "No": "#10b981"})
        fig7.update_layout(**PLOT_THEME)
        fig7.update_xaxes(tickangle=-20)
        st.plotly_chart(fig7, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Models":
    st.markdown("## Model evaluation")

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    df_res = pd.DataFrame(
        {n: {m: results[n][m] for m in metrics} for n in results}
    ).T.reset_index().rename(columns={"index": "Model"})

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_res.sort_values("Accuracy", ascending=True),
            x="Accuracy", y="Model", orientation="h",
            title="Accuracy comparison",
            color="Accuracy", color_continuous_scale=["#1e3a5f", "#3b82f6", "#10b981"],
            text="Accuracy"
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        categories = metrics
        fig2 = go.Figure()
        model_colors = {"Random Forest": "#10b981", "Logistic Regression": "#3b82f6",
                        "Decision Tree": "#f59e0b", "KNN": "#8b5cf6", "Naive Bayes": "#ef4444"}
        for name in results:
            vals = [results[name][m] for m in categories]
            vals.append(vals[0])
            fig2.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                name=name, fill="toself",
                line_color=model_colors.get(name, "#64748b"),
                fillcolor=model_colors.get(name, "#64748b").replace(")", ",0.1)").replace("rgb", "rgba")
                    if "rgb" in model_colors.get(name, "") else model_colors.get(name, "#64748b") + "1a"
            ))
        fig2.update_layout(
            **PLOT_THEME, title="Multi-metric radar",
            polar=dict(
                bgcolor="#111827",
                radialaxis=dict(visible=True, range=[60, 95], gridcolor="#1f2d45",
                                tickfont=dict(color="#64748b", size=10)),
                angularaxis=dict(gridcolor="#1f2d45", tickfont=dict(color="#94a3b8"))
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### All models — full metrics")
    st.dataframe(
        df_res.set_index("Model").style
            .background_gradient(cmap="Blues")
            .format("{:.1f}%"),
        use_container_width=True
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f"### Confusion matrix — {best_model_name} (best model)")

    cm = best["cm"]
    labels_cm = ["No Churn (0)", "Churn (1)"]
    fig_cm = px.imshow(
        cm, text_auto=True, x=labels_cm, y=labels_cm,
        color_continuous_scale=[[0, "#0b0f1a"], [0.5, "#1e3a5f"], [1, "#3b82f6"]],
        labels=dict(x="Predicted", y="Actual"),
        title=f"Confusion matrix — {best_model_name}"
    )
    fig_cm.update_layout(**PLOT_THEME, height=350)
    fig_cm.update_traces(textfont_size=18)
    st.plotly_chart(fig_cm, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{best['Accuracy']}%")
    c2.metric("Precision", f"{best['Precision']}%")
    c3.metric("Recall",    f"{best['Recall']}%")
    c4.metric("F1 Score",  f"{best['F1 Score']}%")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    st.markdown("## Customer segmentation")

    df_cluster = df_raw.copy()
    df_cluster["KMeans_Cluster"] = km_labels
    df_cluster["DBSCAN_Label"] = db_labels

    tab1, tab2, tab3 = st.tabs(["K-Means (k=3)", "Hierarchical", "DBSCAN"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.scatter(
                df_cluster, x="tenure", y="MonthlyCharges",
                color="KMeans_Cluster", title="K-Means clusters — tenure vs monthly charges",
                color_continuous_scale=px.colors.qualitative.Set2,
                opacity=0.7, size_max=5
            )
            fig.update_layout(**PLOT_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Segment profiles")
            seg_info = {
                0: ("🟢", "Loyal", "Long tenure, low charges. Low churn risk."),
                1: ("🔴", "High-risk", "Short tenure, high charges. Highest churn."),
                2: ("🟡", "Mid-tier", "Moderate values. Convertible with offers."),
            }
            for seg, (icon, name, desc) in seg_info.items():
                count = (km_labels == seg).sum()
                pct = round(count / len(km_labels) * 100, 1)
                st.markdown(f"""
                <div class="info-box">
                    <b>{icon} Segment {seg} — {name}</b><br>
                    {desc}<br>
                    <span style="color:#3b82f6;font-family:JetBrains Mono,monospace">{count} customers ({pct}%)</span>
                </div>
                """, unsafe_allow_html=True)

        seg_churn = df_cluster.groupby("KMeans_Cluster")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        seg_churn.columns = ["Segment", "Churn Rate (%)"]
        seg_churn["Segment"] = seg_churn["Segment"].map({0: "Loyal", 1: "High-Risk", 2: "Mid-Tier"})
        fig2 = px.bar(seg_churn, x="Segment", y="Churn Rate (%)",
                      title="Churn rate per segment",
                      color="Churn Rate (%)",
                      color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                      text="Churn Rate (%)")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(**PLOT_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("#### Hierarchical clustering — Ward linkage dendrogram")
        st.markdown('<div class="info-box">Dendrogram shows merge distances. Cut at a large gap to find natural cluster count.</div>', unsafe_allow_html=True)
        sample_idx = np.random.choice(len(X_scaled), min(300, len(X_scaled)), replace=False)
        X_sample = X_scaled[sample_idx]
        linked = linkage(X_sample, method="ward")
        fig_d, ax = plt.subplots(figsize=(12, 5))
        fig_d.patch.set_facecolor("#0b0f1a")
        ax.set_facecolor("#111827")
        dendrogram(linked, ax=ax, color_threshold=10,
                   above_threshold_color="#64748b",
                   leaf_rotation=90, leaf_font_size=0)
        ax.set_title("Dendrogram (sample of 300)", color="#e2e8f0", pad=12)
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1f2d45")
        ax.yaxis.label.set_color("#64748b")
        st.pyplot(fig_d)
        plt.close()

    with tab3:
        st.markdown("#### DBSCAN (eps=1.5, min_samples=5)")
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = (db_labels == -1).sum()
        c1, c2 = st.columns(2)
        c1.metric("Clusters found", n_clusters_db)
        c2.metric("Noise points", n_noise)
        df_cluster["DBSCAN_str"] = db_labels.astype(str).replace("-1", "Noise")
        fig3 = px.scatter(
            df_cluster, x="tenure", y="MonthlyCharges",
            color="DBSCAN_str", title="DBSCAN clusters",
            opacity=0.7
        )
        fig3.update_layout(**PLOT_THEME)
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown("## Live churn predictor")
    st.markdown('<div class="info-box">Enter customer details below to get a churn prediction from the best model (<b>Random Forest</b>).</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly charges ($)", 20.0, 120.0, 65.0, step=0.5)
        total = st.number_input("Total charges ($)", value=float(round(tenure * monthly, 2)))
    with col2:
        contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    with col3:
        senior = st.radio("Senior citizen", ["No", "Yes"])
        partner = st.radio("Has partner", ["Yes", "No"])
        paperless = st.radio("Paperless billing", ["Yes", "No"])

    if st.button("Predict churn →", type="primary"):
        try:
            X_full = df_enc.drop("Churn", axis=1)
            row = pd.DataFrame([{col: 0 for col in X_full.columns}])

            row["tenure"] = tenure
            row["MonthlyCharges"] = monthly
            row["TotalCharges"] = total
            row["SeniorCitizen"] = 1 if senior == "Yes" else 0
            if "Contract" in row.columns:
                row["Contract"] = ["Month-to-month", "One year", "Two year"].index(contract)
            if "InternetService" in row.columns:
                row["InternetService"] = ["DSL", "Fiber optic", "No"].index(internet)
            if "PaperlessBilling" in row.columns:
                row["PaperlessBilling"] = 1 if paperless == "Yes" else 0
            if "Partner" in row.columns:
                row["Partner"] = 1 if partner == "Yes" else 0

            row = row[X_full.columns].astype(float)
            scaler2 = StandardScaler()
            scaler2.fit(X_full.astype(float))
            row_scaled = scaler2.transform(row)

            rf_model = results["Random Forest"]["model"]
            proba = rf_model.predict_proba(row_scaled)[0][1]
            pred = "Yes" if proba >= 0.5 else "No"
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", "Will Churn" if pred == "Yes" else "Will Stay",
                  delta="High risk" if pred == "Yes" else "Low risk")
        c2.metric("Churn probability", f"{round(proba * 100, 1)}%")
        c3.metric("Model used", "Random Forest", delta=f"{best['Accuracy']}% accuracy")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix": "%", "font": {"color": "#e2e8f0", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b",
                         "tickfont": {"color": "#64748b"}},
                "bar": {"color": "#ef4444" if proba >= 0.5 else "#10b981"},
                "bgcolor": "#111827",
                "steps": [
                    {"range": [0, 40], "color": "#0f2018"},
                    {"range": [40, 65], "color": "#1e2d0a"},
                    {"range": [65, 100], "color": "#2d0f0f"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 3}, "value": 50}
            },
            title={"text": "Churn probability", "font": {"color": "#94a3b8", "size": 14}}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Space Grotesk"), height=280,
                                margin=dict(l=30, r=30, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if pred == "Yes":
            st.markdown("""
            <div class="info-box">
                ⚠️ <b>Retention recommendation:</b> This customer is at high churn risk.
                Consider offering a discounted annual contract, loyalty credits, or improved internet service plan.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                ✅ <b>Low risk:</b> This customer is likely to stay.
                Good candidate for upsell — offer additional services or an upgrade plan.
            </div>
            """, unsafe_allow_html=True)
