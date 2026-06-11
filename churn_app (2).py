import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Telco Churn Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }
    .metric-card {
        background: linear-gradient(135deg, #1e2435 0%, #252d40 100%);
        border: 1px solid #2d3655;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h2 { color: #7c9ef8; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #8892b0; font-size: 0.85rem; margin: 0.3rem 0 0; }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #cdd6f4; margin: 1.5rem 0 1rem;
        border-left: 3px solid #7c9ef8; padding-left: 0.8rem;
    }
    div[data-testid="stSidebar"] { background: #13171f; border-right: 1px solid #2d3655; }
    .stSelectbox label, .stFileUploader label { color: #8892b0 !important; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; }
    .stTabs [aria-selected="true"] { color: #7c9ef8 !important; border-color: #7c9ef8 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #7c9ef8, #5b78e8);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    h1, h2, h3 { color: #cdd6f4 !important; }
    p, li { color: #8892b0; }
    .stDataFrame { border: 1px solid #2d3655; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#cdd6f4",
)

# ── Demo data generator ──────────────────────────────────────────────────────
@st.cache_data
def make_demo_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male","Female"], n),
        "SeniorCitizen": rng.choice([0,1], n, p=[.84,.16]),
        "Partner": rng.choice(["Yes","No"], n),
        "Dependents": rng.choice(["Yes","No"], n),
        "tenure": rng.integers(1, 73, n),
        "PhoneService": rng.choice(["Yes","No"], n, p=[.9,.1]),
        "MultipleLines": rng.choice(["Yes","No","No phone service"], n),
        "InternetService": rng.choice(["DSL","Fiber optic","No"], n),
        "OnlineSecurity": rng.choice(["Yes","No","No internet service"], n),
        "OnlineBackup": rng.choice(["Yes","No","No internet service"], n),
        "DeviceProtection": rng.choice(["Yes","No","No internet service"], n),
        "TechSupport": rng.choice(["Yes","No","No internet service"], n),
        "StreamingTV": rng.choice(["Yes","No","No internet service"], n),
        "StreamingMovies": rng.choice(["Yes","No","No internet service"], n),
        "Contract": rng.choice(["Month-to-month","One year","Two year"], n, p=[.55,.24,.21]),
        "PaperlessBilling": rng.choice(["Yes","No"], n),
        "PaymentMethod": rng.choice([
            "Electronic check","Mailed check",
            "Bank transfer (automatic)","Credit card (automatic)"], n),
        "MonthlyCharges": rng.uniform(18, 118, n).round(2),
        "TotalCharges": rng.uniform(18, 8600, n).round(2),
        "Churn": rng.choice(["Yes","No"], n, p=[.265,.735]),
    })
    return df

# ── Preprocessing ────────────────────────────────────────────────────────────
@st.cache_data
def preprocess(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # drop customerID if present
    df.drop(columns=[c for c in ["customerID"] if c in df.columns], inplace=True)

    # TotalCharges sometimes comes as string with spaces
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # target
    if "Churn" not in df.columns:
        st.error("'Churn' column not found in dataset.")
        st.stop()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    y = df["Churn"]
    df_enc = df.copy()

    # encode all object / category columns
    le = LabelEncoder()
    for col in df_enc.select_dtypes(include=["object", "category"]).columns:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    # make sure SeniorCitizen is numeric (it can arrive as "Yes/No" text)
    if "SeniorCitizen" in df_enc.columns:
        df_enc["SeniorCitizen"] = pd.to_numeric(df_enc["SeniorCitizen"], errors="coerce").fillna(0)

    X = df_enc.drop(columns=["Churn"])

    # force everything to float safely
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, df_enc, X.columns.tolist(), scaler

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Telco Churn")
    st.markdown("**OEL Project** · By Daniyal")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data instead", value=uploaded is None)
    st.markdown("---")
    page = st.radio("Navigate", ["Overview","EDA","Models","Clustering","Predict"])

# ── Load data ────────────────────────────────────────────────────────────────
if uploaded and not use_demo:
    df_raw = pd.read_csv(uploaded)
    data_source = "Uploaded"
else:
    df_raw = make_demo_data()
    data_source = "Demo"

X_scaled, y, df_enc, feature_names, scaler = preprocess(df_raw)
df = df_raw.copy()
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
if "Churn" in df.columns:
    df["Churn_num"] = df["Churn"].map({"Yes":1,"No":0}).fillna(df["Churn"])

# ════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("📡 Telco Customer Churn Dashboard")
    st.caption(f"Data source: **{data_source}** — {len(df):,} customers")

    churn_rate = (df["Churn"] == "Yes").mean() if df["Churn"].dtype == object else df["Churn"].mean()
    avg_tenure = df["tenure"].mean() if "tenure" in df.columns else 0
    avg_charge = df["MonthlyCharges"].mean() if "MonthlyCharges" in df.columns else 0
    total_customers = len(df)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{total_customers:,}", "Total Customers"),
        (c2, f"{churn_rate:.1%}", "Churn Rate"),
        (c3, f"{avg_tenure:.1f} mo", "Avg Tenure"),
        (c4, f"${avg_charge:.2f}", "Avg Monthly Charge"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <h2>{val}</h2><p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
        churn_counts = df["Churn"].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index,
                     color_discrete_sequence=["#7c9ef8","#f87c7c"], hole=0.5)
        fig.update_layout(**PLOTLY_THEME, height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Contract Type vs Churn</div>', unsafe_allow_html=True)
        if "Contract" in df.columns:
            ct = df.groupby(["Contract","Churn"]).size().reset_index(name="count")
            fig = px.bar(ct, x="Contract", y="count", color="Churn",
                         barmode="group",
                         color_discrete_map={"Yes":"#f87c7c","No":"#7c9ef8"})
            fig.update_layout(**PLOTLY_THEME, height=300, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Tenure Distribution by Churn</div>', unsafe_allow_html=True)
    if "tenure" in df.columns:
        fig = px.histogram(df, x="tenure", color="Churn", nbins=40, barmode="overlay",
                           color_discrete_map={"Yes":"#f87c7c","No":"#7c9ef8"},
                           opacity=0.75)
        fig.update_layout(**PLOTLY_THEME, height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE: EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Monthly Charges by Churn</div>', unsafe_allow_html=True)
        if "MonthlyCharges" in df.columns:
            fig = px.box(df, x="Churn", y="MonthlyCharges",
                         color="Churn", color_discrete_map={"Yes":"#f87c7c","No":"#7c9ef8"})
            fig.update_layout(**PLOTLY_THEME, height=320, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Internet Service vs Churn</div>', unsafe_allow_html=True)
        if "InternetService" in df.columns:
            grp = df.groupby(["InternetService","Churn"]).size().reset_index(name="n")
            fig = px.bar(grp, x="InternetService", y="n", color="Churn", barmode="stack",
                         color_discrete_map={"Yes":"#f87c7c","No":"#7c9ef8"})
            fig.update_layout(**PLOTLY_THEME, height=320, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_df = df_enc.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(**PLOTLY_THEME, height=500, margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        st.markdown('<div class="section-header">Monthly Charges vs Tenure</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn",
                         opacity=0.6, color_discrete_map={"Yes":"#f87c7c","No":"#7c9ef8"})
        fig.update_layout(**PLOTLY_THEME, height=350, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE: MODELS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Models":
    st.title("🤖 Model Training & Evaluation")

    model_choice = st.selectbox("Select Model", [
        "Logistic Regression", "Random Forest", "Gradient Boosting"])
    test_size = st.slider("Test Split %", 10, 40, 20) / 100

    if st.button("Train Model"):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y)

        with st.spinner("Training…"):
            if model_choice == "Logistic Regression":
                clf = LogisticRegression(max_iter=1000, random_state=42)
            elif model_choice == "Random Forest":
                clf = RandomForestClassifier(n_estimators=150, random_state=42)
            else:
                clf = GradientBoostingClassifier(n_estimators=150, random_state=42)
            clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob)

        c1, c2, c3 = st.columns(3)
        for col, val, label in [
            (c1, f"{acc:.2%}", "Accuracy"),
            (c2, f"{auc:.4f}", "ROC-AUC"),
            (c3, model_choice, "Model"),
        ]:
            col.markdown(f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>',
                         unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc:.3f}",
                                     line=dict(color="#7c9ef8", width=2)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="#555")))
            fig.update_layout(**PLOTLY_THEME, height=350,
                              xaxis_title="FPR", yaxis_title="TPR",
                              margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_te, y_pred)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual"))
            fig.update_layout(**PLOTLY_THEME, height=350, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        if hasattr(clf, "feature_importances_"):
            st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
            imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=True)
            fig = px.bar(imp.tail(15), orientation="h",
                         color=imp.tail(15).values,
                         color_continuous_scale="Blues")
            fig.update_layout(**PLOTLY_THEME, height=400, margin=dict(t=10,b=10),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.text(classification_report(y_te, y_pred, target_names=["No Churn","Churn"]))

# ════════════════════════════════════════════════════════════════════════════
#  PAGE: CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    st.title("🔵 Customer Clustering")

    k = st.slider("Number of Clusters (K)", 2, 8, 3)

    if st.button("Run K-Means"):
        with st.spinner("Clustering…"):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(components, columns=["PC1","PC2"])
        pca_df["Cluster"] = labels.astype(str)
        pca_df["Churn"] = y.values

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">PCA — Cluster View</div>', unsafe_allow_html=True)
            fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", opacity=0.7)
            fig.update_layout(**PLOTLY_THEME, height=380, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Churn Rate per Cluster</div>', unsafe_allow_html=True)
            pca_df["Churn_n"] = pca_df["Churn"]
            cr = pca_df.groupby("Cluster")["Churn_n"].mean().reset_index()
            cr.columns = ["Cluster","Churn Rate"]
            fig = px.bar(cr, x="Cluster", y="Churn Rate",
                         color="Churn Rate", color_continuous_scale="Reds")
            fig.update_layout(**PLOTLY_THEME, height=380, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Cluster Size Distribution</div>', unsafe_allow_html=True)
        sz = pd.Series(labels).value_counts().reset_index()
        sz.columns = ["Cluster","Count"]
        fig = px.pie(sz, values="Count", names="Cluster",
                     color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig.update_layout(**PLOTLY_THEME, height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("🎯 Predict Churn for a Single Customer")
    st.info("Fill in the form below to get a churn probability estimate.", icon="ℹ️")

    # Train a quick RF on full data
    clf_pred = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_pred.fit(X_scaled, y)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male","Female"])
            senior = st.selectbox("Senior Citizen", ["No","Yes"])
            partner = st.selectbox("Partner", ["Yes","No"])
            dependents = st.selectbox("Dependents", ["Yes","No"])
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes","No"])
        with c2:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
            internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            online_sec = st.selectbox("Online Security", ["Yes","No","No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
            device_prot = st.selectbox("Device Protection", ["Yes","No","No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        with c3:
            streaming_tv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes","No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check","Mailed check",
                "Bank transfer (automatic)","Credit card (automatic)"])
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
            total = st.number_input("Total Charges ($)", 18.0, 9000.0, monthly*tenure, 1.0)

        submitted = st.form_submit_button("Predict Churn 🔮")

    if submitted:
        input_dict = {
            "gender": gender, "SeniorCitizen": 1 if senior=="Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup, "DeviceProtection": device_prot,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        input_df = pd.DataFrame([input_dict])
        le2 = LabelEncoder()
        for col in input_df.select_dtypes(include=["object"]).columns:
            # fit on training df_enc to stay consistent
            if col in df_enc.columns:
                le2.classes_ = df_enc[col].unique().astype(str)
                try:
                    input_df[col] = le2.transform(input_df[col].astype(str))
                except ValueError:
                    input_df[col] = 0
            else:
                input_df[col] = 0

        input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        # align columns
        for fn in feature_names:
            if fn not in input_df.columns:
                input_df[fn] = 0.0
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        prob = clf_pred.predict_proba(input_scaled)[0][1]

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            color = "#f87c7c" if prob > 0.5 else "#7cf8a8"
            risk = "HIGH RISK 🔴" if prob > 0.5 else "LOW RISK 🟢"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e2435,#252d40);
                        border:2px solid {color};border-radius:16px;
                        padding:2rem;text-align:center;margin-top:1.5rem;">
                <h1 style="color:{color};font-size:3rem;margin:0">{prob:.1%}</h1>
                <p style="color:#cdd6f4;font-size:1.2rem;margin:.5rem 0 0">Churn Probability</p>
                <p style="color:{color};font-size:1rem;font-weight:700;margin:.3rem 0 0">{risk}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            number={"suffix":"%","font":{"color":"#7c9ef8","size":40}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#8892b0"},
                "bar":{"color":"#7c9ef8"},
                "steps":[
                    {"range":[0,40],"color":"#1a2a1a"},
                    {"range":[40,70],"color":"#2a2a1a"},
                    {"range":[70,100],"color":"#2a1a1a"},
                ],
                "threshold":{"value":50,"line":{"color":"#f87c7c","width":3},"thickness":0.8},
            }
        ))
        fig.update_layout(**PLOTLY_THEME, height=300, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
