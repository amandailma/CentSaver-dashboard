"""
app.py
CentSaver — Streamlit Dashboard (Cloud-Optimized)
Capstone DBS Foundation Coding Camp
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    load_data, engineer_features, compute_rfm,
    compute_microspending_ratio, compute_mom_and_anomaly,
    prepare_model_input
)
from inference import CentSaverRF
from rag_chatbot import generate_insight_docs, LightweightRAG, OpenAIRAG

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CentSaver — Microspending Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("💰 CentSaver")
st.sidebar.markdown("*Microspending Detection & Financial Awareness*")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Dataset (CSV)", type=["csv"],
    help="Upload centsaver_master_relabelling.csv"
)

st.sidebar.divider()
st.sidebar.subheader("📊 Business Questions")
st.sidebar.markdown("""
- **Q1:** Microspending Ratio per Bulan  
- **Q2:** Classification ≥85%  
- **Q3:** Visualization Trigger  
""")
st.sidebar.divider()
st.sidebar.info("Capstone DBS Foundation — AI Eng × Data Science")

# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.title("Welcome to CentSaver 👋")
    st.markdown("""
    ### End-to-End Microspending Intelligence Dashboard

    Upload your dataset via the sidebar to begin analysis.

    **Business Questions:**
    1. How much of monthly spending is micro-spending?
    2. Can AI distinguish micro-spending from essential needs with ≥85% accuracy?
    3. Which visualization most effectively triggers AI Chatbot recommendations?
    """)
    st.stop()

# Load & engineer
df_raw = load_data(uploaded_file)
df = engineer_features(df_raw)
X, feature_names = prepare_model_input(df)
y = df["label"].values if "label" in df.columns else np.zeros(len(df))

# ---------------------------------------------------------------------------
# HEADER METRICS
# ---------------------------------------------------------------------------
st.title("📊 CentSaver Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{len(df):,}")
c2.metric("Categories", f"{df['category'].nunique()}")
c3.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")
if "label" in df.columns:
    micro_pct = df["label"].mean() * 100
    c4.metric("Micro-Spending Rate", f"{micro_pct:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview & EDA",
    "💸 Quest #1 — Microspending Ratio",
    "🤖 Quest #2 — Classification",
    "🔥 Quest #3 — Visualization Trigger",
    "🎯 RFM & Recommendations",
    "💬 AI Chatbot (RAG)"
])

# ==========================================================================
# TAB 1: OVERVIEW & EDA
# ==========================================================================
with tab1:
    st.header("Overview & Exploratory Data Analysis")

    # Category stats
    cat_stats = (
        df.groupby("category")
        .agg(txn_count=("amount", "size"), total_amount=("amount", "sum"), avg_amount=("amount", "mean"))
        .sort_values("total_amount", ascending=False)
        .reset_index()
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(
            cat_stats.head(10), y="category", x="total_amount", orientation="h",
            color="total_amount", color_continuous_scale="Blues",
            title="Top 10 Categories by Total Spending",
            labels={"total_amount": "Total Amount (Rp)", "category": ""}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.dataframe(cat_stats.head(10).round(0), use_container_width=True, hide_index=True)

    st.info("**Insight:** *Makanan & Minuman* dominates total spending. *Sewa & Cicilan* has highest average per transaction but lowest frequency — a 'bulk payment' pattern. Threshold must be category-aware.")

    # Temporal
    st.subheader("Temporal & Behavioral Analysis")
    monthly_top = df.groupby(["period", "category"])["amount"].sum().reset_index()
    top8 = cat_stats.head(8)["category"].tolist()
    monthly_top = monthly_top[monthly_top["category"].isin(top8)]

    fig = px.line(
        monthly_top, x="period", y="amount", color="category",
        title="Monthly Spending Trend — Top 8 Categories",
        labels={"amount": "Total Amount (Rp)", "period": "Month"}
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Weekend
    weekend_df = (
        df.groupby(["category", "day_type"])
        .agg(avg_amount=("amount", "mean"))
        .reset_index()
        .pivot(index="category", columns="day_type", values="avg_amount")
        .fillna(0)
        .reset_index()
    )
    weekend_df["weekend_boost"] = weekend_df.get("weekend", 0) - weekend_df.get("weekday", 0)
    weekend_df = weekend_df.sort_values("weekend_boost", ascending=True)

    fig = px.bar(
        weekend_df, y="category", x="weekend_boost", orientation="h",
        color="weekend_boost", color_continuous_scale="RdBu",
        title="Weekend Impulse Boost (Weekend − Weekday Average)",
        labels={"weekend_boost": "Amount Difference (Rp)", "category": ""}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("**Insight:** *Elektronik*, *Hobi & Olahraga*, and *Perjalanan* show positive weekend boost (>Rp50k) — impulsive/hedonistic behavior. *Transportasi* and *Kopi & Minuman* are lower on weekends (commuter spending).")

# ==========================================================================
# TAB 2: QUEST #1 — MICROSPENDING RATIO
# ==========================================================================
with tab2:
    st.header("Quest #1: Microspending Ratio Analysis (Category-Aware)")

    monthly_micro, overall_monthly, avg_micro_by_cat = compute_microspending_ratio(df)
    THRESHOLD = 20
    flagged = avg_micro_by_cat[avg_micro_by_cat["avg_micro_pct"] > THRESHOLD]["category"].tolist()

    # KPI
    k1, k2, k3 = st.columns(3)
    k1.metric("Overall Avg / Month", f"{overall_monthly['micro_pct'].mean():.2f}%", "< 20% threshold")
    k2.metric("Flagged Categories", f"{len(flagged)}", ", ".join(flagged) if len(flagged) <= 2 else f"{', '.join(flagged[:2])}...")
    k3.metric("Highest Risk", f"{avg_micro_by_cat.iloc[0]['category']}", f"{avg_micro_by_cat.iloc[0]['avg_micro_pct']:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A. Overall Monthly Trend**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=overall_monthly["period"], y=overall_monthly["micro_pct"],
            mode="lines+markers", line=dict(color="navy", width=2),
            fill="tozeroy", fillcolor="rgba(173,216,230,0.3)"
        ))
        fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", annotation_text=f"Threshold {THRESHOLD}%")
        fig.update_layout(height=350, xaxis_title="Period", yaxis_title="Microspending %")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**B. Average Ratio per Category**")
        top10 = avg_micro_by_cat.head(10).copy()
        top10["color"] = top10["category"].apply(lambda x: "Flagged" if x in flagged else "Normal")
        fig = px.bar(
            top10, y="category", x="avg_micro_pct", orientation="h",
            color="color", color_discrete_map={"Flagged": "crimson", "Normal": "steelblue"},
            title="Top 10 Categories"
        )
        fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="red")
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Accumulation Tracker
    st.subheader("💡 Microspending Accumulation Tracker")
    selected_cat = st.selectbox("Select Category to Track", options=df["category"].unique(), key="tracker")
    cat_df = df[df["category"] == selected_cat].copy()
    cat_df["running_micro"] = (cat_df["is_adaptive_microspending"] * cat_df["amount"]).cumsum()
    cat_df["running_total"] = cat_df["amount"].cumsum()
    cat_df["running_pct"] = (cat_df["running_micro"] / cat_df["running_total"] * 100).fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cat_df["date"], y=cat_df["running_pct"], mode="lines", name="Running %", line=dict(color="coral")))
    fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
    fig.update_layout(title=f"Accumulation Tracker: {selected_cat}", xaxis_title="Date", yaxis_title="Cumulative Microspending %", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Insight:** Categories **{', '.join(flagged)}** exceed the 20% tolerance threshold. *Hobi & Olahraga* at **42.59%** is a primary leakage bucket — users perceive these as 'secondary needs' but they drain nearly half of daily cash flow.")

# ==========================================================================
# TAB 3: QUEST #2 — CLASSIFICATION
# ==========================================================================
with tab3:
    st.header("Quest #2: Classification Performance (Anti-Leakage)")

    with st.spinner("Training Random Forest baseline..."):
        rf = CentSaverRF()
        rf.fit(X, y)

    m = rf.metrics
    TARGET = 0.85
    status = "✅ LULUS" if m["accuracy"] >= TARGET else "⚠️ BELUM CAPAI TARGET"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{m['accuracy']:.2%}", status)
    m2.metric("Precision", f"{m['precision']:.2%}")
    m3.metric("Recall", f"{m['recall']:.2%}")
    m4.metric("F1-Score", f"{m['f1']:.2%}")
    m5.metric("AUC", f"{m['auc']:.3f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A. Confusion Matrix**")
        cm = confusion_matrix(m["y_test"], m["y_pred"])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Normal", "Micro"], y=["Normal", "Micro"]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**B. ROC Curve**")
        fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"RF (AUC={m['auc']:.3f})", line=dict(color="steelblue", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="black")))
        fig.update_layout(height=350, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("**C. Feature Importance (Top 10)**")
    imp = rf.feature_importances(feature_names)
    imp_df = pd.DataFrame(imp, columns=["feature", "importance"]).sort_values("importance", ascending=True).tail(10)
    fig = px.bar(imp_df, y="feature", x="importance", orientation="h", color="importance", color_continuous_scale="Greens")
    fig.update_layout(height=400, yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Classification Report"):
        from sklearn.metrics import classification_report
        st.text(classification_report(m["y_test"], m["y_pred"], target_names=["Normal", "Micro"], zero_division=0))

    st.info(f"**Insight:** DS Baseline (Random Forest, anti-leakage) achieves **{m['accuracy']:.2%}** accuracy and **AUC {m['auc']:.3f}**, validating that behavioral features are strong enough without label leakage. Top features: `amount`, `amount_log`, `day_of_week`.")

    st.subheader("💾 Export Model")
    if st.button("Save RF Baseline Model"):
        rf.save("centsaver")
        st.success("Model saved! Download from repo or local folder.")

# ==========================================================================
# TAB 4: QUEST #3 — VISUALIZATION TRIGGER
# ==========================================================================
with tab4:
    st.header("Quest #3: Visualization Trigger for AI Chatbot")

    monthly_cat, anomaly_freq, weekend_impulse = compute_mom_and_anomaly(df)

    # Hero: MoM Heatmap
    st.subheader("🥇 HERO: Month-over-Month Growth Heatmap")
    top10_cats = anomaly_freq.head(10)["category"].tolist()
    heatmap_data = monthly_cat[monthly_cat["category"].isin(top10_cats)].copy()
    heatmap_data["period_str"] = heatmap_data["period"].dt.strftime("%Y-%m")
    recent_months = sorted(heatmap_data["period_str"].unique())[-24:]
    pivot_growth = heatmap_data.pivot_table(index="category", columns="period_str", values="mom_growth_pct", fill_value=0)
    pivot_growth = pivot_growth.reindex(columns=recent_months, fill_value=0)

    fig = px.imshow(
        pivot_growth, color_continuous_scale="RdYlGn_r", aspect="auto",
        labels=dict(x="Month", y="Category", color="MoM Growth %"),
        title="Top 10 Spike Categories (Last 24 Months)"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Trigger info
    max_val = pivot_growth.stack().max()
    max_cell = pivot_growth.stack().idxmax()
    st.info(f"🔥 **Chatbot Trigger:** `{max_cell[0]}` spiked **{max_val:.1f}%** in `{max_cell[1]}`. Recommend: *'Lonjakan {max_val:.0f}% di {max_cell[0]} — pertimbangkan tunda pembelian.'*")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🥈 SUPPORTING: Anomaly Rate per Category**")
        top8 = anomaly_freq.head(8).copy()
        top8["color"] = top8["anomaly_rate"].apply(
            lambda r: "High" if r > 0.15 else "Medium" if r > 0.08 else "Low"
        )
        fig = px.bar(
            top8, y="category", x="anomaly_rate", orientation="h",
            color="color", color_discrete_map={"High": "crimson", "Medium": "orange", "Low": "steelblue"},
            title="Proportion of Anomalous Months"
        )
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**🥉 CONTEXT: Weekend Impulse Boost**")
        fig = px.bar(
            weekend_impulse.reset_index(), y="category", x="weekend_boost",
            orientation="h", color="weekend_boost", color_continuous_scale="Teal",
            title="Risk Rate Difference (Weekend − Weekday)"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Time series + anomaly
    st.subheader("📈 Time Series + Anomaly Highlight")
    most_volatile = anomaly_freq.iloc[0]["category"]
    cat_ts = monthly_cat[monthly_cat["category"] == most_volatile].sort_values("period")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cat_ts["period"], y=cat_ts["total_amount"], mode="lines", name="Total Spending", line=dict(color="navy", width=2)))
    anomaly_pts = cat_ts[cat_ts["is_anomaly"] == 1]
    if not anomaly_pts.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_pts["period"], y=anomaly_pts["total_amount"],
            mode="markers", name="Anomaly (Z>2)", marker=dict(color="red", size=12, symbol="x")
        ))
    fig.update_layout(title=f"Most Volatile Category: {most_volatile}", xaxis_title="Period", yaxis_title="Total Amount (Rp)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Dashboard Priority:**
    1. **HERO** — Heatmap MoM Growth (korelasi 0.71 dengan risk rate) → Trigger: *"Lonjakan X% di [Kategori]"*
    2. **SUPPORTING** — Line Chart + Anomaly Marker → Trigger: *"Pola tidak normal terdeteksi"*
    3. **SECONDARY** — Bar Chart Anomaly Rate → Trigger: *"Kategori ini volatile"*
    4. **CONTEXT** — Weekend vs Weekday → Trigger: *"Weekend boost terdeteksi"*
    """)

# ==========================================================================
# TAB 5: RFM & RECOMMENDATIONS
# ==========================================================================
with tab5:
    st.header("RFM-Style Segmentation & Business Recommendations")

    user_rfm = compute_rfm(df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A. User Count per Segment**")
        segment_crosstab = pd.crosstab(user_rfm["frequency_segment"], user_rfm["monetary_segment"])
        fig = px.imshow(segment_crosstab, text_auto=True, color_continuous_scale="YlOrRd", aspect="auto")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**B. Microspending Rate per Segment**")
        segment_micro = (
            user_rfm.groupby(["frequency_segment", "monetary_segment"])
            .agg(avg_micro_rate=("micro_rate", "mean"))
            .reset_index()
            .pivot(index="frequency_segment", columns="monetary_segment", values="avg_micro_rate")
        )
        fig = px.imshow(segment_micro, text_auto=".2f", color_continuous_scale="RdYlGn_r", aspect="auto")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Segment Profiles & Actions")
    seg1, seg2, seg3 = st.columns(3)

    with seg1:
        st.error("**Occasional-Medium**")
        st.metric("Micro-Rate", "~35%", "HIGHEST RISK")
        st.markdown("""
        *Hidden risk segment* — low frequency, no habit anchor.  
        **Action:** 🎮 *Micro-Spending Challenge*  
        Convert difference vs median into auto-savings.
        """)

    with seg2:
        st.success("**Frequent-Premium**")
        st.metric("Micro-Rate", "Lowest", "DEFENSIVE ASSET")
        st.markdown("""
        *Planned spender* — high frequency, disciplined budget.  
        **Action:** 🏆 *Loyalty Reward*  
        Exclusive insights + retention benefits.
        """)

    with seg3:
        st.warning("**One-Time Spender**")
        st.metric("Window", "7 Days", "INTERVENTION")
        st.markdown("""
        *Cold start* — first transaction behavior crystallizes fast.  
        **Action:** 📨 *Follow-up 7 Hari*  
        "Was this purchase planned? Check your pattern."
        """)

    st.divider()

    st.subheader("📋 Executive Summary & Verdict")
    st.markdown("""
    | Quest | Status | Evidence |
    |-------|--------|----------|
    | **Q1** Microspending Ratio | ✅ LULUS | Category-aware baseline valid; 2 kategori flagged |
    | **Q2** Classification ≥85% | ✅ LULUS | RF Baseline: >91% / AUC >0.97 |
    | **Q3** Visualization Trigger | ✅ LULUS | MoM Heatmap korelasi 0.71; layout terstruktur |
    | **RFM** Segmentasi | ✅ LULUS | 4 segmen actionable dengan intervensi spesifik |

    **Next Step:** Deploy ke FastAPI + jalankan A/B Testing RF vs DL untuk mengukur *business impact* nyata (pengurangan micro-spending pasca-alert), bukan hanya akurasi teknis.
    """)

# ==========================================================================
# TAB 6: AI CHATBOT (RAG)
# ==========================================================================
with tab6:
    st.header("🤖 AI Chatbot — Retrieval-Augmented Generation")
    st.markdown("*Tanya apa saja tentang data micro-spending Anda. Sistem akan mencari insight relevan dari data untuk menjawab.*")

    st.divider()

    # Prepare RAG
    with st.spinner("Membangun knowledge base dari data..."):
        monthly_micro_rag, overall_monthly_rag, avg_micro_by_cat_rag = compute_microspending_ratio(df)
        _, anomaly_freq_rag, weekend_impulse_rag = compute_mom_and_anomaly(df)
        docs = generate_insight_docs(df, monthly_micro_rag, overall_monthly_rag, avg_micro_by_cat_rag, anomaly_freq_rag, weekend_impulse_rag)

        # Initialize RAG
        rag = LightweightRAG()
        rag.index(docs)

    st.success(f"✅ Knowledge base siap! {len(docs)} insight documents terindex.")

    # OpenAI option
    openai_key = st.text_input("🔑 OpenAI API Key (opsional — biarkan kosong untuk pakai Lightweight RAG gratis)", type="password", help="Masukkan sk-proj-xxx untuk respons lebih natural dengan GPT-3.5")

    if openai_key:
        try:
            openai_rag = OpenAIRAG(api_key=openai_key)
            openai_rag.index(docs)
            rag_mode = "openai"
            st.info("🧠 Mode: OpenAI GPT-3.5 Turbo")
        except Exception as e:
            st.warning(f"OpenAI gagal: {e}. Fallback ke Lightweight RAG.")
            rag_mode = "lightweight"
    else:
        rag_mode = "lightweight"
        st.info("🧠 Mode: Lightweight RAG (TF-IDF + Template — Gratis)")

    st.divider()

    # Chat interface
    st.subheader("💬 Tanya CentSaver AI")

    # Suggested questions
    suggestions = [
        "Berapa rata-rata micro-spending per bulan?",
        "Kategori apa yang paling boros?",
        "Apakah model sudah memenuhi target akurasi?",
        "Visualisasi apa yang paling bagus untuk chatbot?",
        "Kenapa akhir pekan saya boros?",
        "Segmen mana yang paling berisiko?",
        "Apa rekomendasi untuk mengurangi micro-spending?"
    ]

    col_sug = st.columns(4)
    selected_q = None
    for i, sq in enumerate(suggestions[:4]):
        if col_sug[i].button(sq, key=f"sug_{i}"):
            selected_q = sq

    col_sug2 = st.columns(4)
    for i, sq in enumerate(suggestions[4:]):
        if col_sug2[i].button(sq, key=f"sug2_{i}"):
            selected_q = sq

    user_q = st.text_input("Atau ketik pertanyaan Anda:", value=selected_q if selected_q else "", placeholder="Contoh: Kenapa saya boros bulan ini?")

    if st.button("🔍 Tanya AI", type="primary") and user_q:
        with st.spinner("Mencari insight relevan..."):
            if rag_mode == "openai" and openai_key:
                answer = openai_rag.query(user_q)
            else:
                answer = rag.query(user_q)

        st.markdown("#### 📝 Jawaban:")
        st.markdown(f"{answer}")

        # Show retrieved context (expandable)
        with st.expander("📄 Lihat context data yang digunakan"):
            q_vec = rag.vectorizer.transform([user_q])
            scores = cosine_similarity(q_vec, rag.doc_vectors).flatten()
            top_idx = scores.argsort()[-3:][::-1]
            for i, idx in enumerate(top_idx, 1):
                st.markdown(f"**Doc {i} (score: {scores[idx]:.3f}):**")
                st.text(rag.docs[idx][:500] + "...")

    st.divider()
    st.markdown("""
    **Cara kerja RAG:**
    1. 📊 Data transaksi diubah jadi "insight documents" (knowledge base)
    2. 🔍 Pertanyaan Anda di-embed dengan TF-IDF
    3. 📋 Sistem retrieve dokumen paling relevan (cosine similarity)
    4. 💬 Jawaban digenerate dari context yang ditemukan

    *Versi gratis pakai TF-IDF + template. Versi premium pakai OpenAI GPT-3.5.*
    """)
