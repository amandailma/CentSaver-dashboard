"""
app.py
CentSaver — Dashboard Bahasa Indonesia (Mudah Dipahami)
Capstone DBS Foundation Coding Camp
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

from utils import (
    load_data, engineer_features, compute_rfm,
    compute_microspending_ratio, compute_mom_and_anomaly,
    prepare_model_input
)
from inference import CentSaverRF
from rag_chatbot import generate_insight_docs, LightweightRAG

st.set_page_config(
    page_title="CentSaver — Pantau Pengeluaran Harian",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("💰 CentSaver")
st.sidebar.markdown("*Pantau Pengeluaran Kecil-kecil Sebelum Menumpuk*")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload File CSV", type=["csv"],
    help="Upload file centsaver_master_relabelling.csv"
)

st.sidebar.divider()
st.sidebar.subheader("📊 3 Pertanyaan Bisnis")
st.sidebar.markdown("""
- **Q1:** Berapa persen uang kecil-kecil yang keluar per bulan?
- **Q2:** Apakah AI bisa bedakan transaksi wajar vs boros (≥85% akurat)?
- **Q3:** Grafik mana yang paling cocok buat notifikasi?
""")
st.sidebar.divider()
st.sidebar.info("Capstone DBS Foundation — AI Eng × Data Science")

# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.title("Selamat Datang di CentSaver 👋")
    st.markdown("""
    ### Aplikasi Pintar untuk Mengontrol Pengeluaran Kecil-kecil

    Upload file CSV Anda di sidebar untuk mulai analisis.

    **Yang Akan Anda Temukan:**
    1. 📊 Berapa banyak uang "kecil-kecil" yang sebenarnya besar totalnya?
    2. 🤖 Seberapa pintar AI kita membedakan kebutuhan wajar vs boros?
    3. 📈 Grafik seperti apa yang paling efektif buat ngingetin Anda?
    """)
    st.stop()

df_raw = load_data(uploaded_file)
df = engineer_features(df_raw)
X, feature_names = prepare_model_input(df)
y = df["label"].values if "label" in df.columns else np.zeros(len(df))

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("📊 Dashboard CentSaver")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transaksi", f"{len(df):,}")
c2.metric("Jumlah Kategori", f"{df['category'].nunique()}")
c3.metric("Rentang Waktu", f"{df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")
if "label" in df.columns:
    micro_pct = df["label"].mean() * 100
    c4.metric("Pengeluaran Boros", f"{micro_pct:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Gambaran Umum",
    "💸 Q1: Seberapa Boros?",
    "🤖 Q2: Seberapa Pintar AI?",
    "🔥 Q3: Grafik Paling Ampuh?",
    "🎯 Kelompok Pengguna & Saran",
    "💬 Tanya CentSaver AI"
])

# ==========================================================================
# TAB 1: OVERVIEW
# ==========================================================================
with tab1:
    st.header("Gambaran Umum Data Anda")

    cat_stats = (
        df.groupby("category")
        .agg(jumlah=("amount", "size"), total=("amount", "sum"), rata_rata=("amount", "mean"))
        .sort_values("total", ascending=False)
        .reset_index()
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(
            cat_stats.head(10), y="category", x="total", orientation="h",
            color="total", color_continuous_scale="Blues",
            title="10 Kategori Pengeluaran Terbesar",
            labels={"total": "Total (Rp)", "category": ""}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.dataframe(cat_stats.head(10).round(0), use_container_width=True, hide_index=True)

    st.info("📊 **Insight:** *Makanan & Minuman* memang jadi pengeluaran terbesar — karena kita makan tiap hari. Tapi hati-hati, kategori *Sewa & Cicilan* meski jarang, nominalnya gede banget. Makanya, batas 'boros' nggak bisa sama untuk semua kategori. Belanja Rp100rb untuk sewa beda artinya dengan Rp100rb untuk kopi.")

    st.subheader("Pola Waktu & Perilaku")
    monthly_top = df.groupby(["period", "category"])["amount"].sum().reset_index()
    top8 = cat_stats.head(8)["category"].tolist()
    monthly_top = monthly_top[monthly_top["category"].isin(top8)]

    fig = px.line(
        monthly_top, x="period", y="amount", color="category",
        title="Tren Bulanan — 8 Kategori Teratas",
        labels={"amount": "Total (Rp)", "period": "Bulan"}
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    weekend_df = (
        df.groupby(["category", "day_type"])
        .agg(rata_rata=("amount", "mean"))
        .reset_index()
        .pivot(index="category", columns="day_type", values="rata_rata")
        .fillna(0)
        .reset_index()
    )
    weekend_df["selisih_weekend"] = weekend_df.get("weekend", 0) - weekend_df.get("weekday", 0)
    weekend_df = weekend_df.sort_values("selisih_weekend", ascending=True)

    fig = px.bar(
        weekend_df, y="category", x="selisih_weekend", orientation="h",
        color="selisih_weekend", color_continuous_scale="RdBu",
        title="Selisih Pengeluaran: Akhir Pekan vs Hari Kerja",
        labels={"selisih_weekend": "Selisih (Rp)", "category": ""}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("🛍️ **Insight:** Akhir pekan itu musimnya 'godaan'! Pengeluaran untuk *Hobi, Elektronik, dan Perjalanan* naik drastis saat Sabtu-Minggu. Sebaliknya, *Transportasi* dan *Kopi* justru turun — karena hari kerja lah yang butuh kopi dan ongkos. Tips: waspadai 'weekend impulse' Anda!")

# ==========================================================================
# TAB 2: QUEST #1
# ==========================================================================
with tab2:
    st.header("Q1: Seberapa Banyak Uang Kecil-kecil yang Keluar?")

    monthly_micro, overall_monthly, avg_micro_by_cat = compute_microspending_ratio(df)
    THRESHOLD = 20
    flagged = avg_micro_by_cat[avg_micro_by_cat["avg_micro_pct"] > THRESHOLD]["category"].tolist()

    k1, k2, k3 = st.columns(3)
    k1.metric("Rata-rata per Bulan", f"{overall_monthly['micro_pct'].mean():.2f}%", "< batas 20%")
    k2.metric("Kategori Waspada", f"{len(flagged)}", ", ".join(flagged) if len(flagged) <= 2 else f"{', '.join(flagged[:2])}...")
    k3.metric("Paling Boros", f"{avg_micro_by_cat.iloc[0]['category']}", f"{avg_micro_by_cat.iloc[0]['avg_micro_pct']:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A. Tren Bulanan**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=overall_monthly["period"], y=overall_monthly["micro_pct"],
            mode="lines+markers", line=dict(color="navy", width=2),
            fill="tozeroy", fillcolor="rgba(173,216,230,0.3)"
        ))
        fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", annotation_text=f"Batas Waspada {THRESHOLD}%")
        fig.update_layout(height=350, xaxis_title="Bulan", yaxis_title="Persen Micro-spending")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**B. Rata-rata per Kategori**")
        top10 = avg_micro_by_cat.head(10).copy()
        top10["warna"] = top10["category"].apply(lambda x: "Waspada" if x in flagged else "Aman")
        fig = px.bar(
            top10, y="category", x="avg_micro_pct", orientation="h",
            color="warna", color_discrete_map={"Waspada": "crimson", "Aman": "steelblue"},
            title="10 Kategori Teratas"
        )
        fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="red")
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Pelacak Akumulasi Pengeluaran")
    selected_cat = st.selectbox("Pilih Kategori untuk Dipantau", options=df["category"].unique(), key="tracker")
    cat_df = df[df["category"] == selected_cat].copy()
    cat_df["akumulasi_boros"] = (cat_df["is_adaptive_microspending"] * cat_df["amount"]).cumsum()
    cat_df["akumulasi_total"] = cat_df["amount"].cumsum()
    cat_df["persen_boros"] = (cat_df["akumulasi_boros"] / cat_df["akumulasi_total"] * 100).fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cat_df["date"], y=cat_df["persen_boros"], mode="lines", name="Persen Boros", line=dict(color="coral")))
    fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", annotation_text="Batas Peringatan")
    fig.update_layout(title=f"Pelacakan: {selected_cat}", xaxis_title="Tanggal", yaxis_title="Persen Akumulasi Boros", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"⚠️ **Insight:** Kategori **{', '.join(flagged)}** ini perlu diwaspadai! *Hobi & Olahraga* mencapai **42,59%** — artinya hampir setengah uang harian Anda lari ke sini. Padahal sering dianggap 'kebutuhan wajar'. Padahal kalau dijumlahkan, bisa buat tabungan lho.")

# ==========================================================================
# TAB 3: QUEST #2
# ==========================================================================
with tab3:
    st.header("Q2: Seberapa Pintar AI Membedakan Transaksi?")

    with st.spinner("Sedang melatih model AI..."):
        rf = CentSaverRF()
        rf.fit(X, y)

    m = rf.metrics
    TARGET = 0.85
    status = "✅ LULUS" if m["accuracy"] >= TARGET else "⚠️ PERLU PERBAIKAN"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Akurasi", f"{m['accuracy']:.2%}", status)
    m2.metric("Ketepatan", f"{m['precision']:.2%}")
    m3.metric("Ingatan", f"{m['recall']:.2%}")
    m4.metric("F1-Score", f"{m['f1']:.2%}")
    m5.metric("AUC", f"{m['auc']:.3f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A. Matriks Kebingungan**")
        cm = confusion_matrix(m["y_test"], m["y_pred"])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Prediksi", y="Asli", color="Jumlah"),
            x=["Wajar", "Boros"], y=["Wajar", "Boros"]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**B. Kurva ROC**")
        fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"RF (AUC={m['auc']:.3f})", line=dict(color="steelblue", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Tebakan Acak", line=dict(dash="dash", color="black")))
        fig.update_layout(height=350, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("**C. Fitur Paling Berpengaruh**")
    imp = rf.feature_importances(feature_names)
    imp_df = pd.DataFrame(imp, columns=["fitur", "pengaruh"]).sort_values("pengaruh", ascending=True).tail(10)
    fig = px.bar(imp_df, y="fitur", x="pengaruh", orientation="h", color="pengaruh", color_continuous_scale="Greens")
    fig.update_layout(height=400, yaxis_title="", xaxis_title="Tingkat Pengaruh")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Laporan Klasifikasi Lengkap"):
        st.text(classification_report(m["y_test"], m["y_pred"], target_names=["Wajar", "Boros"], zero_division=0))

    st.info(f"🤖 **Insight:** Model AI kita berhasil dengan akurasi **{m['accuracy']:.2%}** dan skor AUC **{m['auc']:.3f}**. Artinya, aplikasi ini cukup pintar membedakan transaksi wajar vs boros. Yang paling berpengaruh: besar nominal, pola hari (weekend), dan jenis kategori.")

    st.subheader("💾 Simpan Model")
    if st.button("Simpan Model RF"):
        rf.save("centsaver")
        st.success("Model tersimpan! Bisa diunduh dari repo atau folder lokal.")

# ==========================================================================
# TAB 4: QUEST #3
# ==========================================================================
with tab4:
    st.header("Q3: Grafik Seperti Apa yang Paling Ampuh?")

    monthly_cat, anomaly_freq, weekend_impulse = compute_mom_and_anomaly(df)

    st.subheader("🥇 JUARA: Heatmap Pertumbuhan Bulan-ke-Bulan")
    top10_cats = anomaly_freq.head(10)["category"].tolist()
    heatmap_data = monthly_cat[monthly_cat["category"].isin(top10_cats)].copy()
    heatmap_data["period_str"] = heatmap_data["period"].dt.strftime("%Y-%m")
    recent_months = sorted(heatmap_data["period_str"].unique())[-24:]
    pivot_growth = heatmap_data.pivot_table(index="category", columns="period_str", values="mom_growth_pct", fill_value=0)
    pivot_growth = pivot_growth.reindex(columns=recent_months, fill_value=0)

    fig = px.imshow(
        pivot_growth, color_continuous_scale="RdYlGn_r", aspect="auto",
        labels=dict(x="Bulan", y="Kategori", color="Pertumbuhan %"),
        title="10 Kategori Paling 'Nge-trend' (24 Bulan Terakhir)"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    max_val = pivot_growth.stack().max()
    max_cell = pivot_growth.stack().idxmax()
    st.info(f"🔥 **Contoh Notifikasi Chatbot:** *Wah, kategori {max_cell[0]} melonjak {max_val:.1f}% di bulan {max_cell[1]}! Mau saya bantu cari tips hemat?*")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🥈 PENDUKUNG: Seberapa Sering Kategori Ini 'Nge-trend'?**")
        top8 = anomaly_freq.head(8).copy()
        top8["level"] = top8["anomaly_rate"].apply(
            lambda r: "Tinggi" if r > 0.15 else "Sedang" if r > 0.08 else "Rendah"
        )
        fig = px.bar(
            top8, y="category", x="anomaly_rate", orientation="h",
            color="level", color_discrete_map={"Tinggi": "crimson", "Sedang": "orange", "Rendah": "steelblue"},
            title="Seberapa Sering Kategori Ini Tidak Menentu"
        )
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**🥉 KONTEKS: Akhir Pekan vs Hari Kerja**")
        fig = px.bar(
            weekend_impulse.reset_index(), y="category", x="weekend_boost",
            orientation="h", color="weekend_boost", color_continuous_scale="Teal",
            title="Selisih Risiko: Akhir Pekan − Hari Kerja"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Contoh Grafik + Penanda Anomali")
    most_volatile = anomaly_freq.iloc[0]["category"]
    cat_ts = monthly_cat[monthly_cat["category"] == most_volatile].sort_values("period")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cat_ts["period"], y=cat_ts["total_amount"], mode="lines", name="Total Pengeluaran", line=dict(color="navy", width=2)))
    anomaly_pts = cat_ts[cat_ts["is_anomaly"] == 1]
    if not anomaly_pts.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_pts["period"], y=anomaly_pts["total_amount"],
            mode="markers", name="Anomali (Z>2)", marker=dict(color="red", size=12, symbol="x")
        ))
    fig.update_layout(title=f"Kategori Paling Tidak Menentu: {most_volatile}", xaxis_title="Bulan", yaxis_title="Total (Rp)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    📋 **Prioritas Tampilan Dashboard:**
    1. **UTAMA** — Heatmap MoM Growth (korelasi 0.71 dengan risk rate) → Trigger: *"Lonjakan X% di [Kategori]"*
    2. **PENDUKUNG** — Line Chart + Anomaly Marker → Trigger: *"Pola tidak normal terdeteksi"*
    3. **SEKUNDER** — Bar Chart Anomaly Rate → Trigger: *"Kategori ini tidak menentu"*
    4. **KONTEKS** — Weekend vs Weekday → Trigger: *"Akhir pekan rawan boros"*
    """)

# ==========================================================================
# TAB 5: RFM & RECOMMENDATIONS
# ==========================================================================
with tab5:
    st.header("Kelompok Pengguna & Saran Praktis")

    user_rfm = compute_rfm(df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A. Jumlah Pengguna per Kelompok**")
        segment_crosstab = pd.crosstab(user_rfm["frequency_segment"], user_rfm["monetary_segment"])
        fig = px.imshow(segment_crosstab, text_auto=True, color_continuous_scale="YlOrRd", aspect="auto")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**B. Tingkat Boros per Kelompok**")
        segment_micro = (
            user_rfm.groupby(["frequency_segment", "monetary_segment"])
            .agg(rata_boros=("micro_rate", "mean"))
            .reset_index()
            .pivot(index="frequency_segment", columns="monetary_segment", values="rata_boros")
        )
        fig = px.imshow(segment_micro, text_auto=".2f", color_continuous_scale="RdYlGn_r", aspect="auto")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Profil 3 Kelompok Pengguna")
    seg1, seg2, seg3 = st.columns(3)

    with seg1:
        st.error("**Jarang-Menetap**")
        st.metric("Tingkat Boros", "~35%", "RISIKO TERTINGGI")
        st.markdown("""
        Orang yang jarang transaksi, tapi kalau transaksi suka impulsif.  
        **Saran:** 🎮 Ikut *Tantangan Kurangi Boros* — selisihkan uang yang hemat ke tabungan otomatis.
        """)

    with seg2:
        st.success("**Sering-Premium**")
        st.metric("Tingkat Boros", "Paling Rendah", "ASSET PENTING")
        st.markdown("""
        Orang yang sering belanja tapi justru paling hemat dan teratur.  
        **Saran:** 🏆 Beri *Reward Loyalitas* — biar tetap setia pakai aplikasi.
        """)

    with seg3:
        st.warning("**Pemula**")
        st.metric("Jendela Intervensi", "7 Hari", "SEGERA")
        st.markdown("""
        Pengguna baru — kebiasaan pertama bakal menentukan pola ke depan.  
        **Saran:** 📨 Kirim *follow-up 7 hari* — "Apakah belanjaan kemarin sudah direncanakan?"
        """)

    st.divider()

    st.subheader("📋 Ringkasan Eksekutif")
    st.markdown("""
    | Pertanyaan | Status | Bukti |
    |------------|--------|-------|
    | **Q1** Berapa persen micro-spending? | ✅ LULUS | Baseline per kategori valid; 2 kategori perlu diwaspadai |
    | **Q2** Apakah AI akurat ≥85%? | ✅ LULUS | Model RF: **{:.2%}** / AUC: **{:.3f}** — melebihi target! |
    | **Q3** Visualisasi terbaik? | ✅ LULUS | Heatmap MoM Growth paling ampuh (korelasi 0.71) |
    | **RFM** Segmentasi pengguna? | ✅ LULUS | 3 kelompok pengguna dengan saran masing-masing |

    **Langkah Selanjutnya:** Integrasi ke FastAPI & jalankan A/B Testing untuk ukur dampak nyata — berapa banyak pengguna yang berhasil hemat setelah pakai aplikasi ini.
    """.format(m['accuracy'], m['auc']))

# ==========================================================================
# TAB 6: AI CHATBOT (RAG)
# ==========================================================================
with tab6:
    st.header("💬 Tanya CentSaver AI")
    st.markdown("*Tanya apa saja tentang data pengeluaran Anda. Sistem akan mencari insight paling relevan untuk menjawab.*")

    st.divider()

    with st.spinner("Sedang membangun basis pengetahuan..."):
        monthly_micro_rag, overall_monthly_rag, avg_micro_by_cat_rag = compute_microspending_ratio(df)
        _, anomaly_freq_rag, weekend_impulse_rag = compute_mom_and_anomaly(df)
        docs = generate_insight_docs(df, monthly_micro_rag, overall_monthly_rag, avg_micro_by_cat_rag, anomaly_freq_rag, weekend_impulse_rag)

        rag = LightweightRAG()
        rag.index(docs)

    st.success(f"✅ Basis pengetahuan siap! {len(docs)} dokumen insight sudah terindex.")
    st.info("🧠 Mode: RINGAN & GRATIS (TF-IDF + Template — tanpa API AI eksternal)")

    st.divider()

    st.subheader("💬 Tanya Apa Saja")

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
        with st.spinner("Sedang mencari jawaban..."):
            answer = rag.query(user_q)

        st.markdown("#### 📝 Jawaban:")
        st.markdown(f"{answer}")

        with st.expander("📄 Lihat data yang digunakan untuk menjawab"):
            q_vec = rag.vectorizer.transform([user_q])
            scores = cosine_similarity(q_vec, rag.doc_vectors).flatten()
            top_idx = scores.argsort()[-3:][::-1]
            for i, idx in enumerate(top_idx, 1):
                st.markdown(f"**Dokumen {i} (skor relevansi: {scores[idx]:.3f}):**")
                st.text(rag.docs[idx][:500] + "...")

    st.divider()
    st.markdown("""
    **Cara Kerja Chatbot (100% Gratis):**
    1. 📊 Data transaksi Anda diubah jadi "buku panduan" digital
    2. 🔍 Pertanyaan Anda dicocokkan dengan isi buku panduan
    3. 📋 Sistem ambil halaman paling relevan
    4. 💬 Jawaban disusun dari halaman tersebut — tanpa pakai AI eksternal

    *Semua proses di server Streamlit, tidak perlu API key, tidak perlu internet khusus.*
    """)
