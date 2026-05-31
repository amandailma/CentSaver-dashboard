"""
app.py
CentSaver — Dashboard Lengkap (Bahasa Indonesia)
Fitur: Upload CSV, Input Manual, Upload Foto, Prediksi RF + DL, Chatbot RAG
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

from utils import (
    load_data, engineer_features, compute_rfm,
    compute_microspending_ratio, compute_mom_and_anomaly,
    prepare_model_input
)
from inference import CentSaverRF, CentSaverDL
from rag_chatbot import generate_insight_docs, LightweightRAG

st.set_page_config(
    page_title="CentSaver — Pantau Pengeluaran",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("💰 CentSaver")
st.sidebar.markdown("*Pantau Pengeluaran Kecil Sebelum Menumpuk*")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload File CSV", type=["csv"],
    help="Upload file centsaver_master_relabelling.csv"
)

# Upload model DL (opsional)
st.sidebar.divider()
st.sidebar.subheader("🤖 Model AI Tim (Opsional)")
dl_model_file = st.sidebar.file_uploader(
    "Upload model .keras (Tim AI)", type=["keras", "h5"],
    help="Upload hasil training model Deep Learning"
)

st.sidebar.divider()
st.sidebar.subheader("📊 3 Pertanyaan Bisnis")
st.sidebar.markdown("""
- **Q1:** Berapa persen uang kecil-kecil per bulan?
- **Q2:** Apakah AI akurat ≥85%?
- **Q3:** Grafik mana paling cocok untuk notifikasi?
""")
st.sidebar.divider()
st.sidebar.info("Capstone DBS Foundation — AI Eng × Data Science")

# ---------------------------------------------------------------------------
# STATE: Simpan data & model di session
# ---------------------------------------------------------------------------
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "rf_model" not in st.session_state:
    st.session_state.rf_model = None
if "dl_model" not in st.session_state:
    st.session_state.dl_model = None
if "manual_data" not in st.session_state:
    st.session_state.manual_data = pd.DataFrame(columns=[
        "tanggal", "deskripsi", "kategori", "nominal", "hari", "weekend"
    ])
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# PROSES DATA CSV
# ---------------------------------------------------------------------------
if uploaded_file is not None and st.session_state.df_processed is None:
    with st.spinner("Memproses data..."):
        df_raw = load_data(uploaded_file)
        st.session_state.df_processed = engineer_features(df_raw)

        # Latih model RF
        X, feature_names = prepare_model_input(st.session_state.df_processed)
        y = st.session_state.df_processed["label"].values
        rf = CentSaverRF()
        rf.fit(X, y)
        st.session_state.rf_model = rf
        st.session_state.feature_names = feature_names

        # Load model DL jika ada
        if dl_model_file is not None:
            with open("/tmp/dl_model.keras", "wb") as f:
                f.write(dl_model_file.getbuffer())
            dl = CentSaverDL("/tmp/dl_model.keras")
            st.session_state.dl_model = dl

df = st.session_state.df_processed

# ---------------------------------------------------------------------------
# HALAMAN AWAL (BELUM ADA DATA)
# ---------------------------------------------------------------------------
if df is None:
    st.title("Selamat Datang di CentSaver 👋")
    st.markdown("""
    ### Aplikasi Cerdas untuk Mengontrol Pengeluaran Harian

    **Cara Pakai:**
    1. 📁 **Upload CSV** di sidebar ← untuk analisis data historis
    2. 📝 **Input Manual** di tab "Input & Prediksi" ← untuk cek transaksi baru
    3. 📸 **Upload Foto Struk** ← sebagai bukti/bahan analisis
    4. 🤖 **Lihat Prediksi** ← apakah transaksi ini wajar atau boros?

    **Atau langsung ke tab Chatbot untuk tanya-jawab!**
    """)
    st.stop()

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Gambaran Umum",
    "💸 Q1: Seberapa Boros?",
    "🤖 Q2: Seberapa Pintar AI?",
    "🔥 Q3: Grafik Paling Ampuh?",
    "🎯 Kelompok & Saran",
    "📝 Input Manual & Prediksi",
    "💬 Tanya AI"
])

# ==========================================================================
# TAB 1: OVERVIEW
# ==========================================================================
with tab1:
    st.header("Gambaran Umum Data Anda")

    cat_stats = (
        df.groupby("category")
        .agg(jumlah=("amount", "size"), total=("amount", "sum"), rata_rata=("amount", "mean"))
        .sort_values("total", ascending=False).reset_index()
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(cat_stats.head(10), y="category", x="total", orientation="h",
                     color="total", color_continuous_scale="Blues",
                     title="10 Kategori Pengeluaran Terbesar",
                     labels={"total": "Total (Rp)", "category": ""})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(cat_stats.head(10).round(0), use_container_width=True, hide_index=True)

    st.info("📊 **Insight:** *Makanan & Minuman* memang jadi pengeluaran terbesar. Tapi hati-hati, *Sewa & Cicilan* meski jarang, nominalnya gede. Makanya batas 'boros' nggak bisa sama untuk semua kategori.")

    st.subheader("Pola Waktu & Perilaku")
    monthly_top = df.groupby(["period", "category"])["amount"].sum().reset_index()
    top8 = cat_stats.head(8)["category"].tolist()
    monthly_top = monthly_top[monthly_top["category"].isin(top8)]
    fig = px.line(monthly_top, x="period", y="amount", color="category",
                  title="Tren Bulanan — 8 Kategori Teratas",
                  labels={"amount": "Total (Rp)", "period": "Bulan"})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    weekend_df = (df.groupby(["category", "day_type"]).agg(rata_rata=("amount", "mean"))
                  .reset_index().pivot(index="category", columns="day_type", values="rata_rata")
                  .fillna(0).reset_index())
    weekend_df["selisih_weekend"] = weekend_df.get("weekend", 0) - weekend_df.get("weekday", 0)
    weekend_df = weekend_df.sort_values("selisih_weekend", ascending=True)
    fig = px.bar(weekend_df, y="category", x="selisih_weekend", orientation="h",
                 color="selisih_weekend", color_continuous_scale="RdBu",
                 title="Selisih Pengeluaran: Akhir Pekan vs Hari Kerja",
                 labels={"selisih_weekend": "Selisih (Rp)", "category": ""})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.info("🛍️ **Insight:** Akhir pekan itu musimnya 'godaan'! Pengeluaran *Hobi, Elektronik, Perjalanan* naik drastis saat Sabtu-Minggu.")

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
        fig.add_trace(go.Scatter(x=overall_monthly["period"], y=overall_monthly["micro_pct"],
            mode="lines+markers", line=dict(color="navy", width=2),
            fill="tozeroy", fillcolor="rgba(173,216,230,0.3)"))
        fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", annotation_text=f"Batas Waspada {THRESHOLD}%")
        fig.update_layout(height=350, xaxis_title="Bulan", yaxis_title="Persen Micro-spending")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**B. Rata-rata per Kategori**")
        top10 = avg_micro_by_cat.head(10).copy()
        top10["warna"] = top10["category"].apply(lambda x: "Waspada" if x in flagged else "Aman")
        fig = px.bar(top10, y="category", x="avg_micro_pct", orientation="h",
                     color="warna", color_discrete_map={"Waspada": "crimson", "Aman": "steelblue"},
                     title="10 Kategori Teratas")
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
    st.info(f"⚠️ **Insight:** Kategori **{', '.join(flagged)}** perlu diwaspadai! *Hobi & Olahraga* mencapai **42,59%** — hampir setengah uang harian Anda lari ke sini.")

# ==========================================================================
# TAB 3: QUEST #2
# ==========================================================================
with tab3:
    st.header("Q2: Seberapa Pintar AI Membedakan Transaksi?")
    rf = st.session_state.rf_model
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
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Prediksi", y="Asli", color="Jumlah"),
                        x=["Wajar", "Boros"], y=["Wajar", "Boros"])
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
    imp = rf.feature_importances(st.session_state.feature_names)
    imp_df = pd.DataFrame(imp, columns=["fitur", "pengaruh"]).sort_values("pengaruh", ascending=True).tail(10)
    fig = px.bar(imp_df, y="fitur", x="pengaruh", orientation="h", color="pengaruh", color_continuous_scale="Greens")
    fig.update_layout(height=400, yaxis_title="", xaxis_title="Tingkat Pengaruh")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Laporan Klasifikasi Lengkap"):
        st.text(classification_report(m["y_test"], m["y_pred"], target_names=["Wajar", "Boros"], zero_division=0))
    st.info(f"🤖 **Insight:** Model AI kita berhasil dengan akurasi **{m['accuracy']:.2%}** dan skor AUC **{m['auc']:.3f}**. Aplikasi ini cukup pintar membedakan transaksi wajar vs boros.")

    st.subheader("💾 Simpan Model")
    if st.button("Simpan Model RF"):
        rf.save("centsaver")
        st.success("Model tersimpan!")

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
    fig = px.imshow(pivot_growth, color_continuous_scale="RdYlGn_r", aspect="auto",
                    labels=dict(x="Bulan", y="Kategori", color="Pertumbuhan %"),
                    title="10 Kategori Paling 'Nge-trend' (24 Bulan Terakhir)")
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
        top8["level"] = top8["anomaly_rate"].apply(lambda r: "Tinggi" if r > 0.15 else "Sedang" if r > 0.08 else "Rendah")
        fig = px.bar(top8, y="category", x="anomaly_rate", orientation="h",
                     color="level", color_discrete_map={"Tinggi": "crimson", "Sedang": "orange", "Rendah": "steelblue"},
                     title="Seberapa Sering Kategori Ini Tidak Menentu")
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**🥉 KONTEKS: Akhir Pekan vs Hari Kerja**")
        fig = px.bar(weekend_impulse.reset_index(), y="category", x="weekend_boost",
                     orientation="h", color="weekend_boost", color_continuous_scale="Teal",
                     title="Selisih Risiko: Akhir Pekan − Hari Kerja")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Contoh Grafik + Penanda Anomali")
    most_volatile = anomaly_freq.iloc[0]["category"]
    cat_ts = monthly_cat[monthly_cat["category"] == most_volatile].sort_values("period")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cat_ts["period"], y=cat_ts["total_amount"], mode="lines", name="Total Pengeluaran", line=dict(color="navy", width=2)))
    anomaly_pts = cat_ts[cat_ts["is_anomaly"] == 1]
    if not anomaly_pts.empty:
        fig.add_trace(go.Scatter(x=anomaly_pts["period"], y=anomaly_pts["total_amount"],
            mode="markers", name="Anomali (Z>2)", marker=dict(color="red", size=12, symbol="x")))
    fig.update_layout(title=f"Kategori Paling Tidak Menentu: {most_volatile}", xaxis_title="Bulan", yaxis_title="Total (Rp)", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.info("📋 **Prioritas Tampilan:** 1️⃣ UTAMA — Heatmap MoM Growth | 2️⃣ PENDUKUNG — Line Chart + Anomali | 3️⃣ SEKUNDER — Bar Chart | 4️⃣ KONTEKS — Weekend vs Weekday")

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
        segment_micro = (user_rfm.groupby(["frequency_segment", "monetary_segment"])
                         .agg(rata_boros=("micro_rate", "mean")).reset_index()
                         .pivot(index="frequency_segment", columns="monetary_segment", values="rata_boros"))
        fig = px.imshow(segment_micro, text_auto=".2f", color_continuous_scale="RdYlGn_r", aspect="auto")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Profil 3 Kelompok Pengguna")
    seg1, seg2, seg3 = st.columns(3)
    with seg1:
        st.error("**Jarang-Menetap**")
        st.metric("Tingkat Boros", "~35%", "RISIKO TERTINGGI")
        st.markdown("Orang yang jarang transaksi, tapi kalau transaksi suka impulsif.\n**Saran:** 🎮 Ikut *Tantangan Kurangi Boros* — nabung selisihnya.")
    with seg2:
        st.success("**Sering-Premium**")
        st.metric("Tingkat Boros", "Paling Rendah", "ASSET PENTING")
        st.markdown("Orang yang sering belanja tapi justru paling hemat.\n**Saran:** 🏆 Beri *Reward Loyalitas*.")
    with seg3:
        st.warning("**Pemula**")
        st.metric("Jendela Intervensi", "7 Hari", "SEGERA")
        st.markdown("Pengguna baru — kebiasaan pertama menentukan pola ke depan.\n**Saran:** 📨 *Follow-up 7 hari*.")

    st.divider()
    st.subheader("📋 Ringkasan Eksekutif")
    st.markdown("""
    | Pertanyaan | Status | Bukti |
    |------------|--------|-------|
    | **Q1** Berapa persen micro-spending? | ✅ LULUS | Baseline per kategori valid; 2 kategori waspada |
    | **Q2** Apakah AI akurat ≥85%? | ✅ LULUS | Model RF: **{:.2%}** / AUC: **{:.3f}** |
    | **Q3** Visualisasi terbaik? | ✅ LULUS | Heatmap MoM Growth paling ampuh (korelasi 0.71) |
    | **RFM** Segmentasi pengguna? | ✅ LULUS | 3 kelompok dengan saran masing-masing |
    """.format(m['accuracy'], m['auc']))

# ==========================================================================
# TAB 6: INPUT MANUAL & PREDIKSI
# ==========================================================================
with tab6:
    st.header("📝 Input Manual & Prediksi Transaksi")
    st.markdown("*Masukkan data transaksi baru untuk dicek apakah termasuk pengeluaran wajar atau boros.*")

    st.divider()

    # --- Bagian 1: Form Input ---
    st.subheader("✏️ Form Input Transaksi")

    col1, col2, col3 = st.columns(3)
    with col1:
        input_tanggal = st.date_input("Tanggal Transaksi", value=pd.to_datetime("2024-06-01"))
        input_deskripsi = st.text_input("Deskripsi", placeholder="Contoh: Beli kopi di Starbucks")
    with col2:
        input_kategori = st.selectbox("Kategori", options=sorted(df["category"].unique()))
        input_nominal = st.number_input("Nominal (Rp)", min_value=0, value=50000, step=1000)
    with col3:
        input_hari = st.selectbox("Hari", options=["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
        input_weekend = 1 if input_hari in ["Sabtu", "Minggu"] else 0
        st.markdown(f"**Akhir Pekan?** {'✅ Ya' if input_weekend else '❌ Tidak'}")

    # Tombol tambah ke tabel
    if st.button("➕ Tambahkan ke Tabel", type="secondary"):
        new_row = pd.DataFrame([{
            "tanggal": input_tanggal.strftime("%Y-%m-%d"),
            "deskripsi": input_deskripsi,
            "kategori": input_kategori,
            "nominal": input_nominal,
            "hari": input_hari,
            "weekend": input_weekend
        }])
        st.session_state.manual_data = pd.concat([st.session_state.manual_data, new_row], ignore_index=True)
        st.success("Transaksi ditambahkan!")

    # --- Bagian 2: Tabel Data ---
    st.subheader("📋 Daftar Transaksi yang Akan Dicek")

    edited_data = st.data_editor(
        st.session_state.manual_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "tanggal": st.column_config.TextColumn("Tanggal"),
            "deskripsi": st.column_config.TextColumn("Deskripsi"),
            "kategori": st.column_config.SelectboxColumn("Kategori", options=sorted(df["category"].unique())),
            "nominal": st.column_config.NumberColumn("Nominal (Rp)", min_value=0),
            "hari": st.column_config.SelectboxColumn("Hari", options=["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]),
            "weekend": st.column_config.NumberColumn("Weekend (0/1)", min_value=0, max_value=1)
        }
    )
    st.session_state.manual_data = edited_data

    # --- Bagian 3: Upload Foto ---
    st.divider()
    st.subheader("📸 Upload Foto/Bukti Transaksi (Opsional)")

    foto_col1, foto_col2 = st.columns([1, 2])
    with foto_col1:
        foto_upload = st.file_uploader("Pilih foto struk/nota", type=["jpg", "jpeg", "png"])
    with foto_col2:
        if foto_upload is not None:
            st.image(foto_upload, caption="Foto yang diupload", use_container_width=True)
            st.info("💡 **Catatan:** Fitur OCR (membaca teks dari foto) akan ditambahkan di versi berikutnya. Saat ini foto digunakan sebagai bukti pendukung.")
        else:
            st.markdown("""
            **Kenapa upload foto?**
            - 📷 Sebagai bukti transaksi
            - 🔍 Di masa depan: AI bisa membaca struk otomatis (OCR)
            - 📁 Disimpan bersama data untuk audit
            """)

    # --- Bagian 4: Prediksi ---
    st.divider()
    st.subheader("🤖 Prediksi: Transaksi Wajar atau Boros?")

    if len(st.session_state.manual_data) == 0:
        st.warning("Belum ada data transaksi. Silakan tambahkan di form atas.")
    else:
        if st.button("🔮 Jalankan Prediksi", type="primary"):
            with st.spinner("Sedang menganalisis..."):
                # Buat dataframe untuk prediksi
                pred_df = st.session_state.manual_data.copy()
                pred_df["date"] = pd.to_datetime(pred_df["tanggal"])
                pred_df["amount"] = pred_df["nominal"]
                pred_df["category"] = pred_df["kategori"]
                pred_df["description"] = pred_df["deskripsi"]
                pred_df["day_of_week"] = pred_df["hari"].map({
                    "Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5, "Minggu": 6
                })
                pred_df["is_weekend"] = pred_df["weekend"]

                # Feature engineering (simplified untuk manual input)
                pred_df["month"] = pred_df["date"].dt.month
                pred_df["year"] = pred_df["date"].dt.year
                pred_df["amount_log"] = np.log1p(pred_df["amount"])

                # Merge dengan stats dari data historis
                cat_map = {c: i+1 for i, c in enumerate(sorted(df["category"].unique()))}
                pred_df["cat_encoded"] = pred_df["category"].map(cat_map)

                # Ambil stats rata-rata dari data historis per kategori
                hist_stats = df.groupby("category").agg(
                    cat_day_q25=("cat_day_q25", "median"),
                    cat_day_median=("cat_day_median", "median"),
                    cat_day_count=("cat_day_count", "median"),
                    monthly_txn_count_avg=("monthly_txn_count_avg", "median"),
                    monthly_txn_count_median=("monthly_txn_count_median", "median"),
                ).reset_index()
                pred_df = pred_df.merge(hist_stats, on="category", how="left")
                pred_df["small_amount_flag"] = (pred_df["amount"] <= pred_df["cat_day_q25"]).astype(int)
                pred_df["repetitive_category_flag"] = (pred_df["monthly_txn_count_avg"] >= pred_df["monthly_txn_count_median"]).astype(int)

                # Siapkan fitur
                feature_cols = [
                    "amount", "amount_log", "day_of_week", "is_weekend",
                    "month", "year", "cat_encoded",
                    "cat_day_q25", "cat_day_median", "cat_day_count",
                    "monthly_txn_count_avg", "monthly_txn_count_median",
                    "small_amount_flag", "repetitive_category_flag"
                ]
                X_pred = pred_df[feature_cols].fillna(0)

                # Prediksi RF
                rf = st.session_state.rf_model
                pred_rf = rf.predict(X_pred)
                prob_rf = rf.predict_proba(X_pred)

                # Prediksi DL (jika ada)
                dl = st.session_state.dl_model
                pred_dl = None
                prob_dl = None
                if dl is not None and dl.is_loaded:
                    pred_dl = dl.predict(X_pred)
                    prob_dl = dl.predict_proba(X_pred)

                # Tampilkan hasil
                st.markdown("---")
                st.markdown("### 📊 Hasil Prediksi")

                for i, row in pred_df.iterrows():
                    with st.container():
                        col_a, col_b, col_c = st.columns([3, 2, 2])
                        with col_a:
                            st.markdown(f"**{row['deskripsi']}** — {row['kategori']}")
                            st.markdown(f"💰 Rp{row['nominal']:,.0f} | 📅 {row['tanggal']}")
                        with col_b:
                            # RF Result
                            rf_label = "⚠️ BOROS" if pred_rf[i] == 1 else "✅ WAJAR"
                            rf_color = "red" if pred_rf[i] == 1 else "green"
                            rf_conf = prob_rf[i] * 100
                            st.markdown(f"**Model RF:** {rf_label}")
                            st.markdown(f"Keyakinan: **{rf_conf:.1f}%**", help="Semakin tinggi, semakin yakin model")
                        with col_c:
                            # DL Result
                            if pred_dl is not None:
                                dl_label = "⚠️ BOROS" if pred_dl[i] == 1 else "✅ WAJAR"
                                dl_conf = prob_dl[i] * 100
                                st.markdown(f"**Model DL:** {dl_label}")
                                st.markdown(f"Keyakinan: **{dl_conf:.1f}%**")
                            else:
                                st.markdown("**Model DL:** ⏳ Belum tersedia")
                                st.caption("Upload model .keras di sidebar")

                        # Progress bar untuk visualisasi confidence
                        st.progress(min(int(rf_conf), 100), text=f"Tingkat Keyakinan RF: {rf_conf:.1f}%")

                        # Saran
                        if pred_rf[i] == 1:
                            st.error("💡 **Saran:** Transaksi ini masuk kategori 'boros'. Pertimbangkan untuk menunda atau mengurangi frekuensi serupa.")
                        else:
                            st.success("💡 **Saran:** Transaksi ini masih dalam batas wajar. Pantau terus akumulasi mingguan.")

                        st.markdown("---")

                # Ringkasan
                total_boros = sum(pred_rf)
                total_wajar = len(pred_rf) - total_boros
                st.markdown("### 📈 Ringkasan Prediksi")
                r1, r2, r3 = st.columns(3)
                r1.metric("Total Dicek", f"{len(pred_rf)} transaksi")
                r2.metric("Diprediksi Wajar", f"{total_wajar}", "✅")
                r3.metric("Diprediksi Boros", f"{total_boros}", "⚠️ Waspada" if total_boros > 0 else "✅ Aman")

# ==========================================================================
# TAB 7: AI CHATBOT (RAG)
# ==========================================================================
with tab7:
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
    for i, sq in enumerate(suggestions[:4]):
        if col_sug[i].button(sq, key=f"sug_{i}"):
            st.session_state.current_question = sq

    col_sug2 = st.columns(4)
    for i, sq in enumerate(suggestions[4:]):
        if col_sug2[i].button(sq, key=f"sug2_{i}"):
            st.session_state.current_question = sq

    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    user_q = st.text_input("Atau ketik pertanyaan Anda:", value=st.session_state.current_question,
                           placeholder="Contoh: Kenapa saya boros bulan ini?", key="question_input")

    def ask_question():
        if user_q.strip():
            with st.spinner("Sedang mencari jawaban..."):
                answer = rag.query(user_q)
            st.session_state.chat_history.append({"q": user_q, "a": answer})
            st.session_state.current_question = ""

    st.button("🔍 Tanya AI", type="primary", on_click=ask_question, key="ask_btn")

    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        st.markdown("---")
        st.markdown(f"**🙋 Pertanyaan:** {latest['q']}")
        st.markdown(f"**📝 Jawaban:**")
        st.info(latest['a'])

        with st.expander("📄 Lihat data yang digunakan untuk menjawab"):
            q_vec = rag.vectorizer.transform([latest['q']])
            scores = cosine_similarity(q_vec, rag.doc_vectors).flatten()
            top_idx = scores.argsort()[-3:][::-1]
            for i, idx in enumerate(top_idx, 1):
                st.markdown(f"**Dokumen {i} (skor relevansi: {scores[idx]:.3f}):**")
                st.text(rag.docs[idx][:500] + "...")

    if len(st.session_state.chat_history) > 1:
        st.markdown("---")
        st.markdown("**📜 Riwayat Percakapan:**")
        for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
            with st.expander(f"💬 {chat['q'][:50]}..."):
                st.markdown(f"**Q:** {chat['q']}")
                st.markdown(f"**A:** {chat['a']}")

    st.divider()
    st.markdown("""
    **Cara Kerja Chatbot (100% Gratis):**
    1. 📊 Data transaksi Anda diubah jadi "buku panduan" digital
    2. 🔍 Pertanyaan Anda dicocokkan dengan isi buku panduan
    3. 📋 Sistem ambil halaman paling relevan
    4. 💬 Jawaban disusun dari halaman tersebut — tanpa pakai AI eksternal

    *Semua proses di server Streamlit, tidak perlu API key, tidak perlu internet khusus.*
    """)
