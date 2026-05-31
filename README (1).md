# 💰 CentSaver — Streamlit Dashboard

**Capstone Project — DBS Foundation Coding Camp**  
*AI Engineering × Data Science Track*

End-to-end microspending detection dashboard answering 3 Business Questions:
1. **Q1:** Berapa persentase micro-spending vs total pengeluaran per bulan?
2. **Q2:** Apakah model klasifikasi mampu membedakan micro-spending dengan akurasi ≥85%?
3. **Q3:** Visualisasi mana yang paling signifikan memicu rekomendasi AI Chatbot?

**Bonus:** 🤖 AI Chatbot dengan RAG (Retrieval-Augmented Generation) untuk tanya-jawab insight data.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit
```bash
streamlit run app.py
```

### 3. Upload Dataset
Upload file `centsaver_master_relabelling.csv` melalui sidebar.

---

## 📁 File Structure

```
.
├── app.py              # Main Streamlit dashboard (6 tabs)
├── utils.py            # Data loading, feature engineering, RFM, microspending ratio
├── inference.py        # Random Forest predictor (lightweight, no TensorFlow)
├── rag_chatbot.py      # RAG module: TF-IDF (gratis) + OpenAI (premium)
├── requirements.txt    # Minimal Python dependencies
└── README.md           # This file
```

---

## 📊 Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Overview & EDA** | Category profiling, temporal trends, weekend impulse analysis |
| **Quest #1** | Microspending ratio per category, accumulation tracker, flagged categories |
| **Quest #2** | Classification performance (RF baseline), confusion matrix, ROC, feature importance |
| **Quest #3** | MoM Growth Heatmap, anomaly detection, weekend boost, Chatbot trigger logic |
| **RFM & Recommendations** | User segmentation, business action items, executive summary |
| **🤖 AI Chatbot (RAG)** | Tanya-jawab insight data dengan Retrieval-Augmented Generation |

---

## 🤖 RAG Chatbot

Fitur bonus untuk tanya-jawab natural language tentang data micro-spending.

### Mode Gratis (Default)
- **TF-IDF + Cosine Similarity** untuk retrieve dokumen relevan
- **Template-based response generation** — cepat, tidak perlu API key
- Cukup upload data dan langsung tanya

### Mode Premium (Opsional)
- **OpenAI GPT-3.5 Turbo** untuk respons lebih natural
- Masukkan API key di tab Chatbot
- Memerlukan koneksi internet dan API credit

### Contoh Pertanyaan
- *"Berapa rata-rata micro-spending per bulan?"*
- *"Kategori apa yang paling boros?"*
- *"Apakah model sudah memenuhi target akurasi?"*
- *"Kenapa akhir pekan saya boros?"*
- *"Apa rekomendasi untuk mengurangi micro-spending?"*

---

## 🔧 Model Notes

- **Random Forest (Baseline):** Trained on-the-fly dengan fitur **anti-leakage** (tidak menggunakan `amount_ratio`/`amount_zscore` sebagai input model).
- **Deep Learning:** Inference placeholder tersedia di `inference.py` jika model `.keras` sudah dilatih terpisah.
- **RAG:** Knowledge base dibangun secara otomatis dari data yang di-upload. Tidak perlu training.

---

## 🎯 Key Features

- ✅ **Category-aware baseline** untuk microspending detection
- ✅ **Anti-leakage feature engineering** untuk valid model evaluation
- ✅ **Interactive Plotly** visualizations (heatmap, line chart, bar chart)
- ✅ **Real-time accumulation tracker** per kategori
- ✅ **RFM-style segmentation** dengan action recommendations
- ✅ **RAG Chatbot** untuk tanya-jawab insight data (gratis & premium)
- ✅ **Model export** (joblib) untuk deployment

---

## 📝 Capstone Team

- **AI Engineering:** Deep Learning model (TensorFlow Functional API + Custom Components)
- **Data Science:** EDA, feature engineering, baseline Random Forest, RAG Chatbot, Streamlit deployment
