# 💰 CentSaver — Streamlit Dashboard

**Capstone Project — DBS Foundation Coding Camp**  
*AI Engineering × Data Science Track*

End-to-end microspending detection dashboard yang menjawab 3 Business Questions:
1. **Q1:** Berapa persentase micro-spending vs total pengeluaran per bulan?
2. **Q2:** Apakah model klasifikasi mampu membedakan micro-spending dengan akurasi ≥85%?
3. **Q3:** Visualisasi mana yang paling signifikan memicu rekomendasi AI Chatbot?

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
├── app.py              # Main Streamlit dashboard (5 tabs)
├── utils.py            # Data loading, feature engineering, RFM, microspending ratio
├── inference.py        # Model wrapper (Random Forest + Deep Learning)
├── requirements.txt    # Python dependencies
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

---

## 🔧 Model Notes

- **Random Forest (Baseline):** Trained on-the-fly dengan fitur **anti-leakage** (tidak menggunakan `amount_ratio`/`amount_zscore` sebagai input model).
- **Deep Learning:** Load pre-trained `.keras` model via `inference.py` jika tersedia.
- **Scaler:** `StandardScaler` di-fit pada training split dan diterapkan pada test/inference.

---

## 🎯 Key Features

- ✅ **Category-aware baseline** untuk microspending detection
- ✅ **Anti-leakage feature engineering** untuk valid model evaluation
- ✅ **Interactive Plotly** visualizations (heatmap, line chart, bar chart)
- ✅ **Real-time accumulation tracker** per kategori
- ✅ **RFM-style segmentation** dengan action recommendations
- ✅ **Model export** (pickle joblib) untuk deployment

---

## 📝 Capstone Team

- **AI Engineering:** Deep Learning model (TensorFlow Functional API + Custom Components)
- **Data Science:** EDA, feature engineering, baseline Random Forest, Streamlit deployment
