import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

st.set_page_config(page_title="Klasifikasi Hujan Kolkata", layout="wide")

@st.cache_resource
def load():
    m = XGBClassifier()
    m.load_model('model_xgb.json')
    return m, joblib.load('scaler.pkl')

model, scaler = load()
data = pd.read_csv('data_dashboard.csv')
metrik = pd.read_csv('metrik_evaluasi.csv', index_col=0)
fitur = [c for c in data.columns if c != 'target']

st.title("Klasifikasi Hujan Harian Kolkata dengan XGBoost")
st.write("Dashboard ini memprediksi apakah suatu hari **hujan** atau **tidak hujan** "
         "berdasarkan suhu, angin, dan durasi sinar matahari. Dataset: Kolkata Climate "
         "and Weather 2021-2025 (Kaggle).")

st.header("Ringkasan Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah data", len(data))
c2.metric("Jumlah fitur", len(fitur))
c3.metric("Hujan / Tidak Hujan", f"{(data.target==1).sum()} / {(data.target==0).sum()}")
st.dataframe(data.head())

st.header("Eksplorasi Data (EDA)")
a, b = st.columns(2)
a.image('eda_histogram.png', caption='Distribusi fitur')
b.image('eda_countplot.png', caption='Proporsi kelas target')
st.image('eda_heatmap.png', caption='Heatmap korelasi')

st.header("Performa Model")
st.table(metrik)
a, b = st.columns(2)
a.image('hasil_confusion_matrix.png', caption='Confusion matrix')
b.image('hasil_roc_curve.png', caption='ROC curve')
st.image('hasil_feature_importance.png', caption='Feature importance')

st.header("Coba Prediksi Manual")
with st.form("form"):
    nilai = {}
    cols = st.columns(2)
    for i, f in enumerate(fitur):
        nilai[f] = cols[i % 2].number_input(
            f, value=float(round(data[f].median(), 2)))
    kirim = st.form_submit_button("Prediksi")
if kirim:
    x = scaler.transform(pd.DataFrame([nilai])[fitur])
    p = model.predict_proba(x)[0, 1]
    if p >= 0.5:
        st.success(f"Prediksi: HUJAN (peluang {p:.1%})")
    else:
        st.info(f"Prediksi: TIDAK HUJAN (peluang hujan {p:.1%})")
