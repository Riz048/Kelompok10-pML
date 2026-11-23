import joblib
import pandas as pd
import streamlit as st

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
segmen = pd.read_csv("segmen.csv", index_col=0)

st.title("Segmentasi Pelanggan Toko")
st.write("K-Means Clustering (K=5) berdasarkan Annual Income dan Spending Score.")

income = st.number_input("Annual Income (k$)", min_value=0, value=60)
spending = st.slider("Spending Score (1-100)", 1, 100, 50)

if st.button("Prediksi"):
    X = pd.DataFrame(
        [[income, spending]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )
    cluster = int(model.predict(scaler.transform(X))[0])

    st.subheader(f"Cluster {cluster}")
    st.write(f"**Karakteristik:** {segmen.loc[cluster, 'Karakteristik']}")
    st.write(f"**Strategi Pemasaran:** {segmen.loc[cluster, 'Strategi Pemasaran']}")
