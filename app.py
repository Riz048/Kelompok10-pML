import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Segmentasi Pelanggan Toko", layout="wide")

FITUR = ["Annual Income (k$)", "Spending Score (1-100)"]
WARNA = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]


@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    segmen = pd.read_csv("segmen.csv", index_col=0)
    return model, scaler, segmen


@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df["Cluster"] = model.predict(scaler.transform(df[FITUR]))
    return df


model, scaler, segmen = load_model()
df = load_data()

centroid = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FITUR)
centroid["Cluster"] = range(len(centroid))
nama_cluster = [f"Cluster {i}" for i in centroid["Cluster"]]

st.title("Segmentasi Pelanggan Toko")
st.caption("K-Means Clustering (K=5) berdasarkan Annual Income dan Spending Score · Dataset Mall Customers (200 pelanggan)")

col_input, col_hasil = st.columns([1, 2], gap="large")

with col_input:
    st.subheader("Data Pelanggan")
    income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=60)
    spending = st.slider("Spending Score (1-100)", 1, 100, 50)

X = pd.DataFrame([[income, spending]], columns=FITUR)
cluster = int(model.predict(scaler.transform(X))[0])
anggota = int((df["Cluster"] == cluster).sum())

with col_hasil:
    st.subheader("Hasil Segmentasi")
    m1, m2, m3 = st.columns(3)
    m1.metric("Cluster", cluster)
    m2.metric("Anggota cluster", f"{anggota} pelanggan")
    m3.metric("Porsi dari dataset", f"{anggota / len(df):.0%}")
    st.markdown(f"**Karakteristik:** {segmen.loc[cluster, 'Karakteristik']}")
    st.success(f"**Strategi Pemasaran:** {segmen.loc[cluster, 'Strategi Pemasaran']}")

st.divider()

tab_plot, tab_profil = st.tabs(["Visualisasi Cluster", "Profil Setiap Cluster"])

with tab_plot:
    skala = alt.Scale(domain=nama_cluster, range=WARNA)
    data_plot = df.assign(Label=df["Cluster"].map(lambda c: f"Cluster {c}"))

    titik = alt.Chart(data_plot).mark_circle(size=70, opacity=0.8).encode(
        x=alt.X("Annual Income (k$):Q", scale=alt.Scale(zero=False)),
        y=alt.Y("Spending Score (1-100):Q"),
        color=alt.Color("Label:N", scale=skala, legend=alt.Legend(title=None, orient="top")),
        tooltip=["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Label"],
    )

    pusat = alt.Chart(centroid.assign(Label=nama_cluster)).mark_point(
        shape="cross", size=250, filled=True, strokeWidth=1
    ).encode(
        x="Annual Income (k$):Q",
        y="Spending Score (1-100):Q",
        color=alt.Color("Label:N", scale=skala, legend=None),
        tooltip=[alt.Tooltip("Label", title="Centroid"),
                 alt.Tooltip("Annual Income (k$)", format=".1f"),
                 alt.Tooltip("Spending Score (1-100)", format=".1f")],
    )

    input_user = alt.Chart(X).mark_point(
        shape="diamond", size=400, filled=True, color="#f0efec", stroke="#1a1a19", strokeWidth=2
    ).encode(
        x="Annual Income (k$):Q",
        y="Spending Score (1-100):Q",
        tooltip=[alt.Tooltip("Annual Income (k$)", title="Input income"),
                 alt.Tooltip("Spending Score (1-100)", title="Input spending")],
    )

    st.altair_chart((titik + pusat + input_user).properties(height=480), width="stretch")
    st.caption("Lingkaran = pelanggan · Tanda + = centroid cluster · Belah ketupat putih = data yang dimasukkan")

with tab_profil:
    profil = df.groupby("Cluster").agg(
        Jumlah=("CustomerID", "count"),
        Rata_Income=("Annual Income (k$)", "mean"),
        Rata_Spending=("Spending Score (1-100)", "mean"),
        Rata_Usia=("Age", "mean"),
    ).round(1)
    profil = profil.join(segmen)
    profil.columns = ["Jumlah Pelanggan", "Rata-rata Income (k$)", "Rata-rata Spending",
                      "Rata-rata Usia", "Karakteristik", "Strategi Pemasaran"]
    st.dataframe(profil, width="stretch")
