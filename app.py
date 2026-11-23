import gradio as gr
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
segmen = pd.read_csv("segmen.csv", index_col=0)


def prediksi(income, spending):
    X = pd.DataFrame(
        [[income, spending]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )
    cluster = int(model.predict(scaler.transform(X))[0])

    return (
        f"Cluster {cluster}",
        segmen.loc[cluster, "Karakteristik"],
        segmen.loc[cluster, "Strategi Pemasaran"],
    )


demo = gr.Interface(
    fn=prediksi,
    inputs=[
        gr.Number(label="Annual Income (k$)", value=60, minimum=0),
        gr.Slider(1, 100, step=1, value=50, label="Spending Score (1-100)"),
    ],
    outputs=[
        gr.Textbox(label="Cluster"),
        gr.Textbox(label="Karakteristik"),
        gr.Textbox(label="Strategi Pemasaran"),
    ],
    title="Segmentasi Pelanggan Toko",
    description="K-Means Clustering (K=5) berdasarkan Annual Income dan Spending Score.",
    examples=[[20, 80], [20, 10], [55, 50], [90, 85], [90, 15]],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
