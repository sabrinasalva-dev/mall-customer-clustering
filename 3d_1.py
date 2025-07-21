import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.offline as pyo

# Baca dataset
df = pd.read_csv("/Users/mac/Documents/Project 1/Mall_Customers.csv")

# Ambil fitur yang akan digunakan untuk clustering
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Lakukan KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(X)

# Tambahkan kolom cluster ke dataframe
df["cluster"] = clusters

# Fungsi untuk membuat objek Scatter3d untuk setiap cluster
def tracer(db, n, name):
    return go.Scatter3d(
        x=db[db["cluster"] == n]["Age"],
        y=db[db["cluster"] == n]["Spending Score (1-100)"],
        z=db[db["cluster"] == n]["Annual Income (k$)"],
        mode="markers",
        name=name,
        marker=dict(size=5)
    )

# Buat trace untuk setiap cluster
tracers = [tracer(df, i, f"Cluster {i}") for i in range(5)]

# Buat layout 3D-nya
layout = go.Layout(
    title="Hasil Klastering K-Means (3D)",
    scene=dict(
        xaxis=dict(title="Age"),
        yaxis=dict(title="Spending Score (1-100)"),
        zaxis=dict(title="Annual Income (k$)")
    )
)

# Gabungkan data dan layout menjadi satu figure
fig = go.Figure(data=tracers, layout=layout)

# Tampilkan hasil plot di browser
pyo.plot(fig, filename="/Users/mac/Documents/Project 1/kmeans_3d.html")
