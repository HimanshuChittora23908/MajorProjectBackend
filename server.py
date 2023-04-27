import pandas as pd
from flask import Flask, request, jsonify
from flask import jsonify
import numpy as np
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from statistics import mode

app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def hello():
    # Receive the data from the client where content-type is multipart/form-data
    data = request.files["file"]
    data = pd.read_csv(data, sep=",")
    data = data.iloc[:, :]
    data = np.array(data, dtype="float32")
    ROI_L = 0
    ROI_R = 0
    series = data[:, 1:16000]
    N_CLUSTERS = 8
    model = TimeSeriesKMeans(
        n_clusters=N_CLUSTERS,
        metric="euclidean",
        max_iter=50,
        random_state=41,
        init="random",
    )
    y_pred = model.fit_predict(series)
    max_cluster = 0
    max_cluster_size = 0
    for i in range(N_CLUSTERS):
        if max_cluster_size < np.sum(y_pred == i):
            max_cluster = i
            max_cluster_size = np.sum(y_pred == i)
    min_dist = np.inf
    min_dist_col = 0
    max_cluster = np.argmax(np.bincount(model.labels_))
    for i in range(series.shape[1]):
        dist = np.linalg.norm(series[:, i] - model.cluster_centers_[max_cluster])
        if dist < min_dist:
            min_dist = dist
            min_dist_col = i

    response = jsonify(
        {
            "expected_graph": min_dist_col,
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
