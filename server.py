import pandas as pd
from flask import Flask, request, jsonify
from flask import jsonify
import numpy as np
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from statistics import mode

app = Flask(__name__)

model = None
series = None
N_CLUSTERS = 8
# Create an array named labels to store the labels of the clusters
labels = np.zeros(N_CLUSTERS, dtype="int32")


@app.route("/upload", methods=["POST"])
def hello():
    global model
    global series
    global N_CLUSTERS

    # Receive the data from the client where content-type is multipart/form-data
    data = request.files["file"]
    data = pd.read_csv(data, sep=",")
    data = data.iloc[:, :]
    data = np.array(data, dtype="float32")
    series = data[:, 1:]
    model = TimeSeriesKMeans(
        n_clusters=N_CLUSTERS,
        metric="euclidean",
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
    min_dist_row = 0
    max_cluster = np.argmax(np.bincount(model.labels_))
    for i in range(series.shape[0]):
        if model.labels_[i] == max_cluster:
            dist = np.linalg.norm(series[i] - model.cluster_centers_[max_cluster])
            if dist < min_dist:
                min_dist = dist
                min_dist_row = i

    response = jsonify(
        {
            "expected_graph": min_dist_row,
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/getFarthestGraph", methods=["GET"])
def getFarthestGraph():
    max_dist = 0
    max_dist_row = 0
    graph_id = request.args.get("graph_id")
    max_cluster = np.argmax(np.bincount(model.labels_))
    for i in range(series.shape[0]):
        if model.labels_[i] == int(graph_id):
            dist = np.linalg.norm(series[i] - model.cluster_centers_[max_cluster])
            if dist > max_dist:
                max_dist = dist
                max_dist_row = i

    response = jsonify(
        {
            "farthest_graph": max_dist_row,
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/getClosestGraph", methods=["GET"])
def getClosestGraph():
    min_dist = np.inf
    min_dist_row = 0
    graph_id = request.args.get("graph_id")
    max_cluster = np.argmax(np.bincount(model.labels_))
    for i in range(series.shape[0]):
        if model.labels_[i] == int(graph_id):
            dist = np.linalg.norm(series[i] - model.cluster_centers_[max_cluster])
            if dist < min_dist:
                min_dist = dist
                min_dist_row = i

    response = jsonify(
        {
            "closest_graph": min_dist_row,
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/labelTrue", methods=["GET"])
def labelTrue():
    global labels
    print(labels)
    graph_id = request.args.get("graph_id")
    labels[int(graph_id)] = 1
    response = jsonify(
        {
            "success": True,
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    print(labels)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
