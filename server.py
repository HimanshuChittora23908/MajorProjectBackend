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
cluster_series = None
N_CLUSTERS = 8
CLUSTER_INDEX = 8
# Create an array named labels to store the labels of the clusters and initialize it with -1
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


@app.route("/furtherCluster", methods=["POST"])
def furtherCluster():
    global model
    global series
    global cluster_series
    global CLUSTER_INDEX

    # receive integer parameter from request
    cluster_no = request.args.get("cluster_no")

    # further cluster the cluster represented by cluster_no into 2 clusters
    cluster_series = series[model.labels_ == int(cluster_no)]

    # store the indices of the elements where label is equal to cluster_no
    indices = np.where(model.labels_ == int(cluster_no))[0]

    cluster_model = TimeSeriesKMeans(
        n_clusters=2,
        metric="euclidean",
        random_state=41,
        init="random",
    )
    cluster_y_pred = cluster_model.fit_predict(cluster_series)

    # update the labels array
    for i in range(cluster_y_pred.shape[0]):
        if cluster_y_pred[i] == 0:
            model.labels_[indices[i]] = cluster_no
        else:
            model.labels_[indices[i]] = CLUSTER_INDEX

    CLUSTER_INDEX += 1
    labels.resize(CLUSTER_INDEX, refcheck=False)

    response = jsonify(
        {
            "message": "success",
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/getNoOfClusters", methods=["GET"])
def getNoOfClusters():
    # return the number of clusters by counting the distinct elements in model.labels_
    response = jsonify(
        {
            "no_of_clusters": len(np.unique(model.labels_)),
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
    for i in range(
        series.shape[0]
    ):  # label of element `i` is stored in model.labels_[i]
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
    for i in range(
        series.shape[0]
    ):  # label of element `i` is stored in model.labels_[i]
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
    graph_id = request.args.get("graph_id")
    labels[int(graph_id)] = 1  # labelling true
    response = jsonify(
        {
            "success": True,
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    print(labels)
    return response


@app.route("/labelFalse", methods=["GET"])
def labelFalse():
    global labels
    graph_id = request.args.get("graph_id")
    labels[int(graph_id)] = 0  # labelling false
    response = jsonify(
        {
            "success": True,
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    print(labels)
    return response


@app.route("/getSeries", methods=["GET"])
def getSeries():
    global series
    global model
    global labels
    global cluster_series
    global CLUSTER_INDEX

    # return the number of clusters by counting the distinct elements in model.labels_
    response = jsonify(
        {
            "series": series.tolist() if series is not None else [],
        }
    )
    print(len(series))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# route to get the label as true if cluster has label 1 with there graph id
@app.route("/getLabelGraphId", methods=["GET"])
def getLabelGraphId():
    global labels
    response = jsonify(
        {
            "labels": labels.tolist() if labels is not None else [],
            "cluster_id": model.labels_.tolist() if model.labels_ is not None else [],
            "graph_id": np.arange(len(labels)).tolist(),
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
