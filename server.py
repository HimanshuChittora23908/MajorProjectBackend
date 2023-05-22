import pandas as pd
from flask import Flask, request, jsonify
from flask import jsonify
import numpy as np
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from statistics import mode
from sklearn.neighbors import LocalOutlierFactor
from scipy.signal import find_peaks
import json

app = Flask(__name__)

model = None
series = None
cluster_series = None
outlier_indices = None
original_length = 0
N_CLUSTERS = 3
CLUSTER_INDEX = 3
# Create an array named labels to store the labels of the clusters and initialize it with -1
labels = np.zeros(N_CLUSTERS, dtype="int32")
reference_graph = None


@app.route("/upload", methods=["POST"])
def hello():
    global model
    global series
    global N_CLUSTERS
    global outlier_indices
    global original_length
    global labels
    global CLUSTER_INDEX
    global reference_graph

    labels = np.zeros(N_CLUSTERS, dtype="int32")
    CLUSTER_INDEX = N_CLUSTERS

    # Receive the data from the client where content-type is multipart/form-data
    data = request.files["file"]
    data = pd.read_csv(data, sep=",")
    data = data.iloc[:, :]

    # normalized_data = (data - data.mean()) / data.std()
    # normalized_data.dropna(inplace=True)

    # model = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
    # model.fit(normalized_data)
    # outlier_status = model.fit_predict(normalized_data)
    # outlier_indices = normalized_data.index[outlier_status == -1]

    data = np.array(data, dtype="float32")
    series = data[:, 1:]

    series = np.diff(series, axis=0)

    series = series - np.mean(series, axis=0)
    series = series / np.std(series, axis=0)

    # Find the peaks
    peaks, _ = find_peaks(series[:, 0], height=0.1, distance=10) # redundant
    graph_less_than_4_peaks = []
    for i in range(series.shape[0]):
        peaks, _ = find_peaks(series[i, :], height=0.1, distance=10)
        if len(peaks) < 4:
            graph_less_than_4_peaks.append(i)
    series = np.delete(series, np.flipud(graph_less_than_4_peaks), axis=0)
    print("After peaks removal: ", len(series))

    model = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
    outlier_status = model.fit_predict(series)
    outlier_indices = np.where(outlier_status == -1)[0]
    original_length = series.shape[0]
    series = np.delete(series, outlier_indices, axis=0)
    print("After outliers removal: ", len(series))

    model = TimeSeriesKMeans(
        n_clusters=N_CLUSTERS,
        metric="euclidean",
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

    reference_graph = series[min_dist_row]

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
    print(labels)
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
    print(labels)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/getFarthestGraph", methods=["GET"])
def getFarthestGraph():
    print("getFarthestGraph call: ", len(series))
    max_dist = 0
    max_dist_row = -1
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

    # farthest_graph stored at index max_dist_row
    peaks, properties = find_peaks(series[max_dist_row], height=0.1, distance=10)

    # find valleys at index max_dist_row with depth at least 0.1
    valleys, properties2 = find_peaks(-series[max_dist_row], height=0.1, distance=10)

    # calculate the euclidean distance between the farthest_graph and the reference_graph
    dist = np.linalg.norm(series[max_dist_row] - reference_graph)

    # send peaks and properties along with farthest_graph to frontend
    response = jsonify(
        {
            "farthest_graph": max_dist_row,
            "peaks": peaks.tolist(),
            "peak_heights": properties['peak_heights'].tolist(),
            "valleys": valleys.tolist(),
            "valley_heights": properties2['peak_heights'].tolist(),
            "euclidean_dist": json.dumps(str(dist)),
        }

    )
    print(labels)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/getClosestGraph", methods=["GET"])
def getClosestGraph():
    print("getClosestGraph call: ", len(series))
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

    # closest_graph stored at index min_dist_row
    peaks, properties = find_peaks(series[min_dist_row], height=0.1, distance=10)

    # find valleys at index min_dist_row with depth at least 0.1
    valleys, properties2 = find_peaks(-series[min_dist_row], height=0.1, distance=10)

    # calculate the euclidean distance between the farthest_graph and the reference_graph
    dist = np.linalg.norm(series[min_dist_row] - reference_graph)

    # send peaks and properties along with closest_graph to frontend
    response = jsonify(
        {
            "closest_graph": min_dist_row,
            "peaks": peaks.tolist(),
            "peak_heights": properties['peak_heights'].tolist(),
            "valleys": valleys.tolist(),
            "valley_heights": properties2['peak_heights'].tolist(),
            "euclidean_dist": json.dumps(str(dist)),
        }
    )
    print(labels)
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
    print(labels)
    response.headers.add("Access-Control-Allow-Origin", "*")
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
    print(labels)
    response.headers.add("Access-Control-Allow-Origin", "*")
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
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# route to get the label as true if cluster has label 1 with there graph id
@app.route("/getLabelGraphId", methods=["GET"])
def getLabelGraphId():
    global labels
    # series array has some removed elements that are present in outlier_indices and the clustering was done on this series array
    # generate cluster_id array using model.labels_ such that cluster_id[i] = -1 if element i is present in outlier_indices and the remaining elements with model.labels_[i], where i is not present in outlier_indices
    cluster_id = np.zeros(original_length, dtype=int)
    n_outliers = 0
    for i in range(original_length):
        if i in outlier_indices:
            cluster_id[i] = -1
            n_outliers += 1
        else:
            cluster_id[i] = model.labels_[i - n_outliers]

    response = jsonify(
        {
            "labels": labels.tolist() if labels is not None else [],
            "cluster_id": cluster_id.tolist(),
            "graph_id": np.arange(len(labels)).tolist(),
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
