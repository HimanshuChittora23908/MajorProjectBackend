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
series_backup = None
cluster_series = None
outlier_indices = None
original_length = 0
N_CLUSTERS = 3
CLUSTER_INDEX = 3
# Create an array named labels to store the labels of the clusters and initialize it with -1
labels = np.zeros(N_CLUSTERS, dtype="int32")
indexes_array = []
indexes_array_backup = []
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
    global indexes_array
    global reference_graph
    global indexes_array_backup
    global series_backup

    labels = np.zeros(N_CLUSTERS, dtype="int32")
    # labels = np.full(N_CLUSTERS, -1)
    CLUSTER_INDEX = N_CLUSTERS

    # Receive the data from the client where content-type is multipart/form-data
    data = request.files["file"]
    data = pd.read_csv(data, sep=",")
    data = data.iloc[:, :]

    # Push all indices into indexes_array
    for i in range(data.shape[0]):
        indexes_array.append(i)

    data = np.array(data, dtype="float32")
    series = data[:, :]

    # Calculate derivative of the series data retaining the same shape
    series = np.gradient(series, axis=0)

    series = series - np.mean(series, axis=0)
    series = series / np.std(series, axis=0)

    series_backup = series
    indexes_array_backup = indexes_array

    # Find the peaks
    peaks, _ = find_peaks(series[:, 0], height=0.1, distance=10)  # redundant
    graph_less_than_4_peaks = []
    for i in range(series.shape[0]):
        peaks, _ = find_peaks(series[i, :], height=0.1, distance=10)
        if len(peaks) < 4:
            graph_less_than_4_peaks.append(i)
    series = np.delete(series, np.flipud(graph_less_than_4_peaks), axis=0)
    # Remove peak indices from indexes_array
    for i in range(len(graph_less_than_4_peaks)):
        indexes_array.remove(graph_less_than_4_peaks[i])

    # Remove outliers
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
    series_temp = np.zeros((len(indexes_array), series.shape[1]))
    for i in range(len(indexes_array)):
        series_temp[i] = series_backup[indexes_array[i]]
    model.fit(series_temp)
    outlier_indices = np.where(model.fit_predict(series_temp) == -1)[0]
    indexes_array_backup = list(indexes_array_backup)
    for i in range(len(outlier_indices)):
        indexes_array_backup.remove(indexes_array[outlier_indices[i]])

    indexes_array = list(set(indexes_array) & set(indexes_array_backup))

    model = TimeSeriesKMeans(
        n_clusters=N_CLUSTERS,
        metric="euclidean",
    )
    y_pred = model.fit_predict(series_backup[indexes_array])
    max_cluster = 0
    max_cluster_size = 0
    for i in range(N_CLUSTERS):
        if max_cluster_size < np.sum(y_pred == i):
            max_cluster = i
            max_cluster_size = np.sum(y_pred == i)

    min_dist = np.inf
    min_dist_row = 0
    max_cluster = np.argmax(np.bincount(model.labels_))
    for i in range(len(indexes_array)):
        if model.labels_[i] == max_cluster:
            dist = np.linalg.norm(
                series_backup[i] - model.cluster_centers_[max_cluster]
            )
            if dist < min_dist:
                min_dist = dist
                min_dist_row = i

    reference_graph = series[min_dist_row]

    peaks, properties = find_peaks(series[min_dist_row], height=0.1, distance=10)

    valleys, properties2 = find_peaks(-series[min_dist_row], height=0.1, distance=10)

    response = jsonify(
        {
            "expected_graph": indexes_array[min_dist_row],
            "peaks": peaks.tolist(),
            "peak_heights": properties["peak_heights"].tolist(),
            "valleys": valleys.tolist(),
            "valley_heights": properties2["peak_heights"].tolist(),
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
    cluster_series = series[np.where(model.labels_ == int(cluster_no))[0]]

    # store the indices of the elements where label is equal to cluster_no
    indices = np.where(model.labels_ == int(cluster_no))[0]

    cluster_model = TimeSeriesKMeans(
        n_clusters=2,
        metric="euclidean",
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
    max_dist = 0
    max_dist_row = -1
    graph_id = request.args.get("graph_id")
    max_cluster = np.argmax(np.bincount(model.labels_))
    for i in range(
        len(model.labels_)
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
            "farthest_graph": max_dist_row
            if max_dist_row == -1
            else indexes_array[max_dist_row],
            "peaks": peaks.tolist(),
            "peak_heights": properties["peak_heights"].tolist(),
            "valleys": valleys.tolist(),
            "valley_heights": properties2["peak_heights"].tolist(),
            "euclidean_dist": json.dumps(str(dist)),
        }
    )
    print(labels)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/getClosestGraph", methods=["GET"])
def getClosestGraph():
    min_dist = np.inf
    min_dist_row = 0
    graph_id = request.args.get("graph_id")
    max_cluster = np.argmax(np.bincount(model.labels_))
    for i in range(
        len(model.labels_)
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
            "peak_heights": properties["peak_heights"].tolist(),
            "valleys": valleys.tolist(),
            "valley_heights": properties2["peak_heights"].tolist(),
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
    global indexes_array
    global original_length
    global series_backup

    # Initialize cluster_id array with an array of -1 of length equal to the original length of the series
    cluster_id = np.full(len(series_backup), -1)
    print(model.labels_)
    for i in range(len(cluster_id)):
        if i in indexes_array:
            # label that cluster as 1 if the cluster has label 1 else label it as 0
            if labels[model.labels_[indexes_array.index(i)]] == 1:
                cluster_id[i] = 1
            else:
                cluster_id[i] = 0

    print("Labels: ", labels)
    print("Cluster ID: ", cluster_id)
    # print(indexes_array)

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
