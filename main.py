import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

figure_num = 0


def normal_distribution():
    mean1 = [-1, -1]
    cov1 = [[1, 0.5], [0.5, 2]]
    n1 = 100
    mean2 = [2, 2]
    cov2 = [[2, 0.5], [0.5, 2]]
    n2 = 200
    x1, y1 = np.random.multivariate_normal(mean1, cov1, n1).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, n2).T
    plt.figure(get_incremented_fig_num())
    plt.plot(x1, y1, 'x')
    plt.plot(x2, y2, 'o')
    clust1 = np.array(list(zip(x1, y1)))
    clust2 = np.array(list(zip(x2, y2)))
    x_train = np.concatenate((clust1, clust2))
    distance, labels, centers = perform_kmeans_and_calculate_distance(x_train, 2)
    print_and_plot_result(x_train, distance, labels, centers)


def test_data():
    x, y, classes = read_data_from_file()
    num_classes = string_labels_to_num(classes)
    axis_labels = ['Protein', 'Oil']
    plot_data(x, y, num_classes, axis_labels)
    n_clusters = len(set(classes))
    x_train = np.array(list(zip(x, y)))
    distance, labels, centers = perform_kmeans_and_calculate_distance(x_train, n_clusters)
    print_and_plot_result(x_train, distance, labels, centers, axis_labels)


def string_labels_to_num(classes):
    unique_labels = list(set(classes))
    num_labels = []
    for i in range(len(classes)):
        num_labels.append(unique_labels.index(classes[i]))
    return np.array(num_labels)


def perform_kmeans_and_calculate_distance(x_train, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x_train)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    splitted_data = split_data_by_labels(labels, x_train)
    distance = score_by_centers(splitted_data, kmeans)
    return distance, labels, centers


def print_and_plot_result(x_train, distance, labels, centers, axis_labels=None):
    for i in range(len(distance)):
        print('Cluster: ', centers[i], ' Sum of squares of distances: ', -distance[i])
    plot_data(x_train[:, 0], x_train[:, 1], labels, axis_labels)
    for i in range(len(centers)):
        plt.plot(centers[i][0], centers[i][1], 'o', markersize=20)
    plt.show()


def get_incremented_fig_num():
    global figure_num
    figure_num += 1
    return figure_num


def score_by_centers(data, kmeans):
    distance = []
    for i in range(len(data)):
        distance.append(kmeans.score(np.array(data[i])))
    return distance


def split_data_by_labels(labels, data):
    clusters_amount = np.max(labels) + 1
    splitted_data = [[] for i in range(clusters_amount)]
    for i in range(len(data)):
        splitted_data[labels[i]].append(data[i])
    return np.array(splitted_data)


def plot_data(x, y, class_labels, axis_labels=None):
    axis_labels = ['x', 'y'] if axis_labels is None else axis_labels
    plt.figure(get_incremented_fig_num())
    plt.scatter(x, y, marker='x', c=class_labels)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])


def read_data_from_file():
    file = open("01-soybean-data.txt")
    protein = []
    oil = []
    loc = []
    for line in file:
        splitted_line = line.split()
        protein.append(float(splitted_line[9]))
        oil.append(float(splitted_line[10]))
        loc.append(splitted_line[2])
    file.close()
    return np.array(protein), np.array(oil), np.array(loc)


if __name__ == '__main__':
    normal_distribution()
    # test_data()
