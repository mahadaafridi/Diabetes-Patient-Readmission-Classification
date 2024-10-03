import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# class KNNClassifier:

#     def __init__(self, k = 5, ) -> None:
#         self._fitted = False
#         self._X = None
#         self._y = None
#         self._k = k

#     # Trains the model on features X and labels y
#     def fit(self, X, y) -> None:
#         if X.shape[0] != y.shape[0]:
#             raise Exception("Lengths of features and labels don't match")
#         try:
#             self._X = X.toarray()
#             self._y = y

#             # fitting stuff

#             self._fitted = True
#         except Exception as e:
#             self._X = None
#             self._y = None
#             raise e

#     # Returns the predicted labels based on input features
#     def predict(self, X) -> str:   # 0: NO, 1: <30, 2:>30?
#         X = X.toarray()
#         y = np.array([])
#         # Calculate the KNN for each datapoint
#         for X_new in X:
#             # Calculate distance from each point in self._X
#             distances = np.array([self._calculate_distance(X_new, X_old) for X_old in self._X])

#             # Sort by distance and choose the K nearest
#             sort_by_dist = np.argsort(distances)
#             sorted_y = self._y[sort_by_dist]
#             knn = {
#                 0: 0,
#                 1: 0,
#                 2: 0
#             }
#             for i in range(min(self._k, len(sorted_y))):
#                 knn[sorted_y[i]]+= 1
            
#             # Choose 
#             y = np.append(y, max(knn, key=knn.get))
        
#         return y
            
#     # Calculates the accuracy of label predictions
#     def accuracy_score(self, y_true, y_pred):
#         return sum([y_true[i] == y_pred[i] for i in range(len(y_true))]) / len(y_true)

#     # Calculates the Euclidean distance between two points
#     def _calculate_distance(self, X1, X2) -> float:
#         distance_squared = 0
#         for X1_i, X2_i in zip(X1, X2):
#             distance_squared+= (X1_i - X2_i) ** 2
#         return math.sqrt(distance_squared)






def test_hyperparameters(diabetes_130_us_hospitals_for_years_1999_2008):

    X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
    y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

    enc = OneHotEncoder(handle_unknown='ignore')

    mapping = {'NO': 0, '<30': 1, '>30': 2}
    y['readmitted'] = y['readmitted'].replace(mapping)
    

    X_train, X_test, y_train, y_test = train_test_split(enc.fit_transform(X), y, test_size=0.3)
    # flatten y to pass into model
    y_train_flat = np.array(y_train).flatten()

    k_values = [1, 5, 20, 100, 500, 2000]
    weights = ["uniform", "distance"]

    results = []

    best_accuracy = 0
    best_params = None

    for weight in weights:
        for k in k_values:
            hyper_parameters = dict(
                n_neighbors=k,
                weights=weight
            )
            print("TESTING WITH " + str(hyper_parameters))
            
            clf = KNeighborsClassifier(**hyper_parameters)
            clf.fit(X_train, y_train_flat)
            
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = hyper_parameters
            
            print(accuracy)
            print(hyper_parameters)
            print()

            results.append(dict(
                **hyper_parameters,
                accuracy=accuracy
            ))

    print("\nBEST:")
    print(best_params)
    print(best_accuracy)

    return results


def test_hyperparameters_and_plot(diabetes_130_us_hospitals_for_years_1999_2008):
    results = test_hyperparameters(diabetes_130_us_hospitals_for_years_1999_2008)
    distance_results = filter(lambda result : result["weights"] == "distance", results)
    distance_accuracies = [result["accuracy"] for result in distance_results]
    uniform_results = filter(lambda result : result["weights"] == "uniform", results)
    uniform_accuracies = [result["accuracy"] for result in uniform_results]

    print(list(distance_results))
    print(distance_accuracies)
    distance_results = filter(lambda result : result["weights"] == "distance", results)
    neighbors = [result["n_neighbors"] for result in distance_results]
    print(neighbors)

    fig, ax = plt.subplots(layout='constrained')

    locations = np.arange(len(distance_accuracies))
    width = 1 / (len(distance_accuracies) + 1)
    pos = 0

    # distance
    ax.bar(locations + pos, distance_accuracies, width, label="distance")

    # uniform
    pos+= width
    ax.bar(locations + pos, uniform_accuracies, width, label="uniform")

    # set y limits
    y_min = min(min(distance_accuracies), min(uniform_accuracies))
    y_max = max(max(distance_accuracies), max(uniform_accuracies))
    ax.set_ylim(y_min - 0.1, y_max + 0.1)

    ax.set_xlabel('Neighbors(k)')
    ax.set_ylabel('Accuracy')
    plt.title('Hyperparameters for training KNN Classifier')
    ax.set_xticks(locations + (width / 2), neighbors)
    ax.legend()
    # plt.grid(True)
    plt.show()


# Greedily finds best vars to include until no more
def find_best_variables(diabetes_130_us_hospitals_for_years_1999_2008):
    X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
    y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

    enc = OneHotEncoder(handle_unknown='ignore', min_frequency=0.05)

    mapping = {'NO': 0, '<30': 1, '>30': 2}
    y['readmitted'] = y['readmitted'].replace(mapping)

    X_train, X_test, y_train, y_test = train_test_split(enc.fit_transform(X), y, test_size=0.999)   # very small test (1/20 of actual) size to run quickly


    # flatten y to pass into model
    y_train_flat = np.array(y_train).flatten()

    hyper_parameters = dict(
        n_neighbors=3,
        weights="distance"
    )

    old_accuracy = 0
    variables = []
    accuracies = []

    # until converges
    while True:
        # find j with best accuracy
        best_accuracy = 0
        best_var = None
        for j in range(X_train.shape[1]):
            if j in variables:
                continue

            print("Testing " + str([*variables, j]))

            clf = KNeighborsClassifier(**hyper_parameters)
            clf.fit(X_train[:, [*variables, j]], y_train_flat)

            y_pred = clf.predict(X_test[:, [*variables, j]])
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_var = j
                if accuracy > old_accuracy:
                    print("NEW BEST, Accuracy: " + str(best_accuracy))
        
        if best_accuracy > old_accuracy:
            print("Improved accuracy to " + str(best_accuracy))
            print("Added " + str(best_var) + " to variables")
            old_accuracy = best_accuracy
            variables.append(best_var)
            accuracies.append(best_accuracy)
        else:
            break

    print("The best combination of vars was " + str(variables) + " with an accuracy of " + str(old_accuracy))

    print("Running full-size test...\n")
    X_train, X_test, y_train, y_test = train_test_split(enc.fit_transform(X), y, test_size=0.3)
    y_train_flat = np.array(y_train).flatten()

    hyper_parameters["n_neighbors"] = 100
    clf = KNeighborsClassifier(**hyper_parameters)
    clf.fit(X_train[:, variables], y_train_flat)

    y_pred = clf.predict(X_test[:, variables])
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    display_confusion_matrix(y_test, y_pred)

    return variables, accuracies


# Plots the results from find_best_variables and shows them in a graph
def find_best_variables_and_plot(diabetes_130_us_hospitals_for_years_1999_2008):
    # variables, accuracies = find_best_variables(diabetes_130_us_hospitals_for_years_1999_2008)
    variables, accuracies = [67, 22, 68, 12, 44], [0.4, 0.5, 0.6, 0.7, 0.8]

    x_values = []
    y_values = accuracies

    for i in range(len(variables)):
        print(variables[:,i])
        x_values.append(str(variables[:,i]))
    
    fig, ax = plt.subplots()

    ax.plot(x_values, y_values, marker='o', linestyle='-', color='b')

    plt.show()
    

def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NO", "<30", ">30"])

    disp.plot()
    plt.show()


def run_knn(diabetes_130_us_hospitals_for_years_1999_2008, test_size=0.3, k=100, weights="distance"):
    X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
    y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

    enc = OneHotEncoder(handle_unknown='ignore', min_frequency=0.05)

    mapping = {'NO': 0, '<30': 1, '>30': 2}
    y['readmitted'] = y['readmitted'].replace(mapping)

    X_train, X_test, y_train, y_test = train_test_split(enc.fit_transform(X), y, test_size=test_size)   # very small test (1/20 of actual) size to run quickly


    # flatten y to pass into model
    y_train_flat = np.array(y_train).flatten()

    hyper_parameters = dict(
        n_neighbors=k,
        weights=weights
    )

    clf = KNeighborsClassifier(**hyper_parameters)
    clf.fit(X_train, y_train_flat)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    print("\n\nTraining Classification Report:")
    print(classification_report(clf.predict(X_train), y_train))

    print("\nTesting Classification Report:")
    print(classification_report(y_test, y_pred))

    display_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":

    print("Fetching from UCI repo...")
    diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

    # find_best_variables_and_plot(diabetes_130_us_hospitals_for_years_1999_2008)

    # test_hyperparameters_and_plot(diabetes_130_us_hospitals_for_years_1999_2008)
    run_knn(diabetes_130_us_hospitals_for_years_1999_2008)