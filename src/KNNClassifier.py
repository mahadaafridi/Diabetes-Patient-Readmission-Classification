import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
#             self._X = X
#             self._y = y

#             # fitting stuff

#             self._fitted = True
#         except Exception as e:
#             self._X = None
#             self._y = None
#             raise e

#     # Returns the predicted labels based on input features
#     def predict(self, X) -> str:   # 0: NO, 1: <30, 2:>30?
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
            
#     #
#     def accuracy_score(self, X, y):
#         y_pred = self.predict(X)

#         num_same = 0
#         for i in range(len(y)):
#             num_same+= (y[i] == y_pred[i])

#         return num_same / len(y)

#     # Calculates the Euclidean distance between two points
#     def _calculate_distance(self, X1, X2) -> float:
#         distance_squared = 0
#         for X1_i, X2_i in zip(X1, X2):
#             distance_squared+= (X1_i - X2_i) ** 2
#         return math.sqrt(distance_squared)

            

        


if __name__ == "__main__":
    
    diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

    X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
    y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

    enc = OneHotEncoder(handle_unknown='ignore')

    mapping = {'NO': 0, '<30': 1, '>30': 2}
    y['readmitted'] = y['readmitted'].replace(mapping)
    

    X_train, X_test, y_train, y_test = train_test_split(enc.fit_transform(X), y, test_size=0.3)

    # flatten y to pass into model
    y_train_flat = np.array(y_train).flatten()

    k_values = [1 ,5, 20, 100, 500, 2000]
    weights = ["uniform", "distance"]

    best_accuracy = 0
    best_params = None

    for weight in weights:
        for k in k_values:
            hyper_parameters = dict(
                n_neighbors=k,
                weights=weight
            )
            print("TESTING WITH" + str(hyper_parameters))
            
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

    print("\nBEST:")
    print(best_accuracy)
    print(best_params)


    # hyper_parameters = dict(
    #     n_neighbors=100,
    #     weights="distance"
    # )
    # print("TESTING WITH" + str(hyper_parameters))

    # clf = KNeighborsClassifier(**hyper_parameters)
    # clf.fit(X_train, y_train_flat)

    # y_pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))