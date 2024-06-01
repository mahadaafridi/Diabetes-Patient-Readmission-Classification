import math
import numpy as np

class KNNClassifier:

    def __init__(self, k = 5, ) -> None:
        self._fitted = False
        self._X = None
        self._y = None
        self._k = k

    # Trains the model on features X and labels y
    def fit(self, X, y) -> None:
        if len(X) != len(y):
            raise Exception("Lengths of features and labels don't match")
        try:
            self._X = X
            self._y = y

            # fitting stuff

            self._fitted = True
        except Exception as e:
            self._X = None
            self._y = None
            raise e

    # Returns the predicted labels based on input features
    def predict(self, X) -> str:   # 0: NO, 1: <30, 2:>30?
        y = np.array([])
        # Calculate the KNN for each datapoint
        for X_new in X:
            # Calculate distance from each point in self._X
            distances = np.array([self._calculate_distance(X_new, X_old) for X_old in self._X])

            # Sort by distance and choose the K nearest
            sort_by_dist = np.argsort(distances)
            sorted_y = self._y[sort_by_dist]
            knn = {
                0: 0,
                1: 0,
                2: 0
            }
            for i in range(min(self._k, len(sorted_y))):
                knn[sorted_y[i]]+= 1
            
            # Choose 
            y = np.append(y, max(knn, key=knn.get))
        
        return y
            
    #
    def accuracy_score(self, X, y):
        y_pred = self.predict(X)

        num_same = 0
        for i in range(len(y)):
            num_same+= (y[i] == y_pred[i])

        return num_same / len(y)

    # Calculates the Euclidean distance between two points
    def _calculate_distance(self, X1, X2) -> float:
        distance_squared = 0
        for X1_i, X2_i in zip(X1, X2):
            distance_squared+= (X1_i - X2_i) ** 2
        return math.sqrt(distance_squared)

            

        


if __name__ == "__main__":
    clf = KNNClassifier(k=1)
    X_tr = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 4]])
    y_tr = np.array([0, 0, 1])
    clf.fit(X_tr, y_tr)

    X_te = [[0, 0, 1], [0, 0, 0], [0, 1, 5], [1, 0, 9]]
    y_te = [0, 0, 1, 1]
    print(clf.predict(X_te))
    print(clf.accuracy_score(X_te, y_te))