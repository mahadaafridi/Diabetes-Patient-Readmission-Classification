

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
        pass



if __name__ == "__main__":
    clf = KNNClassifier(k=10)
    clf.fit([1],[2])
