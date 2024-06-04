from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import pandas as pd


hyper_parameters = {
    'hidden_layer_sizes': 64,
    'activation': "relu", 
    'solver': "sgd",
    'alpha': 1,
    'learning_rate': "adaptive",
    'learning_rate_init': 0.01,
    'max_iter': 200,
    'n_iter_no_change': 100
    }

def feed_forward_neural_network():
    # fetch dataset 
    diabetes_130_us_hospitals_for_years_1999_2008 = pd.read_csv("diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")
    
    # # data (as pandas dataframes) 
    X, y = diabetes_130_us_hospitals_for_years_1999_2008.iloc[:, :-1], diabetes_130_us_hospitals_for_years_1999_2008.iloc[:, [-1]]
    # X = diabetes_130_us_hospitals_for_years_1999_2008.fe 
    # y = diabetes_130_us_hospitals_for_years_1999_2008

    print(X, y)


    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.1, shuffle=True)

    clf = MLPClassifier(hyper_parameters)

    clf.fit(X_tr, y_tr)


    print(f"trianing accuracy: {clf.score(X_tr, y_tr)}")
    print(f"testing accuracy: {clf.score(X_te, y_te)}")



if __name__ == "__main__":
    feed_forward_neural_network()
