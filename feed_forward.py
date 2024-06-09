from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def feed_forward_neural_network():
    """Scratch work"""
    # fetch dataset 

    hyper_parameters = {
    'hidden_layer_sizes': (64, 32, 16),
    'activation': "relu", 
    'solver': "sgd",
    'alpha': 0.0001,
    'learning_rate': "adaptive",
    'learning_rate_init': 0.0001,
    'max_iter': 700,
    'n_iter_no_change': 100,
    'verbose': True
    }                                 
    
    
    diabetes_130_us_hospitals_for_years_1999_2008 = pd.read_csv("diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")

    print(diabetes_130_us_hospitals_for_years_1999_2008.describe(exclude=['O']).transpose())
    print(diabetes_130_us_hospitals_for_years_1999_2008._get_numeric_data().columns)

    numeric_data = ['admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient',
       'number_diagnoses']
    
    

    # create a figure  with 7 rows and 7 columns for features in hawks_clean
    figure, axes = plt.subplots(11, 11, figsize=(30, 30))

    # Create a dictionary mapping species to unique integers
    species_to_int = {'NO': 'red', '<30': 'blue', '>30': 'green'}

    # Replace species names with integers in the DataFrame
    hawks_color = diabetes_130_us_hospitals_for_years_1999_2008['readmitted'].map(species_to_int)

    ### YOUR CODE STARTS HERE ###
    # Plot a scatter for each feature pair. 
    # Make sure to color the points by their class label.
    # Include an x-label and a y-label for each subplot.
    for y in range(11):
        for x in range(11):
            axes[y][x].scatter(diabetes_130_us_hospitals_for_years_1999_2008[numeric_data[y]], diabetes_130_us_hospitals_for_years_1999_2008[numeric_data[x]], c=hawks_color, s = 1)
            if x == 0:
                axes[y][x].set_ylabel(numeric_data[y])
            if y == 10:
                axes[y][x].set_xlabel(numeric_data[x])
            
            # axes[y][x].set_title(f"{hawk_phys_attr[y]} vs. {hawk_phys_attr[x]}")
            # axes[y][x].set_xlabel(numeric_data[x])
            # axes[y][x].set_ylabel(numeric_data[y])

    plt.tight_layout()
    # plt.legend()
    plt.savefig('fig.png', dpi=150)

    
    # # data (as pandas dataframes) 
    X, y = diabetes_130_us_hospitals_for_years_1999_2008.iloc[:, :-1], diabetes_130_us_hospitals_for_years_1999_2008.iloc[:, [-1]]

    # # print(enc.fit_transform(X))
    # mapping = {'NO': 0, '<30': 1, '>30': 2}
    # y['readmitted'] = y['readmitted'].replace(mapping)
    # X = diabetes_130_us_hospitals_for_years_1999_2008.fe 
    # y = diabetes_130_us_hospitals_for_years_1999_2008

    # print(X, y)

def plot_error_for_hyper_params(data):
    """Change hidden layer size, train_size"""

    def test_hyper_params(test_params, test_param_name, params):
        """Function that makes and saves a graph of varyign hyper parameters"""
        original_value = params[test_param_name]

        training_error = []
        testing_error = []

        for p in test_params:
            params[test_param_name] = p

            clf = MLPClassifier(**params)

            clf.fit(X_tr[:p], y_tr[:p])

            training_error.append(1 - clf.score(X_tr, y_tr))
            testing_error.append(1 - clf.score(X_te, y_te))

            print(f"{test_param_name}: {p} done")

        params[test_param_name] = original_value
        
        plt.semilogx(test_params, training_error, color="orange", label="training error")
        plt.semilogx(test_params, testing_error, color="blue", label="testing error")

        plt.xlabel(test_param_name)
        plt.ylabel("Error")

        plt.legend()

        plt.savefig(f"{test_param_name}_error.png", dpi=100)



    X, y = data.iloc[:, :-1], data.iloc[:, [-1]]

    # remove useless columns
    X.drop(labels=["patient_nbr", "encounter_id", "payer_code"], axis=1, inplace = True)

    # optimal hyper parameters
    hyper_parameters = {
    'hidden_layer_sizes': (64, 32, 16),
    'activation': "relu", 
    'solver': "sgd",
    'alpha': 0.0001,
    'learning_rate': "adaptive",
    'learning_rate_init': 0.0001,
    'max_iter': 700,
    'n_iter_no_change': 100,
    'verbose': True                                  # prints out the training iterations
    }

    # different param values to iterate through
    training_sizes = [1000, 5000, 10000, 50000]
    hidden_layer_sizes = [8, 16, 32, 64]
    alphas = [0.001, 0.01, 0.1, 1]

    enc = OneHotEncoder(handle_unknown='ignore')
    X_tr, X_te, y_tr, y_te = train_test_split(enc.fit_transform(X), y, train_size=0.1, shuffle=True)

    # X_tr, y_tr = X_tr[:50000], y_tr[:50000]   # <- speeds it up a little for testing the hyper params

    clf = MLPClassifier(**hyper_parameters)

    clf.fit(X_tr, y_tr)

    output_metrics(clf, X_te, y_te)

    # make graphs for different hyper_params
    # test_hyper_params(hidden_layer_sizes, "hidden_layer_sizes", hyper_parameters)
    # test_hyper_params(alphas, "alpha", hyper_parameters)

    # print(f"trianing accuracy: {clf.score(X_tr, y_tr)}")
    # print(f"testing accuracy: {clf.score(X_te, y_te)}")

def output_metrics(clf, X_te, y_te):
    """output different statistical measures for the NN"""
    y_pred = clf.predict(X_te)

    # cm = confusion_matrix(y_true=y_te, y_pred=y_pred)

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

    # disp.plot()

    # plt.show()

    
    print(classification_report(y_pred = y_pred, y_true = y_te))
    print(f"testing accuracy: {accuracy_score(y_pred=y_pred, y_true=y_te)}")


if __name__ == "__main__":
    diabetes_130_us_hospitals_for_years_1999_2008 = pd.read_csv("diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")
    # feed_forward_neural_network()

    plot_error_for_hyper_params(diabetes_130_us_hospitals_for_years_1999_2008)
