from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


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

    print(diabetes_130_us_hospitals_for_years_1999_2008.describe(exclude=['O']).transpose())
    print(diabetes_130_us_hospitals_for_years_1999_2008._get_numeric_data().columns)

    numeric_data = ['admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient',
       'number_diagnoses']

    # create a figure  with 7 rows and 7 columns for features in hawks_clean
    figure, axes = plt.subplots(11, 11, figsize=(30, 30))

    # Define the features to plot
    hawks_X = diabetes_130_us_hospitals_for_years_1999_2008[numeric_data]

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



    # enc = OneHotEncoder(handle_unknown='ignore')
    # # print(enc.fit_transform(X))
    # mapping = {'NO': 0, '<30': 1, '>30': 2}
    # y['readmitted'] = y['readmitted'].replace(mapping)
    # X = diabetes_130_us_hospitals_for_years_1999_2008.fe 
    # y = diabetes_130_us_hospitals_for_years_1999_2008

    # print(X, y)


    # X_tr, X_te, y_tr, y_te = train_test_split(enc.fit_transform(X), y, train_size=0.1, shuffle=True)

    # clf = MLPClassifier(**hyper_parameters)

    # clf.fit(X_tr, y_tr)


    # print(f"trianing accuracy: {clf.score(X_tr, y_tr)}")
    # print(f"testing accuracy: {clf.score(X_te, y_te)}")



if __name__ == "__main__":
    feed_forward_neural_network()
