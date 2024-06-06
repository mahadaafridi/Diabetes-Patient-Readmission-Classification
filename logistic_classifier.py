import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

enc = OneHotEncoder(handle_unknown='ignore')
# print(enc.fit_transform(X))
mapping = {'NO': 0, '<30': 1, '>30': 2}
y['readmitted'] = y['readmitted'].replace(mapping)


#random state 
X_train, X_test, y_train, y_test = train_test_split(enc.fit_transform(X), y, test_size=0.3, random_state=42)

# this didn't improve performance ?
scaler = StandardScaler(with_mean=False)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def errors_for_train_sizes_lr(X_tr: np.array, y_tr: np.array, X_te: np.array, y_te: np.array, train_sizes: list[int]) -> tuple[list, list, list, list]:    
    # append error rates to the following lists
    tr_err_lr = [] # training error rates for Logistic Regression
    te_err_lr = [] # testing error rates for Logistic Regression
    accuracy_list = []
    ### YOUR CODE STARTS HERE ###
    for size in train_sizes:
        X_tr_sample = X_tr[:size, :]
        y_tr_sample = y_tr[:size]
        X_te_sample = X_te[:size, :]
        y_te_sample = y_te[:size]
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

        model.fit(X_tr_sample, y_tr_sample)

        y_pred = model.predict(X_te_sample)

        accuracy = accuracy_score(y_te_sample, y_pred)

        tr_err = 1 - model.score(X_tr_sample, y_tr_sample)
        te_err = 1 - model.score(X_te_sample , y_te_sample)
        accuracy_list.append(accuracy)
        tr_err_lr.append(tr_err)
        te_err_lr.append(te_err)

    return tr_err_lr, te_err_lr, accuracy_list 

train_sizes = [50, 500, 2000, 5000, 10000, 20000, 40000]
tr_err_lr, te_err_lr, acc_list = errors_for_train_sizes_lr(X_train, y_train, X_test, y_test, train_sizes)

print(tr_err_lr)
print(te_err_lr)
print(acc_list)

# print(accuracy)
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
