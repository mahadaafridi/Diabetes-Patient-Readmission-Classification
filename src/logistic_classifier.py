import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets
enc = OneHotEncoder(handle_unknown='ignore')
# print(enc.fit_transform(X))
mapping = {'NO': 0, '<30': 1, '>30': 2}
y['readmitted'] = y['readmitted'].replace(mapping)

X_encoded = enc.fit_transform(X)

X_train_val, X_test, y_train_val, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#split even more 
#i used the experimental setup thing she said
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


###   HYPER PARAM TUUUNNNNINGGG
### USE THE VALIDATION SET FO RTHIS PART 



C_vals = [0.001, 0.01, 0.1, 1, 10, 100]
solvers = ['newton-cg', 'lbfgs']

best_accuracy = 0
best_params = None

for C_val in C_vals:
    for solver in solvers:
        model = LogisticRegression(multi_class='multinomial', solver=solver, C=C_val, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_val_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = [C_val, solver]

print(best_params)
# END

#   DIFFERENT LEARNING SIZES 
def errors_for_train_sizes_lr(X_tr: np.array, y_tr: np.array, X_te: np.array, y_te: np.array, train_sizes: list[int]) -> tuple[list, list, list, list]:    
        # append error rates to the following lists
        tr_err_lr = [] # training error rates for Logistic Regression
        te_err_lr = [] # testing error rates for Logistic Regression
        accuracy_list = []

        for size in train_sizes:
                X_tr_sample = X_tr[:size, :]
                y_tr_sample = y_tr[:size]
                X_te_sample = X_te[:size, :]
                y_te_sample = y_te[:size]
                model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000, C=.01)

                model.fit(X_tr_sample, y_tr_sample)

                y_pred = model.predict(X_te_sample)

                accuracy = accuracy_score(y_te_sample, y_pred)

                tr_err = 1 - model.score(X_tr_sample, y_tr_sample)
                te_err = 1 - model.score(X_te_sample , y_te_sample)
                accuracy_list.append(accuracy)
                tr_err_lr.append(tr_err)
                te_err_lr.append(te_err)


        return tr_err_lr, te_err_lr, accuracy_list 

train_sizes = [50, 500, 2000, 5000, 10000, 20000, 40000, X_train.shape[0]] 

tr_err_lr, te_err_lr, acc_list = errors_for_train_sizes_lr(X_train, y_train, X_test, y_test, train_sizes)

print(tr_err_lr)
print(te_err_lr)
print(acc_list)

plt.figure(figsize=(10, 6))

plt.semilogx(train_sizes, tr_err_lr, label='Training Error', color='black')
plt.semilogx(train_sizes, te_err_lr, label='Testing Error', color='purple')
plt.semilogx(train_sizes, acc_list, label='Accuracy', color='red')

plt.xlabel('Training Size')
plt.ylabel('Accuracy or Error Rate')
plt.title('Training Size vs Accuracy or Error Rate')
plt.legend()
plt.grid(True)
plt.show()
#     END



## RUNNING FINAL TEST
model = LogisticRegression(multi_class='multinomial', solver='newton-cg', C=0.01, max_iter=1000)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)


print(test_accuracy)
print(class_report)
print(conf_matrix)
