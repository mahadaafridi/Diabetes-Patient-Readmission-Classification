import pandas as pd
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

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
