from ucimlrepo import fetch_ucirepo 
import pandas as pd

# https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

# fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)
# data (as pandas dataframes)
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets
# print(X)

#the mapping for the code we gonna use 
mapping = {'NO': 0, '<30': 1, '>30': 2}
y['readmitted'] = y['readmitted'].replace(mapping)

print(X.describe())
print(y) # readmitted

# metadata 
# print(diabetes_130_us_hospitals_for_years_1999_2008.metadata) 
  
# variable information 
# print(diabetes_130_us_hospitals_for_years_1999_2008.variables)

