# An Investigation of Classification Methods for the Diabetes 130-US Hospitals Dataset


## Dataset
**Dataset**: [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

### Classification Methods:
- Logistic Classifier
- K-Nearest Neighbors
- Feed-Forward Neural Network
- Random Forest

---

## Summary

This project investigates the "Diabetes 130-US Hospitals for Years 1999-2008" database, representing patients diagnosed with diabetes throughout 130 US hospitals from 1999-2008. The goal is to determine the readmission of diabetes patients based on whether they were readmitted within <30 days, >30 days, or not at all. We experimented with K-Nearest Neighbors, Logistic Regression, Feed-Forward Neural Networks, and Random Forest classifiers using 49 features of diabetes patients. Our findings suggested that all classifiers performed similarly, with Random Forest achieving slightly higher accuracy. However, all classifiers struggled with identifying one of the labels (<30 days readmission).

---

## Data Description

The dataset contains 10 years (1999-2008) of clinical care from 130 US hospitals. Each row represents a patient diagnosed with diabetes, with various test results and patient data. The goal is to predict early readmission (within 30 days) after discharge. The dataset has:
- **101,766 instances** 
- **49 features** (categorical and integer values).

### Ordinal Variables
We generated summary statistics and pairwise scatter plots for the 13 ordinal variables. These helped uncover relationships, patterns, and outliers in the data.

---

## Classifiers

### Logistic Classifier
**Description**: Estimates the probability of class membership using a logistic function.
- **Hyperparameters**:
  - `C`: Regularization strength. Values tested: [0.001, 0.01, 0.1, 1, 10, 100]
  - `Solver`: Algorithm for optimization. Values tested: ['newton-cg', 'lbfgs']
- **Software used**: Scikit-learn, Matplotlib

---

### Feed-Forward Neural Network
**Description**: Multilayer Perceptron (MLP) with layers of perceptrons.
- **Hyperparameters**:
  - `hidden_layer_sizes`: Tested values: [8, 16, 32, 64]
  - `activation`, `solver`, `alpha`, `learning_rate`, `learning_rate_init`, `max_iter`, `n_iter_no_change`
  - **Software used**: Scikit-learn, Pandas, Matplotlib

---

### K-Nearest Neighbors Classifier
**Description**: Predicts the label by finding the k-nearest neighbors.
- **Hyperparameters**:
  - `n_neighbors`: Values tested: [1, 5, 20, 100, 500, 2000]
  - `weights`: Values tested: ['uniform', 'distance']
- **Software used**: Scikit-learn, Numpy, Matplotlib

---

### Random Forest Classifier
**Description**: Constructs multiple decision trees and outputs the most common prediction.
- **Hyperparameters**:
  - `n_estimators`: Tested values: [50, 100, 200]
  - `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`
- **Software used**: Scikit-learn, Matplotlib

---

## Experimental Setup

### Data Preprocessing
- Missing values were handled, and categorical variables were OneHotEncoded using Scikit-learn, resulting in over 2,000 dimensions.
  
### Data Partitioning
- The dataset was split into:
  - 20% for testing
  - 60% for training
  - 20% for validation
  
- K-fold cross-validation (k=5) was applied to ensure generalization.

---

## Experimental Results

### Logistic Classifier
- Best hyperparameters: `C = 0.01`, `solver = 'newton-cg'`
- Accuracy: ~59%

### Feed-Forward Neural Network
- Overfitting observed unless `alpha` was high. Accuracy: ~58%

### K-Nearest Neighbors Classifier
- Best results with `k=100` and `distance` weights. Accuracy: ~57%

### Random Forest Classifier
- Best accuracy: 59%, with 92% training accuracy.

---

## Classifiers Performance on Different Training Sizes

The KNN classifier performed best on smaller training sizes, while Random Forest captured non-linear relationships, performing best with larger datasets.

---

## Insights

All models struggled to predict the <30 days readmission label, suggesting bias in the dataset. Boosting techniques may improve results. Despite challenges, the Random Forest classifier demonstrated value, outperforming random classification.


---

## Works Cited
- Strack, B. et al. "Impact of HbA1c Measurement on Hospital Readmission Rates". *BioMed Research International*, 2014.
- Hu, Y. & Sokolova, M. "Convolutional Neural Networks in Multi-Class Classification of Medical Data". *Cornell University*, 2020.
- Clore, J., & Cios, K. "Diabetes 130-US Hospitals for Years 1999-2008". *UCI Machine Learning Repository*, 2014.
