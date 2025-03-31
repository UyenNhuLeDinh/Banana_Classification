# %% [markdown]
# # CA 3 - Banana Classification
# 
# **Group 12**: Le Uyen Nhu Dinh, Sheikh Hasan Elahi, Isma Sohail.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# %% [markdown]
# ### Reading data

# %%
df = pd.read_csv('train.csv')

# %%
sc = StandardScaler()
df.iloc[:,:-1] = sc.fit_transform(df.iloc[:,:-1])


# %%
# Function removing outliers using Z_score:
def remove_outliers(df, threshold=3):
    df_clean = df.copy()
    numerical_cols = df_clean.select_dtypes(include=['number']).columns

    for col in numerical_cols:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        
        # Compute Z-score
        df_clean['Z_score'] = (df_clean[col] - mean) / std
        
        # Filter out outliers
        df_clean = df_clean[abs(df_clean['Z_score']) < threshold]

    df_clean = df_clean.drop(columns=['Z_score'])
    return df_clean

# %%
# df was scaled before 
df = remove_outliers(df, threshold=3)

X = df.drop(['Quality', 'Peel Thickness', 'Banana Density'], axis = 1)
y = df['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# ### EVALUATING MODELS AND PARAMETERS

# %%
def manual_search_evaluate(model_class, param_grid, X_train, X_test, 
                               y_train, y_test, scoring='accuracy', cv=5):
    # Initialize variables to store values after each iteration
    best_score = 0
    best_params = None
    best_model = None
    
    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Loop over each combination of parameters
    for params in param_combinations:
        scores = []
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation
        for train_idx, val_idx in skf.split(X_train, y_train):
            # Split training data into training and validation folds
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Train model with parameter combination:
            model = model_class(**params)
            model.fit(X_fold_train, y_fold_train)
            
            # Predict on validation set and compute accuracy
            y_val_pred = model.predict(X_fold_val)
            score = accuracy_score(y_fold_val, y_val_pred)
            scores.append(score)
        
        # Average cross-validation score for the param combination
        avg_score = np.mean(scores)
        
        # Update best score and best params
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    # Train best model on full training set
    best_model = model_class(**best_params)
    best_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = best_model.predict(X_test)
    
    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {acc:.3f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix: {model_class.__name__}")
    plt.show()
    
    return best_model, acc, cm


# %%
# RBF KERNEL SVM
param_grid_rbf = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

rbf_SVC = manual_search_evaluate(SVC, param_grid_rbf, X_train, X_test, y_train, y_test)

# %%
df_test = pd.read_csv('test.csv')
df_test = df_test.drop(['Peel Thickness', 'Banana Density'], axis =1)
df_test = sc.fit_transform(df_test)
y_test_kaggle = rbf_SVC[0].predict(df_test)
y_test_kaggle = pd.DataFrame(y_test_kaggle, columns=["Quality"])
y_test_kaggle.index.name = "ID"
y_test_kaggle[['Quality']].to_csv("submission6.csv")

# %% [markdown]
# So far RBF-kernel Support Vector Machine is the best model with C=10 and gamma=0.1.


