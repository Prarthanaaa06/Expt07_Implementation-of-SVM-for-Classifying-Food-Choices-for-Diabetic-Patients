# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Start**

2. **Import Libraries**

   * Import required libraries:

     * `pandas`, `numpy`
     * `StandardScaler`
     * `train_test_split`, `GridSearchCV`
     * `SVC` (Support Vector Classifier)
     * Evaluation metrics (`accuracy`, `classification_report`, `confusion_matrix`)
     * `seaborn`, `matplotlib`

3. **Load Dataset**

   * Read dataset `food_items_binary.csv` into a DataFrame
   * Display first few rows and column names

4. **Select Features and Target**

   * Define input features `X`:

     * `Calories`, `Total Fat`, `Saturated Fat`, `Sugars`, `Dietary Fiber`, `Protein`
   * Define target variable `y`:

     * `class`

5. **Split Dataset**

   * Divide data into:

     * Training set (80%)
     * Testing set (20%)
   * Use `random_state = 42`

---

### **Data Preprocessing**

6. **Feature Scaling**

   * Initialize `StandardScaler`
   * Fit and transform training data
   * Transform testing data using same scaler

---

### **Model Training with Hyperparameter Tuning**

7. **Initialize SVM Model**

   * Create `SVC` model

8. **Define Parameter Grid**

   * Set hyperparameters for tuning:

     * `C`: [0.1, 1, 10, 100]
     * `kernel`: ['linear', 'rbf']
     * `gamma`: ['scale', 'auto']

9. **Apply Grid Search**

   * Use `GridSearchCV` with:

     * 5-fold cross-validation (`cv=5`)
     * Scoring metric: accuracy

10. **Train Model**

* Fit grid search on training data
* Select best model (`best_estimator_`)

---

### **Prediction**

11. **Make Predictions**

* Predict class labels using test data (`X_test`)
* Store predictions in `y_pred`

---

### **Model Evaluation**

12. **Calculate Accuracy**

* Compute accuracy score

13. **Generate Classification Report**

* Display precision, recall, and F1-score

14. **Compute Confusion Matrix**

* Generate confusion matrix for predictions

---

### **Visualization**

15. **Plot Confusion Matrix**

* Use heatmap to visualize confusion matrix
* Label axes and add title

16. **Display Plot**

17. **End**

---

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: Prarthana D
RegisterNumber:  212225230213

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv('food_items_binary.csv')

print(df.head())
print(df.columns)
features = ['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target = 'class'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC()
param_grid = {
    'C': [0.1,1,10,100],
    'kernel': ['linear','rbf'],
    'gamma' : ['scale','auto']
}
grid_search = GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model=grid_search.best_estimator_


print("Name: PRARTHANA D")
print("Reg. No: 212225230213")
print("Best Parameters:",grid_search.best_params_)
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print("Accuracy: ",acc)

print("Classification Report: \n", classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


*/
```

## Output:

<img width="1226" height="696" alt="image" src="https://github.com/user-attachments/assets/935ac867-8c0b-4534-bd24-6634332dac25" />

<img width="855" height="795" alt="image" src="https://github.com/user-attachments/assets/8e73ee02-95f6-4a37-a1ea-82d7173ca9d7" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
