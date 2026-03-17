# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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
