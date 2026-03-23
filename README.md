# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas as pd
### Step2

Read the csv file

### Step3
let the value of X and y variables

### Step4
Create the linear regression model and fit.

### Step5
Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.>

## Program:
```

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 3 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(100)  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

new_data = np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]])
new_predictions = model.predict(new_data)

print(f"Predictions on new data: {new_predictions}")

```
## Output:

<img width="1264" height="242" alt="image" src="https://github.com/user-attachments/assets/24955def-af26-4805-a6f8-6713f1861e79" />

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
