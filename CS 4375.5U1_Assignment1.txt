import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

url_train = "https://raw.githubusercontent.com/Phu-D-Le/CS-4375_Asssignment1/main/heart_disease_dataset_train.csv"
url_test = "https://raw.githubusercontent.com/Phu-D-Le/CS-4375_Asssignment1/main/heart_disease_dataset_test.csv"

df_train = pd.read_csv(url_train)
df_test = pd.read_csv(url_test)

df_train['dataset'] = 'train'
df_test['dataset'] = 'test'

for col in ['Gender', 'Family History', 'Diabetes', 'Obesity', 'Stress Level', 'Blood Sugar', 'Exercise Induced Angina']:
    df_train[col] = df_train[col].astype('category').cat.codes
    df_test[col] = df_test[col].astype('category').cat.codes

numerical_cols = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours']
for col in numerical_cols:
    df_train[col] = (df_train[col] - df_train[col].mean()) / df_train[col].std()
    df_test[col] = (df_test[col] - df_test[col].mean()) / df_test[col].std()

X_train = df_train.drop(['Heart Disease', 'dataset'], axis=1).values.astype(np.float32)
y_train = df_train['Heart Disease'].values.astype(np.float32)
X_test = df_test.drop(['Heart Disease', 'dataset'], axis=1).values.astype(np.float32)
y_test = df_test['Heart Disease'].values.astype(np.float32)

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (linear_model - y))
            db = (1 / n_samples) * np.sum(linear_model - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

model = LinearRegression(learning_rate=0.001, n_iters=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = np.mean((predictions - y_test)**2)
print(f"Test MSE: {mse}")

# Plots
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Heart Disease")
plt.ylabel("Predicted Heart Disease")
plt.title("Linear Regression: Actual vs. Predicted Heart Disease")
plt.show()
