import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, explained_variance_score

class LinearRegressionGD:
    def __init__(self, learning_rate=0.17, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0
        self.train_log = []

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iterations):
            y_hat = np.dot(X, self.w) + self.b
            error = y_hat - y
            dw = (2/m) * np.dot(X.T, error)
            db = (2/m) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            mse = np.mean(error ** 2)
            self.train_log.append(mse)

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def plot_mse(self, filename="mse_plot.png"):
        plt.plot(self.train_log)
        plt.xlabel("Iteration")
        plt.ylabel("Training MSE")
        plt.title("MSE vs Iterations")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot as {filename}")

# Load and preprocess the data
df = pd.read_csv('https://raw.githubusercontent.com/maangulaid/cs4375-linear-regression-Arahman/main/AirQualityUCI.csv', sep=';', decimal=',')
df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
df.dropna(inplace=True)
df.drop(columns=["Date", "Time"], inplace=True)

x = df.drop(columns=["C6H6(GT)"])
y = df["C6H6(GT)"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
  x_scaled,
  y,
  train_size=0.80,
  test_size=0.20,
  random_state=90)


# Initialize and train the model
model = LinearRegressionGD(learning_rate=0.17, iterations=1000)
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)

# Evaluation
final_train_mse = model.train_log[-1]
test_mse = np.mean((y_test_pred - y_test) ** 2)
r2 = r2_score(y_test, y_test_pred)
explained_var = explained_variance_score(y_test, y_test_pred)

# Output results
print("\n[Gradient Descent Linear Regression Results]")
print(f"Learning Rate     : {model.learning_rate}")
print(f"Iterations        : {model.iterations}")
print(f"Final Train MSE   : {final_train_mse:.4f}")
print(f"Final Test MSE    : {test_mse:.4f}")
print(f"R² Score         : {r2:.4f}")
print(f"Explained Var     : {explained_var:.4f}")

# Save plot
model.plot_mse()

    
import os  # ensure this is near the top of the file

trial_log_file = "trial_log.txt"
part_label = "# --- Part 1: Manual Gradient Descent ---"

write_header = not os.path.exists(trial_log_file)

with open(trial_log_file, "a") as log:
    if write_header:
        log.write("# CS4375 Assignment 1 - Parameter Tuning Trial Log\n")
        log.write("# Format: Learning Rate, Iterations, Final Train MSE, Final Test MSE\n\n")
        log.write(f"{part_label}\n")

    log.write(f"{model.learning_rate},{model.iterations},{final_train_mse:.4f},{test_mse:.4f}\n")
