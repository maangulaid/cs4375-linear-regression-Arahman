import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

class LinearRegressionLibrary:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.model = SGDRegressor(max_iter=self.iterations, eta0=self.learning_rate, learning_rate='constant', random_state=90)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

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
    random_state=90
)

# Initialize and train the model
model = LinearRegressionLibrary(learning_rate=0.01, iterations=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Evaluation
train_pred = model.predict(x_train)
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

#  results
print("\n[SGDRegressor Results]")
print(f"Learning Rate     : {model.learning_rate}")
print(f"Iterations        : {model.iterations}")
print(f"Final Train MSE   : {train_mse:.4f}")
print(f"Final Test MSE    : {test_mse:.4f}")
print(f"R² Score         : {r2:.4f}")
print(f"Explained Var     : {explained_var:.4f}")

# Plot predicted vs actual
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual C6H6(GT)")
plt.ylabel("Predicted C6H6(GT)")
plt.title("Actual vs Predicted - Part 2")
plt.grid(True)
plt.savefig("predicted_vs_actual_part2.png")
plt.close()

    
    
    
import os  
trial_log_file = "trial_log.txt"
part_label = "# --- Part 2---"

write_header = not os.path.exists(trial_log_file)

with open(trial_log_file, "a") as log:
    if write_header:
        log.write("# CS4375 Assignment 1 - Parameter Tuning Trial Log\n")
        log.write("# Format: Learning Rate, Iterations, Final Train MSE, Final Test MSE\n\n")
        log.write(f"Part2.py\n")
        log.write(f"{part_label}\n")
        
    
    log.write(f"{model.learning_rate},{model.iterations},{train_mse:.4f},{test_mse:.4f}\n")
