import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#let get the data base from maangulaid github
df = pd.read_csv('https://raw.githubusercontent.com/maangulaid/cs4375-linear-regression-Arahman/main/AirQualityUCI.csv', sep=';', decimal=',')


# realised it has NAN or missing data let clean up more
# #let clean up and remove unneccesary columns or rows
df.drop(columns= [col for col in df.columns if 'Unnamed' in col], inplace=True)
df.dropna(inplace=True)
df.drop(columns=["Date", "Time"], inplace=True)

#set x and y 
x = df.drop(columns=["C6H6(GT)"])
y = df["C6H6(GT)"]


#standardize 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# train/test ratio
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled,
    y,
    train_size=0.80, 
    test_size=0.20,
    random_state=90
    
    )


##print(df.tail())

# parameters 
learning_rate =  0.17
num_iteration = 1000

m,n = x_train.shape


# wights and bias 
#bias a constant number added to shift the prediction up/down

w=np.zeros(n)
b=0


#let do the trainning here 
#remeber x_train is the input data
train_log = []

for i in range(num_iteration):
    # Predict: y_hat = Xw + b
    y_hat = np.dot(x_train, w) + b
    
    # Step 1: Compute error
    error = y_hat - y_train

    # Step 2: Compute gradients
    dw = (2/m) * np.dot(x_train.T, error)
    db = (2/m) * np.sum(error)

    # Step 3: Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db

    # Step 4: Calculate and store MSE
    mse = np.mean(error ** 2)
    train_log.append(mse)

y_test_pred = np.dot(x_test, w) + b
test_mse = np.mean((y_test_pred - y_test) ** 2)
print(f"\nFinal Test MSE: {test_mse:.4f}")


plt.plot(train_log)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("MSE vs Iterations")
plt.show()


if i % 100 == 0:
    print(f"Iteration {i}: Training MSE = {mse:.4f}")



import csv
import os

# Compute final training MSE from the last iteration
final_train_mse = train_log[-1]  # last logged training MSE

# Log file name
log_file = "logs.csv"

# Save training MSE plot as image
plot_filename = "mse_plot.png"
plt.plot(train_log)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("MSE vs Iterations")
plt.grid(True)
plt.savefig(plot_filename)
plt.close()
print(f"Saved plot as {plot_filename}")

# Append summary to report.txt
with open("report.txt", "a") as f:
    f.write("\n\n")
    f.write("CS4375 Assignment 1 – Trial Report\n")
    f.write("===================================\n")
    f.write(f"Learning Rate     : {learning_rate}\n")
    f.write(f"Iterations        : {num_iteration}\n")
    f.write(f"Final Train MSE   : {final_train_mse:.4f}\n")
    f.write(f"Final Test MSE    : {test_mse:.4f}\n")
    f.write(f"MSE Plot Saved As : {plot_filename}\n")
    f.write("===================================\n")
