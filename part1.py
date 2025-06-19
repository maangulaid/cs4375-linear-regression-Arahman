import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')

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
learning_rate =  0.02
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

y_test_pred = np.dot(x_test, w) + b
test_mse = np.mean((y_test_pred - y_test) ** 2)
print(f"\nFinal Test MSE: {test_mse:.4f}")