#!/bin/python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


train = open("trainingdata.txt",'r')
data = pd.read_csv(train, sep=',', header=None)
data.columns = ["charge_time", "battery_time"]
#Using Jupyter and matplotlib, it's possible to see that the battery time is constant at 8.0 for charge time > 4, and linear otherwise
#Therefore, the part of the data for charge time > 4 is not necessary for the prediction
data = data.drop(data[data.charge_time>=4].index)
x = np.array(data.iloc[:, 0]).reshape(-1, 1)
y = np.array(data.iloc[:, 1])

def result(n):
    if (n>=4):
        return 8.0
    else:
        lin_reg = LinearRegression()
        lin_reg.fit(x,y)
        return round(float(lin_reg.predict(np.array(n).reshape(1,-1))),2)

if __name__ == '__main__':
    timeCharged = float(input())
    print(result(timeCharged))