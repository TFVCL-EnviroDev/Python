import numpy as np
from sklearn import linear_model as lm
from sklearn import preprocessing

F, N = map(int,input().split())

train_set = np.array([input().split() for _ in range(N)],float)

T = int(input())

test_set = np.array([input().split() for _ in range(T)],float)

lin_reg = lm.LinearRegression()
poly = preprocessing.PolynomialFeatures(degree=3)
lin_reg.fit(poly.fit_transform(train_set[:,:-1]), train_set[:,-1])

result = lin_reg.predict(poly.fit_transform(test_set))

print(*np.round(result,2), sep='\n')