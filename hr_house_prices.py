# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
from sklearn import linear_model

F,N = map(int,input().split(' '))
train_data = np.array([input().split(' ') for _ in range(0,N)],dtype=np.float64)
T=int(input())
test_data = np.array([input().split(' ') for _ in range(0,T)],dtype=np.float64)
X = train_data[:,0:F]
Y = train_data[:,-1]

lin_reg = linear_model.LinearRegression()
lin_reg.fit(X,Y)
result = lin_reg.predict(test_data)

print(*np.round(result,2), sep='\n')