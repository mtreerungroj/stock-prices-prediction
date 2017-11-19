import pandas as pd
import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def predict_prices(dates, prices, x):
	dates = np.reshape(dates, (len(dates), 1))

	svr_lin = SVR(kernel='linear', c=1e3)
	svr_poly = SVR(kernel='poly', c=1e3, degree=2)
	svr_rbf = SVR(kernel='rbf', c=1e3, gamma=0.1)
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)
	svr_rbf.fit(dates, prices)

	plt.scatter(dates, prices, color='black', label='Data')
	plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
	plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
	plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


def get_data(filename):
	df = pd.read_csv(filename)
	dates = df['Date/Time'].values
	prices = df['price'].values﻿


predicted_price = predict_prices(dates, prices, '11/11/2017')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

filename = 'SET-DATA.csv'
_df = pd.read_csv(filename)
# prices = df['Close'].values
# plt.scatter(df.index, prices, color='black', label='Data', s=0.1)
# plt.show()

# samples = len(_df.index) -1 
samples = 10443
print('samples=', samples)
df = _df.iloc[:samples, :]

X = df.iloc[:, 3:6] 
y = df['Open'].iloc[1:]
# print(X)
# print(y)

X_train = X.iloc[0:-1] # last row excluded
y_train = df['Open'].iloc[1:] # first row excluded
X_test = X.iloc[-1:,:] # last row for predict

# print(type(X_train))
# print(X_train)
# print(y_train)

# print(X_test)
# print(type(X_test))

########################################################

# from sklearn.svm import SVR

# svr_lin = SVR(kernel='linear')
# svr_lin.fit(X_train, y_train)
# y_pred = svr_lin.predict(X_test)

# print(y_pred)
# print(_df.iloc[11:12, 2:3]) # only Open value

########################################################

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

print('df', _df.tail())
print('last close -->', _df.iloc[samples-1:samples, 5:6])
print('open prediction -->', y_pred)
print('open is -->', _df.iloc[samples:samples+1, 2:3]) # only open_value

# print("R^2: {}".format(reg_all.score(X_test, y_test)))
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("Root Mean Squared Error: {}".format(rmse))

########################################################
