import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

filename = 'SET-DATA.csv'
_df = pd.read_csv(filename)

samples = 10443
print('samples=', samples)
df = _df.iloc[:samples, :]

X = df.iloc[:, 3:6]
# X = X.iloc[0:-1]
y = df['Open'].iloc[1:]

X_train = X.iloc[0:-1] # last row excluded
y_train = df['Open'].iloc[1:] # first row excluded
X_test = X.iloc[-1:,:] # last row for predict

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

print('df', _df.tail())
print('last close -->', _df.iloc[samples-1:samples, 5:6].values)
print('open prediction -->', y_pred)
print('open is -->', _df.iloc[samples:samples+1, 2:3].values) # only open_value

# print("R^2: {}".format(reg_all.score(X_test, y_test)))
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("Root Mean Squared Error: {}".format(rmse))
