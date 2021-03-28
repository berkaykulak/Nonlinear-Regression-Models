import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from warnings import filterwarnings
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from skompiler import skompile

hit = pd.read_csv("Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42)


X_train = pd.DataFrame(X_train["Hits"])
X_test = pd.DataFrame(X_test["Hits"])

cart_model = DecisionTreeRegressor(min_samples_split = 2)

cart_model.fit(X_train, y_train)
X_grid = np.arange(min(np.array(X_train)),max(np.array(X_train)), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, cart_model.predict(X_grid), color = 'blue')
plt.title('CART REGRESON AĞACI')
plt.xlabel('Atış Sayısı(Hits)')
plt.ylabel('Maaş (Salary)') ;

print(skompile(cart_model.predict).to('python/code'))
x = [91]

(345.2011551724138 if x[0] <= 117.5 else ((((1300.0 if x[0] <= 118.5 else
    641.0) if x[0] <= 122.5 else 1468.5236666666667) if x[0] <= 125.5 else
    621.9679230769232) if x[0] <= 143.0 else (958.6111111111111 if x[0] <=
    150.5 else 2460.0) if x[0] <= 151.5 else 499.1666666666667 if x[0] <=
    157.5 else 892.5402413793104) if x[0] <= 225.5 else 1975.0)

cart_model.predict(X_test)[0:5]
cart_model.predict([[91]])
y_pred =cart_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))


cart_model = DecisionTreeRegressor()
cart_model.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
cart_params = {"min_samples_split": range(2,100),
               "max_leaf_nodes": range(2,10)}

cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10)
cart_cv_model.fit(X_train, y_train)
cart_cv_model.best_params_
cart_tuned = DecisionTreeRegressor(max_leaf_nodes = 9, min_samples_split = 37)
cart_tuned.fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
y_pred

np.sqrt(mean_squared_error(y_test, y_pred))











