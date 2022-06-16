#pip install pandas
#pip install numpy
#pip install sklearn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

data = pd.concat(
    map(pd.read_csv, ['1950.csv', '1960.csv'
                    , '1970.csv', '1980.csv'
                    , '1990.csv', '2000.csv']), ignore_index=True)

X = data.iloc[:, 5:-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
testX = pd.read_csv('2010.csv').iloc[:, 5:-1].values
testY = pd.read_csv('2010.csv').iloc[:, -1].values.reshape(-1,1)

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X,Y)

predY = regressor.predict(testX)
score = regressor.score(testX, testY)

print(score)



