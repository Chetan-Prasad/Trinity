import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

#Reading the Training Dataset and preprocessing
dataset = pd.read_csv('F:/Notes an Stuff/Intelligent Systems/Machine Learning/tcd ml 2019-20 income prediction training (with labels).csv')
dataset = dataset.drop(['Instance','Wears Glasses','Hair Color','Body Height [cm]'],axis=1)

#Reading the Test Dataset and a little preprocessing
testDataset = pd.read_csv('F:/Notes an Stuff/Intelligent Systems/Machine Learning/tcd ml 2019-20 income prediction test (without labels).csv')
testDataset = testDataset.drop(['Instance','Wears Glasses','Hair Color','Body Height [cm]'],axis=1)

#Class for Imputing NaN values. Categorical features most common or repeated values are imputed
#For numeric the mean of the column is imputed
class DataImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[col].value_counts().index[0]
            if X[col].dtype == np.dtype('O') else X[col].mean() for col in X],
            index=X.columns)
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
#Imputing    
dataset = DataImputer().fit_transform(dataset)
testDataset = testDataset.iloc[:,:-1]
testDataset = DataImputer().fit_transform(testDataset)

#
def encodeTargetSmooth(data, target, cat_var, smooth):
    
    train_target = data.copy()
    code_map = dict()    # stores mapping between original and encoded values
    default_map = dict() # stores global average of each variable
   
    for v in cat_var:
        prior = data[target].mean()
        n = data.groupby(v).size()
        mu = data.groupby(v)[target].mean()
        mu_smoothed = (n * mu + smooth * prior) / (n + smooth)
       
        train_target.loc[:, v] = train_target[v].map(mu_smoothed)        
        code_map[v] = mu_smoothed
        default_map[v] = prior        
    return train_target, code_map, default_map

#Features being Target Encoded
categ_var = ['Gender','Country','Profession','University Degree']

dataset1,code_map,default_map = encodeTargetSmooth(dataset,'Income in EUR', categ_var,20)

for c in categ_var:
    testDataset.loc[:,c] = testDataset[c].map(code_map[c])
    
X_train = dataset1.iloc[:,:-1]
y_train = dataset1.iloc[:,-1]
X_test = testDataset

import numpy
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

#Using RandomForest Regression
regressor = RandomForestRegressor(n_estimators=300, criterion='mse', max_depth=30)

#Fitting the model with training data
regressor.fit(X_train, y_train)

#Preprocessing test data and predicting final value
X_test = DataImputer().fit_transform(X_test)
y_pred = regressor.predict(X_test)

#Reading the file to write the predict values to 
pred = pd.read_csv('F:/Notes an Stuff/Intelligent Systems/Machine Learning/tcd ml 2019-20 income prediction submission file.csv')
pred['Income'] = y_pred

#Writing the predicted values to the Final Submission file
pred.to_csv('F:/Notes an Stuff/Intelligent Systems/Machine Learning/tcd ml 2019-20 income prediction submission file.csv', index=False)