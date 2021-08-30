import os
from re import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
 

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def split_train_test(data, test_ratio):
    np.random.seed(42) 
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

housing_data_path="C:/Mic/book/handson-ml-master/datasets/housing/"
housing_data_file_name="housing.csv"

housing=pd.read_csv((housing_data_path+housing_data_file_name))
#housing.plot(kind="scatter", x="longitude", y="latitude")
#print(housing.head())
#print(housing.info())

#housing["median_house_value"].hist(bins=50)
#plt.show()

print(housing.columns)

housing=housing[['median_income','ocean_proximity','median_house_value']].copy()
housing=housing[:100]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")


#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    #s=housing["population"]/100, label="population",
    #c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    #)
#plt.legend()
#plt.show()


housing_num = train_set.drop("ocean_proximity", axis=1)
housing_num = housing_num.drop("median_house_value", axis=1)
#print(housing_num.head())
housing_labels = train_set["median_house_value"].copy()
#median = housing_num["total_bedrooms"].median()
#housing_num["total_bedrooms"].fillna(median)
#imputer = SimpleImputer(strategy="median")
#imputer.fit(housing_num)

#housing_cat = housing["ocean_proximity"]
#encoder = LabelEncoder()
#ousing_cat_encoded = encoder.fit_transform(housing_cat)
#print(housing_cat_encoded)
#print(encoder.classes_)

#attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
#housing_extra_attribs = attr_adder.transform(housing.values)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    #('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('one_hot_encoder', OneHotEncoder(sparse=False))
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

#housing_num_tr = num_pipeline.fit_transform(housing_num)
#print(housing_num_tr)
#print(housing_num_tr.shape)

housing_prepared = full_pipeline.fit_transform(train_set)
print(housing_prepared.shape)
print(housing_labels.shape)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

test_set_x=test_set.drop("median_house_value", axis=1)
housing_prepared_topredict = full_pipeline.fit_transform(test_set_x)
prediction=(lin_reg.predict(housing_prepared_topredict))
print(prediction)
test_set_y = test_set["median_house_value"].copy()
print(test_set_y)

lin_mse = mean_squared_error(test_set_y, prediction)
print(lin_mse)

#prediction.hist(bins=50)
x=np.array([i for i in range(len(prediction))])
width=0.3
x2=x-width
plt1=plt.subplot()
plt2=plt.subplot()
plt1.bar(x, prediction, width=width,color='orange',label='predict')
plt2.bar(x2,test_set_y, width=width,color='blue',label='label')
plt.legend()
plt.show()