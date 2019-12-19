import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


class housing_pred(object):
    def __init__(self):
        # this is played by Wendy
        path = '~/github/handson-ml2/datasets/housing/housing.csv'
        self.df = pd.read_csv(path)
        
        self.col_names_ls = self.df.columns.to_list()
        self.ocean_near = 'ocean_proximity'
        
    
    def hist_plot(self):
        # # this version is not as good as below one.
        # fig, ax = plt.subplots(nrows = 3, ncols = 3)
        # idx = 0
        # for row in ax: 
        #     for col in row: 
        #         col.hist(self.df[self.col_names_ls[idx]], bins = 100)
        #         idx += 1
        
        self.df.hist(bins = 50, figsize = (20, 15))

        plt.savefig('hist_sum.png')
    
    def _error_cal(self, model, X_test, y_test): 
        y_pred = model.predict(X_test)
        print (np.sum(np.power(y_pred - y_test, 2)))


    def routine_step(self): 
        '''
        COMMENT: 
        1. 直接用train test + linear regression的效果并不好，error很大
        2. PCA用了之后，score更差，怀疑是我没有用对。。。
        坐等书中案例
        '''
        self.x_col_list = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_near_num']
        self.y_col = 'median_house_value'

        # make the one-hot-encoding/ make the encoding sort by alphabet, and check which one is better
        self.df[self.ocean_near] = pd.Categorical(self.df[self.ocean_near])
        self.df['ocean_near_num'] = self.df[self.ocean_near].cat.codes

        # remove the null value from data/ later need to check if fill in the null value and compare the performance
        self.df = self.df.dropna(how = 'any', axis = 0)

        # split the training dataset and the test dataset
        X_train, X_test, y_train, y_test = train_test_split(self.df[self.x_col_list], self.df[self.y_col], test_size = .3, random_state = 42)
        
        # standarize the scale
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)

        # ? use PCA? 
        pca = PCA(n_components = 2)
        X_train_pca = pca.fit_transform(X_train_scaler)
        # print (pca.explained_variance_ratio_)
        # print (pca.singular_values_)

        # use the ski learn w/ linear regression
        lin_reg = LinearRegression().fit(X_train_pca, y_train)
        print (lin_reg.score(X_train_pca, y_train))

        lin_reg = LinearRegression().fit(X_train_scaler, y_train)
        print (lin_reg.score(X_train_scaler, y_train))
        
        # self._error_cal(lin_reg, X_test, y_test)
        # # self._error_cal(lin_reg, X_train, y_train)
        
    def book_sample(self): 
        '''
        THINGS I DID NOT THINK ABOUT:
        1. stratified the test for the test data. 

        '''

        # create a test set
        in_cat = 'income_cat'
        self.df['income_cat'] = pd.cut(self.df['median_income'], bins = [0., 1.5, 3.0, 4.5, 6.0, np.inf], labels = [1, 2, 3, 4, 5])
        # self.df['income_cat'].hist()
        # plt.savefig('income_cat.png')
        split = StratifiedShuffleSplit(n_splits = 1, test_size = .2, random_state = 42)
        
        for train_index, test_index in split.split(self.df, self.df[''])



if __name__ == '__main__':
    housing = housing_pred()
    housing.book_sample()
    '''
    看到 create a test set 最后一点部分
    '''
