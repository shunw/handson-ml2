import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from scipy import stats


class housing_pred(object):
    def __init__(self):
        # this is played by Wendy
        path = '~/github/handson-ml2/datasets/housing/housing.csv'
        self.df = pd.read_csv(path)
        
        self.col_names_ls = self.df.columns.to_list()
        self.ocean_near = 'ocean_proximity'
        
        self.in_cat = 'income_cat'
        self.m_income = 'median_income'
        self.m_h_value = 'median_house_value'
        self.t_bd_room = 'total_bedrooms'
        self.t_room = 'total_rooms'
        self.households = 'households'
        self.bd_per_rm = 'bedrooms_per_room'
        self.pp_per_household = 'population_per_household'
        self.r_per_household = 'rooms_per_household'
        self.pp = 'population'
    
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
        
        print (np.sqrt(np.sum(np.power(y_pred - y_test, 2))/len(y_test)))

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
        
        self._error_cal(lin_reg, X_test, y_test)
        self._error_cal(lin_reg, X_train, y_train)
        
    def b_sample_data_insights(self): 
        '''
        THINGS I DID NOT THINK ABOUT:
        1. stratified the test for the test data. 

        '''

        #===============================
        # this is to prepare the test set
        #===============================
        self.df[self.in_cat] = pd.cut(self.df[self.m_income], bins = [0., 1.5, 3.0, 4.5, 6.0, np.inf], labels = [1, 2, 3, 4, 5])
        
        self.split = StratifiedShuffleSplit(n_splits = 1, test_size = .2, random_state = 42)
        
        for self.train_index, self.test_index in self.split.split(self.df, self.df[self.in_cat]): 
            self.strat_train_set = self.df.loc[self.train_index]
            self.strat_test_set = self.df.loc[self.test_index]
        for set_ in (self.strat_train_set, self.strat_test_set): 
            set_.drop(self.in_cat, axis = 1, inplace = True)

        # #===============================
        # # this is to check the corr
        # #===============================
        # corr_matrix = self.strat_train_set.corr()
        
        # self.strat_train_set[self.r_per_household] = self.strat_train_set[self.t_room] / self.strat_train_set[self.households]

        # self.strat_train_set[self.bd_per_rm] = self.strat_train_set[self.t_bd_room]/ self.strat_train_set[self.t_room]

        # self.strat_train_set[self.pp_per_household] = self.strat_train_set[self.pp]/ self.strat_train_set[self.households]

        # corr_matrix = self.strat_train_set.corr()
        
    def b_sample_prepare_data(self): 
        # =============================
        # saperate the label and the features
        # =============================
        self.X_train = self.strat_train_set.drop(self.m_h_value, axis = 1)
        self.y_train = self.strat_train_set[self.m_h_value].copy()

        # =============================
        # deal with the na data
        # =============================
        
        # option 1: drop the related row
        self.X_train.dropna(subset = [self.t_bd_room])

        # option 2: drop the feature
        self.X_train.drop(self.t_bd_room, axis = 1)

        # option 3: fill in with the median data
        median = self.X_train[self.t_bd_room].median()
        self.X_train[self.t_bd_room].fillna(median, inplace = True)

        # # another method to deal w/ option 3
        
        imputer = SimpleImputer(strategy = 'median')
        self.X_train_num = self.X_train.drop(self.ocean_near, axis = 1)
        
        imputer.fit(self.X_train_num)
        imputer_x = imputer.transform(self.X_train_num)
        self.X_train_tr = pd.DataFrame(imputer_x, columns = self.X_train_num.columns)
        
        # =============================
        # deal with the categorial attributes
        # =============================
        cat_encoder = OneHotEncoder()        
        self.X_train_cat = self.X_train[[self.ocean_near]]
        self.X_train_cat_1hot = cat_encoder.fit_transform(self.X_train_cat)
        cat_name_array = cat_encoder.categories_
        
        
        # =============================
        # custom transformers
        # =============================
        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = True)
        self.X_train_extra_attribs = attr_adder.transform(self.X_train.values)
        # print (self.X_train_extra_attribs.shape)
        # print (self.X_train.shape)

        # =============================
        # feature scaling & transformation pipelines
        # =============================

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy = 'median')), 
            ('attribs_adder', CombinedAttributesAdder()), 
            ('std_scaler', StandardScaler())
        ])
        self.X_train_num_tr = num_pipeline.fit_transform(self.X_train_num)

        num_attribs = list(self.X_train_num)
        
        cat_attribs = [self.ocean_near]

        self.full_pipeline = ColumnTransformer([
            ('num', num_pipeline, num_attribs), 
            ('cat', OneHotEncoder(), cat_attribs)
        ])

        self.X_train_prepared = self.full_pipeline.fit_transform(self.X_train)
    def _display_scores(self, scores): 
        print ('Scores: {}'.format(scores))
        print ('Mean: {}'.format(scores.mean()))
        print ('Std: {}'.format(scores.std()))

    def b_sample_train_data(self): 
        # =============================
        # linear regression is under fit
        # =============================
        lin_reg = LinearRegression()
        lin_reg.fit(self.X_train_prepared, self.y_train)

        # =============================
        # RMSE
        # =============================
        lin_pred = lin_reg.predict(self.X_train_prepared)
        lin_mse = mean_squared_error(self.y_train, lin_pred)
        lin_rmse = np.sqrt(lin_mse)
        # print (lin_rmse)
        # self._error_cal(lin_reg, self.X_train_prepared, self.y_train)
        
        # check the cv as below
        scores = cross_val_score(lin_reg, self.X_train_prepared, self.y_train, scoring = 'neg_mean_squared_error', cv = 10)
        lin_reg_scores = np.sqrt(-scores)
        # self._display_scores(lin_reg_scores)

        # =============================
        # try complex model due to linear regression under fit
        # decision tree
        # =============================
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(self.X_train_prepared, self.y_train)
        tree_pred = tree_reg.predict(self.X_train_prepared)
        tree_mse = mean_squared_error(self.y_train, tree_pred)
        tree_rmse = np.sqrt(tree_mse)
        
        # =============================
        # try cross-validation feature due to rmse = 0
        # =============================
        scores = cross_val_score(tree_reg, self.X_train_prepared, self.y_train, scoring = 'neg_mean_squared_error', cv = 10)
        tree_rmse_scores = np.sqrt(-scores)
        # print (tree_rmse_scores)

        # =============================
        # try another model due to decision tree over fit
        # Random Forests
        # =============================
        forest_reg = RandomForestRegressor()
        forest_reg.fit(self.X_train_prepared, self.y_train)
        forest_pred = forest_reg.predict(self.X_train_prepared)
        forest_mse = mean_squared_error(self.y_train, forest_pred)
        forest_rmse = np.sqrt(forest_mse)
        print (forest_rmse)
        scores = cross_val_score(forest_reg, self.X_train_prepared, self.y_train, scoring = 'neg_mean_squared_error', cv = 10)
        forest_rmse_scores = np.sqrt(-scores)
        self._display_scores(forest_rmse_scores)

    def b_sample_finetune_data(self): 
        # =============================
        # grid search
        # =============================
        param_grid = [
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, 
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
        ]
        forest_reg = RandomForestRegressor()

        grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)

        grid_search.fit(self.X_train_prepared, self.y_train)
        
        # # show only the param defined. 
        # print (grid_search.best_params_)
        
        # # show all the parameter
        # print (grid_search.best_estimator_)

        # show all the results with the params
        cvres = grid_search.cv_results_
        # for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
        #     print (np.sqrt(-mean_score), params) 

        self.final_model = grid_search.best_estimator_
    
    def b_sample_test_set_evaluate(self):
        self.X_test = self.strat_test_set.drop(self.m_h_value, axis = 1)
        self.y_test = self.strat_test_set[self.m_h_value].copy()

        self.X_test_prepared = self.full_pipeline.transform(self.X_test)

        self.final_predictions = self.final_model.predict(self.X_test_prepared)
        self.final_mse = mean_squared_error(self.y_test, self.final_predictions)
        self.final_rmse = np.sqrt(self.final_mse)
        print (self.final_rmse)

        confidence = .95
        squared_errors = (self.final_predictions - self.y_test) ** 2
        confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc = squared_errors.mean(), scale = stats.sem(squared_errors)))
        print (confidence_interval)


        
    def b_exercised(self): 
        ''' question 1
        Try a Support Vector Machine regressor (sklearn.svm.SVR), 
        with various hyperparameters such as kernel="linear" (with various values for the C hyperparameter) or kernel="rbf" (with various values for the C and gamma hyperparameters). 
        Don’t worry about what these hyperparameters mean for now. 
        How does the best SVR predictor perform?
        '''

        '''question 2
        Try replacing GridSearchCV with RandomizedSearchCV.
        '''
        pass

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin): 
    def __init__(self, add_bedrooms_per_room = True): # no *args or ** kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y = None): 
        return self # nothing else to do
    
    def transform(self, X, y = None): 
        rooms_per_household = X[:, rooms_ix]/ X[:, households_ix]
        population_per_household = X[:, population_ix]/ X[:, households_ix]
        if self.add_bedrooms_per_room: 
            bedrooms_per_room = X[:, bedrooms_ix]/ X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else: 
            return np.c_[X, rooms_per_household, population_per_household]
    



if __name__ == '__main__':
    housing = housing_pred()
    housing.b_sample_data_insights()
    housing.b_sample_prepare_data()
    # housing.b_sample_train_data()
    housing.b_sample_finetune_data()
    housing.b_sample_test_set_evaluate()

    # housing.routine_step()
    '''
    till some part before the randomized search
    '''
