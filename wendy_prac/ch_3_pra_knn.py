import gzip
import pickle
from datetime import datetime
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import joblib

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label = 'Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label = 'Recall')

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'g-', label = 'precision vs recall')


def plot_roc_curve(fpr, tpr, label = None): 
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0, 1], [0, 1], 'k--')

def conf_matrix_accuracy(np_matrix): 
    '''
    method: sum up the diagonal data, then divide by the sum of total
    output: the accuracy of the confusion matrix
    '''
    row_n, col_n = np_matrix.shape
    diag_sum = 0
    for i in range(col_n): 
        diag_sum += np_matrix[i, i]
    # print (diag_sum)
    # print (np_matrix.sum())
    return diag_sum/np_matrix.sum()

if __name__ == '__main__': 

    # =================
    # logging conf area
    # =================
    startTime = datetime.now()
    # LOG_FILENAME = 'logging_ch_3_pra.out'
    LOG_FILENAME = 'logging_example.out'

    logging.basicConfig(
        filename=LOG_FILENAME,
        level=logging.INFO,
        format='%(asctime)s %(message)s', 
        datefmt='%d/%m/%Y %H:%M:%S'
        )   

    module_logger = logging.getLogger(LOG_FILENAME)

    new_logging_area ='=' * 10
    module_logger.info(new_logging_area)

    # =================
    # alg start area
    # =================

    with gzip.open('mnist.pkl.gz', 'rb') as f: 
        (train_set, train_label), (valid_set, valid_label), (test_set, test_label) = pickle.load(f, encoding='latin1')
    
    
    # print (train_set.shape)
    clf = SGDClassifier(random_state = 0)
    forest_clf = RandomForestClassifier(random_state = 42)

    scaler = StandardScaler()

    knn_clf = KNeighborsClassifier()
    train_set_scaled = scaler.fit_transform(train_set)
    test_set_scaled = scaler.transform(test_set)
    
    param_grid = [
        {'n_neighbors': [1, 10, 100], 'weights': ['uniform', 'distance'], 'leaf_size': [3, 30, 300]}, 
        {'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'p': [1, 2]}
    ]

    fl_name = 'grid_search.sav'

    # # ========================
    # # grid search part
    # # ========================
    # grid_search = GridSearchCV(knn_clf, param_grid, cv = 5, scoring = 'accuracy', return_train_score = True)

    # grid_search.fit(train_set_scaled, train_label)
    
    # joblib.dump(grid_search, fl_name)
    # my_model_loaded = joblib.load(fl_name)

    # ========================
    # grid analysis part
    # SVM's parameter check
    # ========================
    try: 
        # # ================================================
        # # with best parameter to fit the whole training set
        # # ================================================
        # # knn_clf_new = KNeighborsClassifier(leaf_size=30, metric='minkowski',
        # #                 metric_params=None, n_jobs=None, n_neighbors=10, p=1,
        # #                 weights='distance')

        # knn_clf_new_book = KNeighborsClassifier(leaf_size=30, metric='minkowski',
        #                 metric_params=None, n_jobs=None, n_neighbors=4, p=2,
        #                 weights='distance')
        
        # knn_clf_new_book.fit(train_set_scaled, train_label)

        # joblib.dump(knn_clf_new_book, 'knn_new_book.sav')

        # # ================================================
        # # predict with the setting, and store the confusion matrix for further study
        # # ================================================
        # # knn_new_saved = joblib.load('knn_new.sav')
        
        # prediction = knn_clf_new_book.predict(test_set_scaled)


        # knn_book_con_matrix = confusion_matrix(prediction, test_label)

        # joblib.dump(knn_book_con_matrix, 'knn_con_bk_matrix.pkl')
        
        # ================================================
        # check the confusion matrix
        # ================================================
        # knn_con_matrix = joblib.load('knn_con_bk_matrix.pkl')
        knn_con_matrix = joblib.load('knn_con_matrix.pkl')
        knn_con_matrix_score = conf_matrix_accuracy(knn_con_matrix)
        # print (knn_con_matrix_score)
        # row_sums = knn_con_matrix.sum(axis = 1, keepdims = True)
        # norm_conf_mx = knn_con_matrix/ row_sums

        # np.fill_diagonal(norm_conf_mx, 0)

        # plt.matshow(norm_conf_mx, cmap = plt.cm.gray)
        # plt.savefig('knn_con_normal.png')
        
        # print (tn, fp, fn, tp)
        # print (knn_con_matrix.sum())
        # plt.matshow(knn_con_matrix, cmap = plt.cm.gray)
        # plt.savefig('knn_con_matrix.png')

        # with this parameter to check the train set accuracy first
        
        # find ways to adjust the parameter to raise the accuracy for the test set
    except: 
        logging.exception('message')
        
    # =================
    # logging endding area
    # =================
    finally: 
        dur = datetime.now() - startTime
        
        # it's like a period of time, in days / seconds / microseconds
        dur_info = 'KNN_whole_training DURATION: {} seconds'.format(dur.seconds)
        # dur_info = 'SCRIPT DURATION: {} minutes'.format(dur.seconds/60.0)
        
        module_logger.info(dur_info)
        module_logger.info(new_logging_area)