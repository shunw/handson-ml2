from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
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

from datetime import datetime
import glob
import gzip
import joblib
import logging
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
import pickle
import re
import shutil
import string
import tarfile


#Tokenization (a list of tokens), will be used as the analyzer
#1.Punctuations are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
#2.Stop words in natural language processing, are useless words (data).
def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''
    
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3
    return clean_words

def prepare_folder(): 
    '''
    this is to create the training/ testing folders
    create a train / test folder and copy the samples into them
        combine two easy_ham and spam together
    train -> easy_ham/ hard_ham/ spam
    labels: easy_ham 1; hard_ham 2; spam 0
    total file count: spam_2: 1396; spam: 500; easy_ham: 2500; easy_ham_2: 1400; hard_ham: 250
    '''

    # target folders
    root_dir = 'data/spam/spam_tt' # tt = train and test

    train_root_dir = 'spam_train'
    test_root_dir = 'spam_test'
    
    # source folder
    src_dir = 'data/spam/spam_src'

    # {'target_path': [src_path1, src_path2]}
    easy_ham_pdict = {'easy_ham': ['easy_ham', 'easy_ham_2']}
    hard_ham_pdict = {'hard_ham': ['hard_ham']}
    spam_pdict = {'spam': ['spam', 'spam_2']}
    all_sub_dict = [easy_ham_pdict, hard_ham_pdict, spam_pdict]

    for sub_d in all_sub_dict: 
        # sub_d = {'target_path': ['src_path1', 'src_path2']}
    
        
        for target_path, src_path_ls in sub_d.items():
            
            # make target path    
            train_target_path = os.path.join(root_dir, train_root_dir, target_path)
            os.makedirs(train_target_path)

            test_target_path = os.path.join(root_dir, test_root_dir, target_path)
            os.makedirs(test_target_path)
            

            # shuffle and prepare the source filenames
            for src_path_i in src_path_ls: 
                one_src_folder = os.path.join(src_dir, src_path_i)
                allFileNames = os.listdir(one_src_folder)
                
                if 'cmds' in allFileNames: 
                    allFileNames.remove('cmds')
                
                np.random.shuffle(allFileNames)
                
                train_flnames, test_flnames = np.split(np.array(allFileNames), [int(len(allFileNames) * .8)])
                
                train_FileNames_src = [os.path.join(src_dir, src_path_i, n) for n in train_flnames.tolist()]
                test_FileNames_src = [os.path.join(src_dir, src_path_i, n) for n in test_flnames.tolist()]

                # copy to the train/ test folder
                for n in train_FileNames_src: 
                    shutil.copy(n, train_target_path)
                for n in test_FileNames_src: 
                    shutil.copy(n, test_target_path)
    
def folder_2_data(): 
    '''
    read each files to database, label the files according to the folder name
    Only add the context/ subject => three cols in pd: head, context, label
    labels: easy_ham 1; hard_ham 2; spam 0
    '''
    root_path = 'data/spam/spam_tt'
    train_path = 'spam_train'
    folder = 'easy_ham'

    label_dict = {'easy_ham': 1, 'hard_ham': 2, 'spam': 0}
    
    fl_name = '0000[1-5]*'
    fl_path = os.path.join(root_path,train_path,folder,fl_name)

    cont_list = list()
    subj_list = list()

    for n in glob.glob(fl_path): 
        
        f = open(n,'r')
        
        cont = ''
        subject = ''
        i_start_c = 0

        for i, c in enumerate(f): 
            # clear_words = process_text(i)
            # message_bow = CountVectorizer(analyzer = process_text).fit_transform(i)
            # print (message_bow)
            
            if c.strip() == '': 
                i_start_c += i
            if i_start_c != 0: 
                cont += c.strip()
                
            if re.findall('^Subject.*', c) and i_start_c == 0: 
                subject = c.strip()
                # print (subject)
                
        # print (len(cont), len(subject))  
        
        f.close()

        cont_list.append(cont)
        subj_list.append(subject)
    
    y_label = label_dict[folder]
    y_label_ls = [y_label] * len(subj_list)
    
    df = pd.DataFrame({'subject': subj_list, 'content': cont_list, 'label': y_label_ls})
    # messages_bow = CountVectorizer(analyzer = process_text).fit_transform(df['content'])
    # df['sub&con'] = df['subject'] + df['content']
    
    # messages_bow1 = CountVectorizer(analyzer = process_text).fit_transform(df['sub&con'])
    # print (messages_bow.shape)
    # print (messages_bow1.shape)
    return df

class RefineAttributes(BaseEstimator, TransformerMixin): 
    '''purpose is to make the data easier
    '''
    def __init__(self, add_subject = True, lower_case = True): 
        # before that need to add the string to the pd
        # make URLs to "URL"
        # make numbers to "NUMBER"
        # need to think about if necessage and how to add punctuation
        
        self.add_subject = add_subject
        self.lower_case = lower_case
    
    def fit(self, X, y = None): 
        return self
    

    def transform(self, X, y = None): 
        pass



if __name__ == '__main__': 

    # # =================
    # # logging conf area
    # # =================
    # startTime = datetime.now()
    # # LOG_FILENAME = 'logging_ch_3_pra.out'
    # LOG_FILENAME = 'logging_example.out'

    # logging.basicConfig(
    #     filename=LOG_FILENAME,
    #     level=logging.INFO,
    #     format='%(asctime)s %(message)s', 
    #     datefmt='%d/%m/%Y %H:%M:%S'
    #     )   

    # module_logger = logging.getLogger(LOG_FILENAME)

    # new_logging_area ='=' * 10
    # module_logger.info(new_logging_area)

    # =================
    # data prepare DONE
    # =================
    # 20% to the test folder/ 80% in the train folder
    # prepare_folder()
    
    # =================
    # train folder files to data
    # =================
    df = folder_2_data() # only have one folder to test about 10 data/ # only have two col, subject col and content col
    # print (df.shape)

    
    # # =================
    # # transform the data col
    # # =================
    # refine_data = RefineAttributes()