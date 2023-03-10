import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics # is used to create classification results
from sklearn.tree import export_graphviz # is used for plotting the decision tree
from six import StringIO # is used for plotting the decision tree
from IPython.display import Image # is used for plotting the decision tree
from IPython.core.display import HTML # is used for showing the confusion matrix
import pydotplus # is used for plotting the decision tree
from argparse import ArgumentParser
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

field_cols = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm', 'attack_cat', 'Label']
feature_cols = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']
dtc_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
lr_cols = ['dur', 'sbytes', 'sttl', 'dttl', 'Sload', 'Sjit', 'Stime', 'Ltime', 'tcprtt', 'synack', 'ackdat', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

def main():
    parser = ArgumentParser(
                    prog = 'COMP 8085 Project 1 model training program.',
                    description = 'This program trains a model with a given training set and classifier')
    parser.add_argument('train_filename', help="Path to CSV file containing training data.")
    parser.add_argument('method', choices=['dtc', 'lr'], help="The classification method to use for testing against the dataset.")
    parser.add_argument('model_name', help="Name of saved pickle next to this script.")
    parser.add_argument('-fs', choices=['on', 'off'], default="on" ,dest="feature_selection_toggle", help="Optionally turn off feature selection to use all features in dataset.")
    args = parser.parse_args()

    traffic = pd.read_csv("traffic_train.csv",names=field_cols, skiprows=1)

    # Normalize the data and labels
    traffic.replace(to_replace=r'^\s*$', value=0, inplace=True, regex=True)
    traffic.attack_cat.fillna(value='None', inplace=True)
    traffic.fillna(value=0, inplace=True)
    traffic_features = traffic[feature_cols]
    traffic.attack_cat = traffic.attack_cat.str.strip()
    traffic.attack_cat.replace(to_replace="Backdoors", value="Backdoor", inplace=True)
    attack_cat_encoder = preprocessing.LabelEncoder()
    attack_cat_encoder.fit(traffic.attack_cat)

    # Categorical fields that need encoding
    traffic_features.srcip, _ = pd.factorize(traffic_features.srcip)
    traffic_features.sport, _ = pd.factorize(traffic_features.sport)
    traffic_features.dstip, _ = pd.factorize(traffic_features.dstip)
    traffic_features.dsport, _ = pd.factorize(traffic_features.dsport)
    traffic_features.proto, _ = pd.factorize(traffic_features.proto)
    traffic_features.state, _ = pd.factorize(traffic_features.state)
    traffic_features.service, _ = pd.factorize(traffic_features.service)

    # Training labels
    attack_cat_train = traffic.attack_cat
    label_train = traffic.Label

    if args.method == 'dtc':
        # Train the Decision Tree Classifiers (Jacob)
        clf = DecisionTreeClassifier(criterion="entropy")
        if args.feature_selection_toggle == 'on':
            clf.fit(traffic_features[dtc_cols], label_train)
        else:
            clf.fit(traffic_features, label_train)

        # save the label model to disk
        filename = args.model_name + '.Label.sav'
        pickle.dump(clf, open(filename, 'wb'))

        if args.feature_selection_toggle == 'on':
            clf.fit(traffic_features[dtc_cols], attack_cat_train)
        else:
            clf.fit(traffic_features, attack_cat_train)

        # save the attack_cat prediction model to disk
        filename = args.model_name + '.attack_cat.sav'
        pickle.dump(clf, open(filename, 'wb'))
        
    elif args.method == 'lr':
        # Train the Logistic Regression model
        lr = LogisticRegression()
        # Define the parameter grid to search over
        param_grid = {'C': [10], 'penalty': ['l2']}

        # Scaler
        scaler = StandardScaler()
        
        
        if args.feature_selection_toggle == 'on':
            # Set up a grid search with cross-validation to find the best hyperparameters
            gs = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1)
            # Fit the grid search to the training data
            X_train = scaler.fit_transform(traffic_features[lr_cols])
            gs.fit(X_train, label_train)
            lr = gs.best_estimator_
        else:
            # Set up a grid search with cross-validation to find the best hyperparameters
            gs = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1)
            # Fit the grid search to the training data
            X_train = scaler.fit_transform(traffic_features)
            gs.fit(X_train, label_train)
            lr = gs.best_estimator_
        # save the label model to disk
        filename = args.model_name + '.Label.sav'
        pickle.dump(lr, open(filename, 'wb'))

        
        # Set up the logistic regression classifier
        
        lr = LogisticRegression()
        if args.feature_selection_toggle == 'on':
            # Set up a grid search with cross-validation to find the best hyperparameters
            gs = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1)
            # Fit the grid search to the training data
            X_train = scaler.fit_transform(traffic_features[lr_cols])
            gs.fit(X_train, attack_cat_train)
            lr = gs.best_estimator_
        else:
            # Set up a grid search with cross-validation to find the best hyperparameters
            gs = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1)
            # Fit the grid search to the training data
            X_train = scaler.fit_transform(traffic_features)
            gs.fit(X_train, attack_cat_train)
            lr= gs.best_estimator_

        # save the attack_cat prediction model to disk
        filename = args.model_name + '.attack_cat.sav'
        joblib.dump(lr, open(filename, 'wb'))
        joblib.dump(scaler, 'scaler_model.pkl')
        
        return

    return 0

if __name__ == '__main__':
    main()