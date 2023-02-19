import warnings
warnings.filterwarnings("ignore")
import pandas as pd
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

field_cols = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm', 'attack_cat', 'Label']
feature_cols = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']
dtc_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']


def main():
    parser = ArgumentParser(
                    prog = 'COMP 8085 Project 1 model training program.',
                    description = 'This program trains a model with a given training set and classifier')
    parser.add_argument('train_filename', help="Path to CSV file containing training data.")
    parser.add_argument('method', choices=['dtc'], help="The classification method to use for testing against the dataset.")
    parser.add_argument('model_name', help="Optional path to pickled, pre-trained model to load and use for test.")
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
        
    elif args.method == 'OTHER CLASSIFIER':
        # Todo, implement the other classifiers
        return


    return 0

if __name__ == '__main__':
    main()