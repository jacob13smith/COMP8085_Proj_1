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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

field_cols = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm', 'attack_cat', 'Label']
feature_cols = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']
dtc_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
lr_cols = ['dur', 'sbytes', 'sttl', 'dttl', 'Sload', 'Sjit', 'Stime', 'Ltime', 'tcprtt', 'synack', 'ackdat', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
def main():
    parser = ArgumentParser(
                    prog = 'NIDS AI Program',
                    description = 'This program satisfies the requirements for Project 1 of COMP 8085')
    parser.add_argument('test_filename', help="Path to CSV file containing test data.")
    parser.add_argument('method', choices=['dtc','lr'], help="The classification method to use for testing the against the dataset.")
    parser.add_argument('task', choices=['Label', 'attack_cat'], help="The label to be predicted by the model.")
    parser.add_argument('model_name', help="Optional path to pickled, pre-trained model to load and use for test.")
    parser.add_argument('-fs', choices=['on', 'off'], default="on" ,dest="feature_selection_toggle", help="Optionally turn off feature selection to use all features in dataset.")
    args = parser.parse_args()

    loaded_model = pickle.load(open(args.model_name + '.' + args.task + '.sav','rb'))

    traffic = pd.read_csv(args.test_filename,names=field_cols, skiprows=1)

    # Normalize the data and labels
    traffic.replace(to_replace=r'^\s*$', value=0, inplace=True, regex=True)
    traffic.attack_cat.fillna(value='None', inplace=True)
    traffic.fillna(value=0, inplace=True)
    traffic = traffic[traffic['is_ftp_login'].isin([0, 1])]
    traffic_features = traffic[feature_cols]
    traffic.attack_cat = traffic.attack_cat.str.strip()
    traffic.attack_cat.replace(to_replace="Backdoors", value="Backdoor", inplace=True)
    attack_cat_encoder = preprocessing.LabelEncoder()
    attack_cat_encoder.fit(traffic.attack_cat)

    labels = {}
    labels['attack_cat'] = traffic.attack_cat
    labels['Label'] = traffic.Label

    # Categorical fields that need encoding
    traffic_features.srcip, _ = pd.factorize(traffic_features.srcip)
    traffic_features.sport, _ = pd.factorize(traffic_features.sport)
    traffic_features.dstip, _ = pd.factorize(traffic_features.dstip)
    traffic_features.dsport, _ = pd.factorize(traffic_features.dsport)
    traffic_features.proto, _ = pd.factorize(traffic_features.proto)
    traffic_features.state, _ = pd.factorize(traffic_features.state)
    traffic_features.service, _ = pd.factorize(traffic_features.service)

    if args.feature_selection_toggle == 'on':
        if args.method == 'dtc':
            prediction = loaded_model.predict(traffic_features[dtc_cols])
        if args.method == 'lr':
            scaler = joblib.load('scaler_model.pkl') 
            normalized_features = scaler.transform(traffic_features[lr_cols])
            loaded_model = joblib.load(args.model_name + '.' + args.task + '.sav')        
            prediction = loaded_model.predict(normalized_features)
    else:
        if args.method == 'dtc':
            prediction = loaded_model.predict(traffic_features)
        if args.method == 'lr':  
            scaler = joblib.load('scaler_model.pkl') 
            normalized_features = scaler.transform(traffic_features)
            loaded_model = joblib.load(args.model_name + '.' + args.task + '.sav')        
            prediction = loaded_model.predict(normalized_features)
            
    print(metrics.classification_report(labels[args.task], prediction))

    return 0

if __name__ == '__main__':
    main()