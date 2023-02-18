import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics # is used to create classification results
from sklearn.tree import export_graphviz # is used for plotting the decision tree
from six import StringIO # is used for plotting the decision tree
from IPython.display import Image # is used for plotting the decision tree
from IPython.core.display import HTML # is used for showing the confusion matrix
import pydotplus # is used for plotting the decision tree

field_cols = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm', 'attack_cat', 'Label']
feature_cols = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']

def main():
    traffic = pd.read_csv("traffic_set.csv",names=field_cols, skiprows=1)

    # Normalize the data and labels
    traffic.replace(to_replace=r'^\s*$', value=0, inplace=True, regex=True)
    traffic.attack_cat.fillna(value='None', inplace=True)
    traffic.fillna(value=0, inplace=True)
    traffic_features = traffic[feature_cols]
    traffic.attack_cat = traffic.attack_cat.str.strip()
    traffic.attack_cat.replace(to_replace="Backdoors", value="Backdoor", inplace=True)
    attack_cat_encoder = preprocessing.LabelEncoder()
    attack_cat_encoder.fit(traffic.attack_cat)

    attack_cat = traffic.attack_cat
    label = traffic.Label

    # Categorical fields that need encoding
    traffic_features.srcip, _ = pd.factorize(traffic_features.srcip)
    traffic_features.sport, _ = pd.factorize(traffic_features.sport)
    traffic_features.dstip, _ = pd.factorize(traffic_features.dstip)
    traffic_features.dsport, _ = pd.factorize(traffic_features.dsport)
    traffic_features.proto, _ = pd.factorize(traffic_features.proto)
    traffic_features.state, _ = pd.factorize(traffic_features.state)
    traffic_features.service, _ = pd.factorize(traffic_features.service)

    # Label Test
    X_train, X_test, y_train, y_test = train_test_split(traffic_features, label, test_size=0.3)

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Attack label accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred)*100))

    # Attack Cat Test
    features_train, features_test, cat_train,cat_test = train_test_split(traffic_features, attack_cat, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(features_train, cat_train)
    
    y_pred = clf.predict(features_test)
    print("Attack category accuracy: {:.2f}%\n".format(metrics.accuracy_score(cat_test, y_pred)*100))

    return 0

if __name__ == '__main__':
    main()