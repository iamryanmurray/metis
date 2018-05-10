from data_clean_script import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle


X,y = split_with_bow()
X_train, X_test, y_train, y_test = rescale_train_test(X,y)



rf = RandomForestClassifier(bootstrap=True, class_weight="balanced", criterion='gini', max_depth=50, 
	max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
	min_samples_leaf=2, min_samples_split=10, min_weight_fraction_leaf=0.0, 
	n_estimators=200, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)

rf.fit(X_train,y_train)

with open('random_forest_model_balanced.pkl', 'wb') as handle:
    pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)