rom data_clean_script import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier 
import matplotlib.pyplot as plt
import pickle



X,y = split_with_bow()
X_train, X_test, y_train, y_test = rescale_train_test(X,y)


params = {'depth':[2,4,6,8,10],
          'iterations':[250,100,500,],
          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50],
          }




cb = CatBoostClassifier()

cb_random = RandomizedSearchCV(estimator = cb, 
                               param_distributions = parms, 
                               n_iter = 10, cv = 3, verbose=10, 
                               n_jobs = -1,scoring='roc_auc')



cb_random.fit(X_train,y_train)

print(cb_random.best_params_)

with open('random_forest_model.pkl', 'wb') as handle:
    pickle.dump(rf_random, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('random_forest_model_params.pkl', 'wb') as handle:
    pickle.dump(rf_random.best_params_, handle, protocol=pickle.HIGHEST_PROTOCOL)