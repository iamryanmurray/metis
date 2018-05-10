from data_clean_script import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle


X,y = split_with_bow()
X_train, X_test, y_train, y_test = rescale_train_test(X,y)


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt','auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 50, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5, 10,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4,10]
# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf = RandomForestClassifier(n_jobs=-1)
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = 20, cv = 3, verbose=2, 
                               n_jobs = -1,scoring='roc_auc')



rf_random.fit(X_train,y_train)

print(rf_random.best_params_)

with open('random_forest_model.pkl', 'wb') as handle:
    pickle.dump(rf_random, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('random_forest_model_params.pkl', 'wb') as handle:
    pickle.dump(rf_random.best_params_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    

'''best_prob = rf_random.predict_proba(X_test)[:,1]

fpr, tpr,thresh = roc_curve(y_test, best_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.title('Random Forest')
plt.plot([0,1],[0,1])
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.draw()
plt.savefig('random_forest_2.eps')'''

