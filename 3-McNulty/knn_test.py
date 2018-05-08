from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from data_clean_script import *

X,y = split_with_bow()
X_train_scaled,X_test_scaled,y_train,y_test = rescale_train_test(X,y)

_,X_new,_,y_new = train_test_split(X_train_scaled,y_train,test_size=.2,stratify=y,random_state=5)

k_range = [(n+1)*5 for n in range(6)]
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_new,y_new, cv=10, scoring='roc_auc')
    k_scores.append(scores.mean())
print(k_scores)

best_n = (k_scores.index(max(k_scores))+1)*5


import matplotlib.pyplot as plt

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.draw()
plt.savefig('knn_cv_roc_auc.eps',format='eps')