import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

from data_clean_script import *

X,y = split_with_bow()
X_train_scaled,X_test_scaled,y_train,y_test = rescale_train_test(X,y)

lr = LogisticRegression()


lr.fit(X_train_scaled,y_train)
y_preds = lr.predict_proba(X_test_scaled)[:,1]

fpr, tpr,thresh = roc_curve(y_test, y_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.title('Logistic Regression')
plt.plot([0,1],[0,1])
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.draw()
plt.savefig('logreg_roc_auc.eps',format='eps')