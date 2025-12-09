# Implementing Cross-validation 

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone

skfolds = StratifiedGroupKFold(n_splits=3)


for train_index , test_index in skfolds.sptt(x_train,y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train_5[train_index]
    x_test_fold = x_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(x_train_folds,y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred =0 y_test_fold)
    print(n_correct/len(y_pred))