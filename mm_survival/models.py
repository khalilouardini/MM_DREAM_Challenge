import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import xgboost as xg

def fit_cv_clf(inputs, model, n_estimators):
    """Fits a classifier on the data using a 5-fold cross validation.
    Parameters
    ----------
    inputs: List[np.array]
        Data
    model: str
        Model to train ('RF'=Random Forest, 'XG'=XGBoost and 'logreg'=Logistic Regression)
    n_estimators: int
        Number of trees for RF
    Returns
    -------
            trained classifier and evaluation metrics
    """
    if model == 'RF':
        hyperparameters_rf = {'n_estimators': n_estimators, 
                'n_jobs': -1, 
                'verbose': 1,
                'max_features': 'auto',
                'max_depth': 60 
                }
    else:
        hyperparameters_xg = {'learning_rate': 0.01,
                          'n_estimators' : n_estimators,
                          'max_depth' :60,
                          'n_jobs': -1,
                        }


    logging.info("Data preparation")
    X, y = inputs

    accs, aucs, recalls, precisions  = [], [], [], []

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold.get_n_splits(X)

    for train_index, valid_index in kfold.split(X):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        if model == 'RF':
            logging.info("Using RandomForest()...")
            clf =  RandomForestClassifier(**hyperparameters_rf)
        elif model == 'XG':
            logging.info("Using XGBClassifier()...")
            clf = xg.XGBClassifier(**hyperparameters_xg)
        elif model == 'logreg':
            logging.info("Using Logistic Regression...")
            clf = LogisticRegression(max_iter=1000)
            
        t0 = time.time()

        logging.info("Fitting Classifier")
        clf.fit(X_train, y_train)
        logging.info("Fit in %0.3fs" % (time.time() - t0))

        logging.info("Inference on validation set")
        y_pred = clf.predict(X_valid)

        # Metrics
        acc = accuracy_score(y_pred, y_valid)
        fpr, tpr, _ = roc_curve(y_pred, y_valid)
        precision = precision_score(y_pred, y_valid)
        recall = recall_score(y_pred, y_valid)
        auc_score = auc(fpr, tpr)

        accs.append(acc), aucs.append(auc_score), precisions.append(precision), recalls.append(recall)
        print("Accuracy: {} | AUC: {} | Precision: {} | Recall: {}".format(acc, auc_score, precision, recall))
            
    logging.info("=== Accuracy : mean = {} ; std = {} ===".format(np.mean(accs), np.std(accs)))
    logging.info("=== AUC : mean = {} ; std = {} ===".format(np.mean(aucs), np.std(aucs)))
    logging.info("=== Precision : mean = {} ; std = {} ===".format(np.mean(precisions), np.std(precisions)))
    logging.info("=== Recall : mean = {} ; std = {} ===".format(np.mean(recalls), np.std(recalls)))

    return clf, accs, aucs, precisions, recalls

def fit_cv_ensemble(inputs, inputs_censored, model, n_estimators, include_censored):
    """Fits a regressor on the data using a 5-fold cross validation.
    Parameters
    ----------
    inputs: List[np.array]
        Data
    inputs_censored: List[np.array]
        Censored data (for regression)
    model: str
        Model to train ('RF'=Random Forest and 'XG'= XGBoost)
    n_estimators: int
        Number of trees
    include_censored: bool
        Whether to include the censored data for the regression
    Returns
    -------
            trained regressor and predictions on the test set
    """
    if model == 'RF':
        hyperparameters_rf = {'n_estimators': n_estimators, 
                'n_jobs': -1, 
                'verbose': 1,
                'max_features': 'auto',
                'max_depth': 60 
                }
    else:
        hyperparameters_xg = {'learning_rate': 0.01,
                          'n_estimators' :n_estimators,
                          'max_depth' :100,
                          'n_jobs': -1
                        }


    logging.info("Data preparation")
    X, y_hr, y_os, y_pfs = inputs
    X_censored, y_os_censored, y_pfs_censored = inputs_censored

    accs, aucs, recalls, precisions  = [], [], [], []

    kfold = KFold(n_splits=5)
    kfold = KFold(n_splits=5)

    kfold.get_n_splits(X)

    for train_index, valid_index in kfold.split(X):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y_hr[train_index], y_hr[valid_index]

        # Regression data
        y_os_train, _ = y_os[train_index], y_os[valid_index]
        y_pfs_train, _ = y_pfs[train_index], y_pfs[valid_index]
        # Stack censored data
        if include_censored:
            y_os_train = np.hstack([y_os_train, y_os_censored])
            y_pfs_train = np.hstack([y_pfs_train, y_pfs_censored])
            X_train_regr = np.vstack([X_train, X_censored])

        if model == 'RF':
            logging.info("Using RandomForest()...")
            clf =  RandomForestClassifier(**hyperparameters_rf)
            regressor_os = RandomForestRegressor(**hyperparameters_rf)
            regressor_pfs = RandomForestRegressor(**hyperparameters_rf)
        else:
            logging.info("Using XGBRegressor()...")
            clf = xg.XGBClassifier(**hyperparameters_xg)
            regressor_os = xg.XGBRegressor(**hyperparameters_xg)
            regressor_pfs = xg.XGBRegressor(**hyperparameters_xg)


        t0 = time.time()
        if not include_censored:
            regressor_os.fit(X_train, y_os_train)
            regressor_pfs.fit(X_train, y_pfs_train)
        else:
            regressor_os.fit(X_train_regr, y_os_train)
            regressor_pfs.fit(X_train_regr, y_pfs_train)

        logging.info("Fit in %0.3fs" % (time.time() - t0))

        logging.info("Inference on validation set")
        # Prediction for D_OS
        y_os_train = regressor_os.predict(X_train)
        y_os_valid = regressor_os.predict(X_valid)
        #y_pred_os = y_pred_os < 18*30
        # Prediction for PFS
        y_pfs_train = regressor_pfs.predict(X_train)
        y_pfs_valid = regressor_pfs.predict(X_valid)
        #y_pred_pfs = y_pred_pfs < 18*30
        # Prediction for regressions
        #y_pred_regr = np.logical_or(y_pred_os, y_pred_pfs).astype(int)
        # Final prediction
        # Train meta-model on the base features + predictions

        X_meta_train = np.hstack([X_train, y_os_train.reshape(-1, 1), y_pfs_train.reshape(-1,1)])
        X_meta_valid = np.hstack([X_valid, y_os_valid.reshape(-1, 1), y_pfs_valid.reshape(-1,1)])
        logging.info("Fitting Ensemble (Meta-Model)")
        clf.fit(X_meta_train, y_train)
        y_pred = clf.predict(X_meta_valid)
        #y_pred = np.logical_or(y_pred_hr, y_pred_regr).astype(int)

        # Metrics
        acc = accuracy_score(y_pred, y_valid)
        fpr, tpr, _ = roc_curve(y_pred, y_valid)
        auc_score = auc(fpr, tpr)
        precision = precision_score(y_pred, y_valid)
        recall = recall_score(y_pred, y_valid)
        accs.append(acc), aucs.append(auc_score), precisions.append(precision), recalls.append(recall)
        print("Accuracy: {} | AUC: {} | Precision: {} | Recall: {}".format(acc, auc_score, precision, recall))
            
    logging.info("=== Accuracy : mean = {} ; std = {} ===".format(np.mean(accs), np.std(accs)))
    logging.info("=== AUC : mean = {} ; std = {} ===".format(np.mean(aucs), np.std(aucs)))
    logging.info("=== Precision : mean = {} ; std = {} ===".format(np.mean(precisions), np.std(precisions)))
    logging.info("=== Recall : mean = {} ; std = {} ===".format(np.mean(recalls), np.std(recalls)))


    return clf, regressor_os, regressor_pfs, accs, aucs, precisions, recalls

def fit_cv_search(inputs, model):
    """Fits a classifier on the data using a 5-fold cross validation.
    Parameters
    ----------
    train_df: pandas.DataFrame
    model: str
        Model to train ('RF'=Random Forest and 'XG'= XGBoost)
    Returns
    -------
            Model initialized with the hyperparameters found with Random Search
    """
    if model == 'RF':
        random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
            'max_features': ['auto', 'sqrt']
            }
    else:
        random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
            'learning_rate': [0.1, 0.01, 0.001]
            }

    logging.info("Data preparation")
    X, y = inputs

    if model == 'RF':
        logging.info("Using RandomForest()...")
        rf = RandomForestClassifier()
    else:
        logging.info("Using XGBRegressor()...")
        rf = xg.XGBClassifer()
    rf_random = RandomizedSearchCV(estimator = rf,
                                param_distributions = random_grid,
                                n_iter = 20,
                                cv = 5,
                                verbose=2,
                                random_state=42,
                                n_jobs = -1
                                )
    rf_random.fit(X, y)

    logging.info("Best parameters {} for {} model".format(model, rf_random.best_params_))
    best_model = rf_random.best_estimator_


    return best_model, rf_random.best_params_