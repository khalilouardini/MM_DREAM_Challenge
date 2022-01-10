import logging
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import xgboost as xg


def fit_cv_clf(inputs, model, n_estimators):
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
    task: str
        Classification or Ensembling
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
    X, y = inputs

    accs = []
    aucs = []

    kfold = KFold(n_splits=5)
    kfold = KFold(n_splits=5)

    kfold.get_n_splits(X)

    for train_index, valid_index in kfold.split(X):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        if model == 'RF':
            logging.info("Using RandomForest()...")
            clf =  RandomForestClassifier(**hyperparameters_rf)
        else:
            logging.info("Using XGBRegressor()...")
            clf = xg.XGBClassifier(**hyperparameters_xg)


        t0 = time.time()

        logging.info("Fitting Ensemble")
        clf.fit(X_train, y_train)
        logging.info("Fit in %0.3fs" % (time.time() - t0))

        logging.info("Inference on validation set")
        y_pred = clf.predict(X_valid)

        # Metrics
        acc = balanced_accuracy_score(y_pred, y_valid)
        fpr, tpr, _ = roc_curve(y_pred, y_valid)
        auc_score = auc(fpr, tpr)
        accs.append(acc)
        aucs.append(auc_score)
        print("Balanced Accuracy: {} | AUC: {}".format(acc, auc_score))
            
    logging.info("=== Balanced accuracy : mean = {} ; std = {} ===".format(np.mean(accs), np.std(accs)))
    logging.info("=== AUC : mean = {} ; std = {} ===".format(np.mean(aucs), np.std(aucs)))


    return clf, accs, aucs

def fit_cv_ensemble(inputs, inputs_censored, model, n_estimators):
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
    task: str
        Classification or Ensembling
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

    accs = []
    aucs = []

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

        logging.info("Fitting Ensemble")
        clf.fit(X_train, y_train)
        regressor_os.fit(X_train_regr, y_os_train)
        regressor_pfs.fit(X_train_regr, y_pfs_train)
        logging.info("Fit in %0.3fs" % (time.time() - t0))

        logging.info("Inference on validation set")
        y_pred_hr = clf.predict(X_valid)
        # Prediction for D_OS
        y_pred_os = regressor_os.predict(X_valid)
        y_pred_os = y_pred_os <= 18*30
        # Prediction for PFS
        y_pred_pfs = regressor_pfs.predict(X_valid)
        y_pred_pfs = y_pred_pfs <= 18*30
        # Prediction for regressions
        y_pred_regr = np.logical_or(y_pred_os, y_pred_pfs).astype(int)
        # Final prediction
        y_pred = np.logical_or(y_pred_hr, y_pred_regr).astype(int)

        # Metrics
        acc = balanced_accuracy_score(y_pred, y_valid)
        fpr, tpr, _ = roc_curve(y_pred, y_valid)
        auc_score = auc(fpr, tpr)
        accs.append(acc)
        aucs.append(auc_score)
        print("Balanced Accuracy: {} | AUC: {}".format(acc, auc_score))
            
    logging.info("=== Balanced accuracy : mean = {} ; std = {} ===".format(np.mean(accs), np.std(accs)))
    logging.info("=== AUC : mean = {} ; std = {} ===".format(np.mean(aucs), np.std(aucs)))


    return clf, regressor_os, regressor_pfs, accs, aucs

def fit_train(train_df, keep_features, model):
    """Fits a regressor on the data using a 5-fold cross validation.
    Parameters
    ----------
    train_df: pandas.DataFrame
    keep_features: List[str]
        List of features to keep in the training set
    model: str
        Model to train ('RF'=Random Forest and 'XG'= XGBoost)
    Returns
    -------
            trained regressor and predictions on the test set
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
    X_train, X_test, y_train, y_test = data_preparation(train_df, keep_features)

    if model == 'RF':
        logging.info("Using RandomForest()...")
        rf = RandomForestRegressor()
    else:
        logging.info("Using XGBRegressor()...")
        rf = xg.XGBRegressor()
    rf_random = RandomizedSearchCV(estimator = rf,
                                param_distributions = random_grid,
                                n_iter = 10,
                                cv = 3,
                                verbose=2,
                                random_state=42,
                                n_jobs = -1
                                )
    rf_random.fit(X_train, y_train)

    logging.info("Best parameters {} for {} model".format(model, rf_random.best_params_))
    best_model = rf_random.best_estimator_
    y_pred_test = best_model.predict(X_test)

    logging.info("Running inference with best {} model".format(model))
    r = pearsonr(y_test, y_pred_test)
    mape_test = mape_error(np.exp(y_test), np.exp(y_pred_test))
    rmse_test = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred_test)))
    mae_test = mean_absolute_error(np.exp(y_test), np.exp(y_pred_test))

    logging.info("===On the test set: MAPE={} | RMSE={} | MAE={} | Pearson (Log) r = {}===".format(str(mape_test),
                str(rmse_test), 
                str(mae_test), 
                str(r))
                )

    return best_model, mape_test, rmse_test, mae_test