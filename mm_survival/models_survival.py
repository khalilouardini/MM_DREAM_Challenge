import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
import logging

from sklearn.model_selection import KFold

def survival_prediction(preds, threshold=540):
    """Computes classification predicitons from survival functions .
    Parameters
    ----------
    preds: List[np.array]
        List of predicted survival functions for each patient in the tesrt set 
    Returns
    -------
            a vector of predicted HR_FLAG
    """
    pred_hr_valid = []
    for pred in preds:
        for i, t in enumerate(pred.x):
            if t >= threshold:
                p = 1 - pred.y[i]
                pred_hr_valid.append(p > 0.5)
                break
    return pred_hr_valid

def fit_cv_survival(inputs):
    """Fits a Survival analysis model (Penalized Cox-regression) on the data using a 5-fold cross validation.
    Parameters
    ----------
    inputs: List[pandas.DataFrame]
        List of datframe inputs
    Returns
    -------
            trained model and cross-validation metrics
    """
    df_train, df_train_censored, df_clin_uncensored, df_clin_censored = inputs

    # "Death Status"
    df_clin_censored['D_Status'] = df_clin_censored['D_OS_FLAG'].astype(bool)
    df_clin_uncensored['D_Status'] = df_clin_uncensored['D_OS_FLAG'].astype(bool)

    # 1. Prepare labels for survival analysis
    data_y = np.array([(df_clin_uncensored.iloc[i]['D_Status'].astype(bool), df_clin_uncensored.iloc[i]['D_OS']) for i in range(df_clin_uncensored.shape[0])],
                 dtype=[('D_Status', bool), ('D_OS', np.int64)])
    data_y_censored = np.array([(df_clin_censored.iloc[i]['D_Status'].astype(bool), df_clin_censored.iloc[i]['D_OS']) for i in range(df_clin_censored.shape[0])],
                 dtype=[('D_Status', bool), ('D_OS', np.int64)])
    # High-Risk label
    y_hr = df_clin_uncensored['HR_FLAG'].replace({'TRUE': 1, 'FALSE': 0}).values

    # 2. Prepare inputs
    data_x = df_train.copy()
    data_x_censored  = df_train_censored.copy()

    # 3. Cross-validation
    accs, aucs, precisions, recalls = [], [], [], []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold.get_n_splits(data_x)

    for train_index, valid_index in kfold.split(data_x):
        X_train, X_valid = data_x.values[train_index], data_x.values[valid_index]
        y_train, y_valid = data_y[train_index], data_y[valid_index]
        y_train_hr, y_valid_hr = y_hr[train_index], y_hr[valid_index]
        # Augment training set with censored data
        X_train = np.vstack([X_train, data_x_censored.values])
        y_train = np.hstack([y_train, data_y_censored])

        # Fit model
        estimator = CoxPHSurvivalAnalysis(alpha=0.01)
        estimator.fit(X_train, y_train)

        # Predictions
        preds_valid = estimator.predict_survival_function(X_valid)
        y_pred = survival_prediction(preds_valid)
        # Metrics
        acc = accuracy_score(y_pred, y_valid_hr)
        fpr, tpr, _ = roc_curve(y_pred, y_valid_hr)
        auc_score = auc(fpr, tpr)
        recall = recall_score(y_pred, y_valid_hr)
        precision = precision_score(y_pred, y_valid_hr)
            
        accs.append(acc), recalls.append(recall), precisions.append(precision), aucs.append(auc_score)
        print("Accuracy: {} | AUC: {} | Precision: {} | Recall: {}".format(acc, auc_score, precision, recall))

    logging.info("=== Accuracy : mean = {} ; std = {} ===".format(np.mean(accs), np.std(accs)))
    logging.info("=== AUC : mean = {} ; std = {} ===".format(np.mean(aucs), np.std(aucs)))
    logging.info("=== Precision : mean = {} ; std = {} ===".format(np.mean(precisions), np.std(precisions)))
    logging.info("=== Recall : mean = {} ; std = {} ===".format(np.mean(recalls), np.std(recalls)))
    
    return estimator, accs, aucs, precisions, recalls

    
    




