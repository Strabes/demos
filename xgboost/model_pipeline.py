import warnings
from typing import Union
import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn import metrics
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.selection import DropConstantFeatures, DropFeatures
from xgboost.sklearn import XGBClassifier
from utils import str_cleaner_df


def create_pipeline(params : dict = None):
    """
    Create sklearn.pipeline.Pipeline

    Parameters
    ----------
    params : dict
        dictionary of parameters for the pipeline

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    # pipeline for numeric variables
    p_num = Pipeline([
        ("num_nan_ind",AddMissingIndicator(missing_only = True)),
        ("rmmean",MeanMedianImputer()),
        ("drop_quasi_constant",DropConstantFeatures(tol=0.97))
        ])

    # pipeline for categorical variables
    p_cat = Pipeline([
        ("fill_cat_nas",CategoricalImputer(fill_value = 'MISSING')),
        ("rlc",RareLabelEncoder()),
        ("one_hot_encoder", OneHotEncoder())
    ])

    # list of pipelines to combine
    transformers = [
        ("num",p_num,make_column_selector(dtype_include = np.number)),
        ("cat",p_cat,make_column_selector(dtype_include = object))
    ]

    # combine pipelines and add XGBClassifier
    col_transforms = ColumnTransformer(transformers)
    p = Pipeline([
        ("col_transformers",col_transforms),
        ("xgb", XGBClassifier(min_child_weight=1, gamma=0, objective= 'binary:logistic',
        nthread=4, scale_pos_weight=1, seed=1, gpu_id=0, tree_method = 'gpu_hist'))
    ])

    if params:
        p.set_params(**params)
    return p


def name_tracker(p, X):
    """
    Track names through pipeline. This function is
    specific to the architecture of the given pipeline.
    If the architecture of the pipeline changes, this
    function will need to change.

    TODO: Figure out if this can be made
    pipeline-architecture independent

    Parameters
    ----------
    p : sklearn.pipeline.Pipeline
        must have already been fit

    X : pandas.DataFrame
        the input to the pipeline

    Returns
    -------
    pandas.DataFrame
        contains feature importances
    """
    cols_in = X.columns.tolist()
    df = pd.DataFrame({"cols":cols_in,"cols_in":cols_in})

    # Indicators for Missing Numeric Columns
    nni = p['col_transformers'].transformers_[0][1]['num_nan_ind']
    try:
        nan_num_ind = pd.DataFrame({
            "cols":[i+"_na" for i in nni.variables_],
            "cols_in":nni.variables_})
        df = pd.concat([df, nan_num_ind])
    except:
        pass

    # Onehot encoding of categorical columns
    one = p['col_transformers'].transformers_[1][1]['one_hot_encoder']
    one_hot_encoder = pd.DataFrame(set().union(*[
        [(k + "_" + i, k) for i in v]
        for k,v in one.encoder_dict_.items()]),
        columns = ["cols","cols_in"])
    df = pd.concat([df,one_hot_encoder])

    # Put everything together
    numeric_preds = p['col_transformers'].transformers_[0][2]
    if len(numeric_preds) > 0:
        final_num_cols = (p["col_transformers"]
            .transformers_[0][1]
            .transform(
                X.head(1)[numeric_preds])
            .columns.tolist())
    else:
        final_num_cols = []

    object_preds = p['col_transformers'].transformers_[1][2]
    if len(object_preds) > 0:
        final_obj_cols = (p["col_transformers"]
            .transformers_[1][1]
            .transform(X.head(1)[object_preds])
            .columns.tolist())
    else:
        final_obj_cols = []

    df_ = pd.DataFrame({"final_cols": final_num_cols + final_obj_cols})

    df = (pd.merge(df_,df,left_on="final_cols",right_on="cols")
            .loc[:,["final_cols","cols_in"]])

    return df


def feature_importances(p, X):
    """
    Get feature_importances in a pandas.DataFrame

    Parameters
    ----------
    p : sklearn.pipeline.Pipeline

    X : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    feature_importances_ = p.steps[-1][1].feature_importances_
    df = name_tracker(p, X)
    df["feature_importances"] = feature_importances_
    return df


def _rename(s):
    if isinstance(s,str):
        return re.sub("[^A-Za-z0-9_]","_",s)
    elif hasattr(s,"__iter__"):
        return [re.sub("[^A-Za-z0-9_]","_",i) for i in s]
    else:
        raise ValueError("s must be str or iterable")


def rename(df):
    df = df.copy(deep=True)
    df.columns = [_rename(i) for i in df.columns.tolist()]
    return df


class RFE:
    def __init__(self, params = None):
        self.rfe_results = []
        self.params = params

    def _run_single_fit(self,X,y):
        if self.params:
            p = create_pipeline(self.params)
        else:
            p = create_pipeline()
        p.fit(X,y)
        return p

    @staticmethod
    def _var_imp(p, X):
        fe = feature_importances(p, X)
        var_imp = (fe.groupby("cols_in")[["feature_importances"]]
            .sum().sort_values("feature_importances",ascending=False)
            .index.tolist())
        return var_imp

    @staticmethod
    def rfe_schedule(curr_features):
        n_features = len(curr_features)
        if n_features > 100:
            vars_to_keep = curr_features[:100]
        elif n_features > 50:
            vars_to_keep = curr_features[:(n_features-10)]
        elif n_features > 20:
            vars_to_keep = curr_features[:(n_features-5)]
        elif n_features > 10:
            vars_to_keep = curr_features[:(n_features-2)]
        elif n_features > 1:
            vars_to_keep = curr_features[:(n_features-1)]
        else:
            vars_to_keep = []
        return vars_to_keep

    @staticmethod
    def get_eval_metric(p,X,y):
        z = pd.DataFrame(
                p.predict_proba(X),
                columns=['prob1','prob2'])
        roc_auc = metrics.roc_auc_score(1 - y, z.prob1.values)
        return roc_auc

    def update_preds(self, p, X):
        var_imp = self._var_imp(p, X)
        vars_to_keep = self.rfe_schedule(var_imp)
        return vars_to_keep


    def run_rfe(self,X_train,y_train,X_val,y_val, supp_warnings = True):
        curr_preds = X_train.columns.tolist()
        while len(curr_preds) > 0:
            print("Fitting model with {} features".format(len(curr_preds)))
            #print("numeric_preds length = {}".format(len(numeric_preds)))
            #print("object_preds length = {}".format(len(object_preds)))
            if supp_warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = self._run_single_fit(X_train,y_train)
            else:
                p = self._run_single_fit(X_train,y_train)
            eval_metric = self.get_eval_metric(p,X_val,y_val)
            vars_to_keep = self.update_preds(p, X_val.head())
            preds_to_drop = [v for v in curr_preds if v not in vars_to_keep]
            self.rfe_results.append(
                {"n_features":len(curr_preds),
                 "preds_to_drop":preds_to_drop,
                 "eval_metric":eval_metric,
                 "vars_to_keep" : vars_to_keep})
            curr_preds = vars_to_keep
            X_train = X_train.copy().loc[:,curr_preds]
            X_val = X_val.copy().loc[:,curr_preds]