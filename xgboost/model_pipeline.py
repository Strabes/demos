import warnings
from typing import Union
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.selection import DropConstantFeatures, DropFeatures
from xgboost.sklearn import XGBClassifier
from utils import str_cleaner_df


def create_pipeline(numeric_preds, object_preds):
    """
    Create sklearn.pipeline.Pipeline

    Parameters
    ----------
    numeric_preds : Union[str, list]
        columns name(s) of numeric predictor(s)

    object_preds : Union[str, list]
        columns name(s) of object predictor(s)

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
        ("num",p_num,numeric_preds),
        ("cat",p_cat,object_preds)
    ]

    # combine pipelines and add XGBClassifier
    col_transforms = ColumnTransformer(transformers)
    p = Pipeline([
        ("col_transformers",col_transforms),
        ("xgb", XGBClassifier(min_child_weight=1, gamma=0, objective= 'binary:logistic',
        nthread=4, scale_pos_weight=1, seed=1, gpu_id=0, tree_method = 'gpu_hist'))
    ])

    return p


def name_tracker(p, X, numeric_preds, object_preds):
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

    numeric_preds : list
        column names of numeric predictor inputs to the pipeline

    object_preds : list
        column names of object predictor inputs to the pipeline

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

    # Put everythin together
    if len(numeric_preds) > 0:
        final_num_cols = (p["col_transformers"]
            .transformers_[0][1]
            .transform(X.head(1)[numeric_preds])
            .columns.tolist())
    else:
        final_num_cols = []

    if len(object_preds) > 0:
        final_obj_cols = (p["col_transformers"]
            .transformers_[1][1]
            .transform(X.head(1)[object_preds])
            .columns.tolist())
    else:
        final_obj_cols = []

    df_ = pd.DataFrame({"final_cols": final_num_cols + final_obj_cols})

    df = pd.merge(df_,df,left_on="final_cols",right_on="cols").loc[:,["final_cols","cols_in"]]

    return df


def feature_importances(p, X, numeric_preds, object_preds):
    """
    Get feature_importances in a pandas.DataFrame

    Parameters
    ----------
    p : sklearn.pipeline.Pipeline

    X : pandas.DataFrame

    numeric_preds : list

    object_preds : list

    Returns
    -------
    pandas.DataFrame
    """
    feature_importances_ = p.steps[-1][1].feature_importances_
    df = name_tracker(p, X, numeric_preds, object_preds)
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
    def __init__(self):
        self.rfe_results = []

    @staticmethod
    def _run_single_fit(X,y,numeric_preds,object_preds):
        p = create_pipeline(numeric_preds,object_preds)
        X = X.copy(deep=True)
        X = X[numeric_preds + object_preds]
        p.fit(X,y)
        return p

    @staticmethod
    def _var_imp(p, X, numeric_preds, object_preds):
        fe = feature_importances(p, X, numeric_preds, object_preds)
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

    def update_preds(self, p, X, numeric_preds, object_preds):
        var_imp = self._var_imp(p, X, numeric_preds, object_preds)
        vars_to_keep = self.rfe_schedule(var_imp)
        numeric_preds = [v for v in numeric_preds if v in vars_to_keep]
        object_preds = [v for v in object_preds if v in vars_to_keep]
        return numeric_preds, object_preds


    def run_rfe(self,X_train,y_train,X_val,y_val,numeric_preds,object_preds, supp_warnings = True):
        curr_preds = numeric_preds + object_preds
        while len(curr_preds) > 0:
            print("Fitting model with {} features".format(len(curr_preds)))
            #print("numeric_preds length = {}".format(len(numeric_preds)))
            #print("object_preds length = {}".format(len(object_preds)))
            if supp_warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = self._run_single_fit(X_train,y_train,numeric_preds,object_preds)
            else:
                p = self._run_single_fit(X_train,y_train,numeric_preds,object_preds)
            eval_metric = self.get_eval_metric(p,X_val[curr_preds],y_val)
            numeric_preds, object_preds = self.update_preds(
                p, X_val[curr_preds].head(),numeric_preds, object_preds)
            preds_to_drop = [v for v in curr_preds if v not in numeric_preds + object_preds]
            self.rfe_results.append(
                {"n_features":len(curr_preds),
                 "preds_to_drop":preds_to_drop,
                 "eval_metric":eval_metric,
                 "vars_to_keep" : numeric_preds + object_preds})
            curr_preds = numeric_preds + object_preds