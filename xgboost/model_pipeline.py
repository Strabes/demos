from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

def create_pipeline(numeric_preds, object_preds, drop_cols):
    p = Pipeline([
    ("str_cleaner",TransformWrapper(lambda x: str_cleaner_df(x,variables = object_preds))),
    ("num_nan_ind",AddMissingIndicator(
        variables = numeric_preds, missing_only = True)),
    ("fill_cat_nas",CategoricalImputer(
        variables = object_preds, fill_value = 'MISSING')),
    ("pcb",PercentThresholdBinner(variables=object_preds,percent_threshold = 0.01)),
    ("max_level_bin",MaxLevelBinner(variables=object_preds,max_levels=15)),
    ("rmmean",MeanMedianImputer(variables=numeric_preds)),
    ("drop_date",DropFeatures(features_to_drop=drop_cols)),
    ("drop_quasi_constant",DropConstantFeatures(tol=0.97)),
    ("one_hot_encoder", OneHotEncoder()),
    ("name_formatter", TransformWrapper(rename)),
    ("xgb", XGBClassifier(min_child_weight=1, gamma=0, objective= 'binary:logistic',
        nthread=4, scale_pos_weight=1, seed=1, gpu_id=0, tree_method = 'gpu_hist'))])

    return p


def name_tracker(p, X):
    cols_in = X.columns.tolist()
    df = pd.DataFrame({"cols":cols_in,"cols_in":cols_in})
    df = pd.concat([
        df,
        pd.DataFrame({
            "cols":[i+"_na" for i in p["num_nan_ind"].variables_],
            "cols_in":p["num_nan_ind"].variables_})
            ])
    df = pd.concat([
        df,
        pd.DataFrame(set().union(*[
            [(k,k + "_" + i) for i in v] for k,v in p['one_hot_encoder'].encoder_dict_.items()]),         columns = ["cols","cols_in"])
            ])
    return df