from model_pipeline import create_pipeline

def rfe_schedule(curr_features):
    n_features = len(curr_features)
    if n_features > 100:
        updated_features = curr_features[:100]
    elif n_features > 50:
        updated_features = curr_features[:(n_features-10)]
    elif n_features > 20:
        updated_features = curr_features[:(n_features-5)]
    elif n_features > 10:
        updated_features = curr_features[:(n_features-2)]
    elif n_features > 1:
        updated_features = curr_features[:(n_features-1)]

class RFE:
    def __init__(self,p):
        self.p = p

    def _run_single_fit(self,X,y,numeric_preds,object_preds):
        X = X.copy(deep=True)
        X = X[numeric_preds + object_preds]
        p.fit()
