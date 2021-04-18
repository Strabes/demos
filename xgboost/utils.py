import numpy as np
import pandas as pd

def datetime_preprocess(df):
    df = df.copy()
    dt_cols = ['issue_d','earliest_cr_line']
    df.loc[:,dt_cols] = df.loc[:,dt_cols] \
      .applymap(lambda x: np.nan if x in ['null','NULL',''] else x) \
      .apply(lambda x: pd.to_datetime(x,format = "%b-%Y"))
    return(df)


def str_cleaner(s):
    if s is None: return('MISSING')
    elif s is np.nan: return('MISSING')
    elif isinstance(s,str):
        s = s.strip().lower()
        if s in ['null','']: return('MISSING')
        else: return(s)
    else:
        raise(TypeError("Found element of type " + type(s)))

def str_cleaner_df(df,variables = None):
    if isinstance(variables,str): variables = [variables]
    elif variables is None:
        variables = [k for k,v in df.dtypes.to_dict().items()
            if pd.api.types.is_object_dtype(v)]
    df.copy()
    df.loc[:,variables] = df.loc[:,variables].applymap(str_cleaner)
    return(df)