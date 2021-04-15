import pandas as pd


class ColumnTypes:
    def __init__(self, df):
        self.dtype_dict = self.get_dtype_dict(df)
        self._numeric_cols = self.get_numeric_cols()
        self._object_cols = self.get_object_cols()
        self._datetime_cols = self.get_datetime_cols()

    @staticmethod
    def get_dtype_dict(df):
        dtype_dict = df.dtypes.to_dict()
        return dtype_dict

    
    def get_numeric_cols(self):
        numeric_cols = [k for k,v in self.dtype_dict.items()
            if pd.api.types.is_numeric_dtype(v)]
        numeric_cols.sort()
        return numeric_cols

    def get_object_cols(self):
        object_cols = [k for k,v in self.dtype_dict.items()
            if pd.api.types.is_object_dtype(v)]
        object_cols.sort()
        return object_cols

    def get_datetime_cols(self):
        datetime_cols = [k for k,v in self.dtype_dict.items()
            if pd.api.types.is_datetime64_any_dtype(v)]
        datetime_cols.sort()
        return datetime_cols

    @property
    def numeric_cols(self):
        return self._numeric_cols

    @property
    def object_cols(self):
        return self._object_cols

    @property
    def datetime_cols(self):
        return self._datetime_cols

    @property
    def other_cols(self):
        all_cols = self.dtype_dict.keys()
        other_cols = [c for c in all_cols if c not in
            self.numeric_cols + self.object_cols + self.datetime_cols]
        return other_cols

    def report(self):
        print("Numeric columns: " + ", ".join(self.numeric_cols) + "\n")
        print("Object columns: " + ", ".join(self.object_cols) + "\n")
        print("Datetime columns: " + ", ".join(self.datetime_cols) + "\n")
        print("Columns not accounted for: " + ", ".join(self.other_cols) + "\n")