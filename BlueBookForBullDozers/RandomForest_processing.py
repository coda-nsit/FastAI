from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# read the dataframe from the saved feather file
from BlueBookForBullDozers.utilities import proc_df, split_vals, print_score

df_raw = pd.read_feather("tmp/raw")

# create the instance of the model we want
m = RandomForestRegressor(n_jobs=-1)

# replace categorical variables by their numeric codes, handle missing values and separate out SalePrice
df, y, _ = proc_df(df_raw, y_fld="SalePrice")

# create training and validation sets
n_validation = 12000
n_training   = len(df) - n_validation

raw_training, raw_validation = split_vals(df_raw, n_training)
X_training, X_validation     = split_vals(df, n_training)
y_training, y_validation     = split_vals(y, n_training)

print(X_training.shape, y_training.shape, X_validation.shape)

# drop the independent variable <SalePrice> as m.fit(df_dependent, df_independent)
m.fit(df, y)
print_score(m, X_training, y_training, X_validation, y_validation)
