from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np

########################################
# ExtraTreesRegressor: rather than trying every variable, try few splits with few variables thus reducing correlation between the individual trees
########################################

from BlueBookForBullDozers.utilities import proc_df, split_vals, print_score, draw_tree, plt, set_rf_samples

# read the dataframe from the saved feather file
df_raw = pd.read_feather("tmp/raw")

# create the instance of the model we want
# if we remove max_depth, #leaves = #rows in data set ie. highly over fit trees
# m = RandomForestRegressor(n_jobs=-1, n_estimators=1, max_depth=3, bootstrap=False)
# min_samples_leaf = stop when number of samples in a leaf becomes <=3
# max_features = only use half of the columns at every spilt
m = ExtraTreesRegressor(n_jobs=-1, n_estimators=40, min_samples_leaf=3, max_features=0.5, oob_score=False)

# replace categorical variables by their numeric codes, handle missing values and separate out SalePrice
df, y, _ = proc_df(df_raw, y_fld="SalePrice")

# create validation set and the final training set
n_validation = 12000
n_training   = len(df) - n_validation
X_training, X_validation = split_vals(df, n_training)
y_training, y_validation = split_vals(y, n_training)

# rather than limit the training data to the first 30k rows or any continuous subset, sample random 20k rows.
# This way the if there are enough number of trees we can get access to entire data set.
set_rf_samples(20000)

print(X_training.shape, y_training.shape, X_validation.shape)
print("\n")

# drop the independent variable <SalePrice> as m.fit(df_dependent, df_independent)
m.fit(X_training, y_training)
print_score(m, X_training, y_training, X_validation, y_validation)
print("\n")

# get the predictions from each individual tree, np.stack concatenates the predictions of each tree on a new axis
predictions = np.stack([tree.predict(X_validation) for tree in m.estimators_])
# will print [10 predictions (one for each tree)], mean of the predictions, actual value
print(predictions[:, 0], np.mean(predictions[:, 0]), y_validation[0])
print("\n")

# plot the r^2 scores for avg predictions of trees[0:i], showing as the # of estimators increases, the r^2 increases
plt.plot([metrics.r2_score(y_validation, np.mean(predictions[:i + 1], axis=0)) for i in range(len(m.estimators_))])
plt.show()

draw_tree(m.estimators_[0], df, precision=3)
print("\n")
