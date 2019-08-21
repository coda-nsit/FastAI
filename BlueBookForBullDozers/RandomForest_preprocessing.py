from fastai.tabular import *

import os

from BlueBookForBullDozers.utilities import train_cats

PATH = "bluebook-for-bulldozers"

# parse_dates = list of all the columns that might contain dates
# low_memory = False, use more memory to read more data to figure out data types of the columns
df_raw = pd.read_csv(f'{PATH}/Train.csv', low_memory=False, parse_dates=["saledate"])

# see the data
# display_all(df_raw.tail().transpose())

# since the evaluation for this competition is root mean square log error (RMSLE) take the log
df_raw.SalePrice = np.log(df_raw.SalePrice)

# remove saledate and add information that can be extracted from the saledate like saleDay, saleWeek, saleDayofweek, saleDayOfYear etc
add_datepart(df_raw, 'saledate')
# display_all(df_raw.head(5))

# convert all the columns that are strings to category
train_cats(df_raw)
# check that the strings indeed got converted to categories
# print(df_raw.UsageBand.cat.categories)
# print(df_raw.UsageBand.cat.codes)

# prints the categories in [High, Low, Medium] which is weird so, reorder the categories,
# inplace = True, changes the dataframe rather than creating a new one
# ordered = True, tell that the categories logically follow a order i.e. High > Medium > Low and -1 for missing/NA
df_raw.UsageBand.cat.set_categories(["High", "Medium", "Low"], ordered=True, inplace=True)

# df_raw.drop('SalePrice', axis=1) = dependent variable. axis=1 means remove column
# df_raw.SalePrice = independent variable we want to predict

# for each column find % of null values
# display_all(df_raw.isnull().sum().sort_index() / len(df_raw))

# save the current df
os.makedirs("tmp", exist_ok=True)
df_raw.to_feather("tmp/raw")
