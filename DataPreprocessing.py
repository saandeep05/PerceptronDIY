import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def reset(x):
    if x > 0:
        return 1
    return 0


water_df = pd.read_csv('waterQuality1.csv')
cols_list = list(water_df.columns)
# water_df = water_df.dropna()
# water_df.reset_index(inplace=True)

for i in range(len(water_df['is_safe'])):
    if water_df['is_safe'][i] == '#NUM!':
        water_df.drop(labels=i, axis=0, inplace=True)

try:
    water_df = water_df.drop(labels='index', axis=1)
except:
    # print("Ok")
    pass

scaler = StandardScaler()
water_df = pd.DataFrame(scaler.fit_transform(water_df))
# water_df.columns = cols_list
water_df.iloc[:, -1] = water_df.iloc[:, -1].apply(lambda x: reset(x))

X_train, X_test, y_train, y_test = train_test_split(water_df.drop(labels=len(water_df.columns)-1, axis=1),
                                                    water_df.iloc[:, -1], train_size=0.7, random_state=17)
# print(X_train.iloc[:, -1])
