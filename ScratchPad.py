from Util import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize
import matplotlib.pyplot as plt
from scipy import stats

# using the same example as above
# df = pd.DataFrame({'country': ['russia', 'germany', 'australia','korea','germany']})
#
# xx = pd.get_dummies(df,prefix=['country'], drop_first=True)
#
# print(xx)

fati = pd.read_csv('#############\\4_Data_OneHotEncoded.csv')

# df = df[(df[['A','C']] != 0).all(axis=1)]
# fati = fati[(fati['CurrBalance'] >= -20000) & (fati['CurrBalance'] <= 20000)]
#
# print(fati['CurrBalance'].min())
# print(fati['CurrBalance'].max())

hist = fati.hist(column='CreditTurnover', bins=10000)

plt.show(hist)


# normalized_X = normalize(scaled_X, norm='l1', axis=1, copy=True)


# fati = pd.read_csv('#############\\2_FATI_MonthCat.csv')
# fatiJoin = pd.DataFrame(columns=fati.columns)
#
# print(fatiJoin)

#
# dd = parse_billingenddate_datetime(x)
#
# print(dd.day)

# df = pd.DataFrame([{'c1':10, 'c2':100}, {'c1':11,'c2':110}, {'c1':12,'c2':120}])
#
# for index, row in df.iterrows():
#     print()
#     print(row['c1'], row['c2'])