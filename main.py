import pandas as pd  # Пандас
#import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy
import itertools

from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики

# Логистическая регрессия
from sklearn.linear_model import LogisticRegression
# Линейный дискриминант
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# Квадратичный дискриминант
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


COUNT = 200

# Get number in lottery
data = pd.read_csv('loto_stat.csv', delimiter=';', encoding='cp1251', header=0, index_col=0)
data = data[-COUNT:]
print(range(len(data)))
print('############')

data['SUM'] = data.sum(axis=1)
print('DATA WITH SUM')
print(data)
print('############ /n')
data['lottery_num'] = data.index

# Append SUM of numbers in each lottery
res = pd.DataFrame(columns=['DIR'], index=data.index)
for i in range(1, len(data)):
    if (data['SUM'].iloc[i] > data['SUM'].iloc[i - 1]):
        res.loc[data['lottery_num'].iloc[i]] = 1
    else:
        res.loc[data['lottery_num'].iloc[i]] = -1
res = res[1:]

# print('RESULT COLUM')
# print(len(res))
# print(res)
# print('#################')

#plt.plot(data['SUM'])
#plt.savefig('demo.png', bbox_inches='tight')

# Make new Data_Model - get indexing from previous
data_for_model = pd.DataFrame(index=data.index)
data_for_model['SUM'] = data['SUM']
# New Column, wich append column with % of change sum from previous
data_for_model['SUM_prc'] = abs(data['SUM'].pct_change()) * 100.0

# Get Procent CHANGE for NetWork
prc = pd.DataFrame(columns=['DIR'], index=data.index)
for i in range(1, len(data)):
    prc.loc[data['lottery_num'].iloc[i]] = round(data_for_model['SUM_prc'].iloc[i])
prc = prc[1:]

# Replace 0 in SUM_prc with small values
#for i, x in enumerate(data_for_model['SUM_prc']):
#    if (abs(x) < 0.0001):
#        data_for_model['SUM_prc'][i] = 0.0001

print(data_for_model.head())

x = data_for_model
y = res.values.ravel()
y = y.astype('int')

x_train = x[1:COUNT - 2]
x_test = x[COUNT -1:]
y_train = y[1:COUNT - 2]
y_test = y[-1]

#print('X TRAIN')
#print(x_train)
#print('X TEST')
#print(x_test)

#print('Y TRAIN')
#print(y_train)
#print('Y TEST')
#print(y_test)

d = pd.DataFrame(index=x_test.index)  # Set for check model
d['FACT'] = res[1:]  # Real values

# Visual
print('##########')
print('##########')

# Logistic Regression
model1 = LogisticRegression()
model1.fit(x_train, y_train)  # Обучение (подбор параметров модели)

d['Predict_LR'] = model1.predict(x_test)  # Тест
# Считаем процент правильно предсказанных направлений изменения цены:
d["Correct_LR"] = (1.0 + d['Predict_LR'] * d['FACT']) / 2.0
print(d)
hit_rate1 = np.mean(d["Correct_LR"])
print("Процент верных предсказаний: %.1f%%" % (hit_rate1 * 100))

# Linear Discriminant
model2 = LDA()
model2.fit(x_train, y_train)  # Обучение (подбор параметров модели)
d['Predict_LDA'] = model2.predict(x_test)  # Тест

# Считаем процент правильно предсказанных направлений изменения цены:
d["Correct_LDA"] = (1.0 + d['Predict_LDA'] * d["FACT"]) / 2.0
print(d)
hit_rate2 = np.mean(d["Correct_LDA"])
print("Процент верных предсказаний: %.1f%%" % (hit_rate2 * 100))

# Qadrical Discriminant
model3 = QDA()
model3.fit(x_train, y_train)  # Обучение (подбор параметров модели)
d['Predict_QDA'] = model3.predict(x_test)  # Тест

# Считаем процент правильно предсказанных направлений изменения цены:
d["Correct_QDA"] = (1.0 + d['Predict_QDA'] * d["FACT"]) / 2.0
print(d)
hit_rate3 = np.mean(d["Correct_QDA"])
print("Процент верных предсказаний: %.1f%%" % (hit_rate3 * 100))

# Take AVG from result's
rates = [hit_rate1,hit_rate2,hit_rate3]
if(np.average(rates) > 0.5): change_sum = 1
else: change_sum = -1


#
# Procent's
#
data_for_model = data_for_model.drop(columns=['SUM'])
data_for_model['SUM_prc'] = round(data_for_model['SUM_prc'])+1
x = data_for_model
x = x.fillna(0)
print(x)
y = prc.values.ravel()
y = y.astype('int')
print(y)

x_train = x[1:COUNT - 2]
x_test = x[COUNT -1:]
y_train = y[1:COUNT - 2]
y_test = y[-1]

d = pd.DataFrame(index=x_test.index)  # Set for check model
d['FACT'] = prc[1:]  # Real values

# Visual
print('##########')
print('##########')

# Логистическая регрессия
model1 = LogisticRegression()
model1.fit(x_train, y_train)  # Обучение (подбор параметров модели)

d['Predict_LR'] = model1.predict(x_test)  # Тест
# Считаем процент правильно предсказанных направлений изменения цены:
d["Correct_LR"] = (1.0 + d['Predict_LR'] * d['FACT']) / 2.0
print(d)
hit_rate1 = np.mean(d["Correct_LR"])
print("Процент верных предсказаний: %.1f%%" % (hit_rate1 * 100))

# Линейный дискриминант
model2 = LDA()
model2.fit(x_train, y_train)  # Обучение (подбор параметров модели)
d['Predict_LDA'] = model2.predict(x_test)  # Тест
# Считаем процент правильно предсказанных направлений изменения цены:
d["Correct_LDA"] = (1.0 + d['Predict_LDA'] * d["FACT"]) / 2.0
print(d)
hit_rate2 = np.mean(d["Correct_LDA"])
print("Процент верных предсказаний: %.1f%%" % (hit_rate2 * 100))

# Квадратичный дискриминант
#model3 = QDA()
#model3.fit(x_train, y_train)  # Обучение (подбор параметров модели)
#d['Predict_QDA'] = model3.predict(x_test)  # Тест
# Считаем процент правильно предсказанных направлений изменения цены:
#d["Correct_QDA"] = (1.0 + d['Predict_QDA'] * d["FACT"]) / 2.0
#print(d)
#hit_rate3 = np.mean(d["Correct_QDA"])
#print("Процент верных предсказаний: %.1f%%" % (hit_rate3 * 100))
# Take AVG from result's
rates = [hit_rate1,hit_rate2]#,hit_rate3]
max_scale = np.average(rates)





# Get All Variations of Numbers 6 of 36
loto_num_36 = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)
loto_values_36 = itertools.combinations(loto_num_36,6)

df_loto_values_36 = pd.DataFrame(list(loto_values_36), columns=['N1','N2','N3','N4','N5','N6'])
df_loto_values_36['SUM'] = df_loto_values_36['N1'] + df_loto_values_36['N2'] + df_loto_values_36['N3'] + df_loto_values_36['N4'] + df_loto_values_36['N5'] + df_loto_values_36['N6']
df_loto_values_36.set_index('SUM')
print('BASE COUNT OF VARIATIONS: ', len(df_loto_values_36))

# Filter Variations for change of sum for future
if (change_sum > 0):
    df_filter = df_loto_values_36[df_loto_values_36['SUM'] > data['SUM'].iloc[-2]]
    df_filter = df_filter[df_filter['SUM'] < data['SUM'].iloc[-2]*(1 + max_scale/100)]
else:
        df_filter = df_loto_values_36[df_loto_values_36['SUM'] < data['SUM'].iloc[-2]]
        df_filter = df_filter[df_filter['SUM'] > data['SUM'].iloc[-2] * (1 + max_scale / 100)]
print('COUNT OF VARIATIONS: ', len(df_filter))
print('### HEAD ###')
print(df_filter.head())
print('TAIL')
print(df_filter.tail())

# Filter one by one nubmers -- 1 2 3 4 ? ? /
df_filter = df_filter[ (df_filter['N1'] != (df_filter['N2']-1)) | (df_filter['N1'] != (df_filter['N3']-2)) | (df_filter['N1'] != (df_filter['N3']-3)) ]
df_filter = df_filter[ (df_filter['N2'] != (df_filter['N3']-1)) | (df_filter['N2'] != (df_filter['N4']-2)) | (df_filter['N2'] != (df_filter['N5']-3)) ]
df_filter = df_filter[ (df_filter['N3'] != (df_filter['N4']-1)) | (df_filter['N3'] != (df_filter['N5']-2)) | (df_filter['N3'] != (df_filter['N6']-3)) ]


print('AFTER CLEAR -- COUNT OF VARIATIONS: ', len(df_filter))
print('### HEAD ###')
print(df_filter.head())
print('TAIL')
print(df_filter.tail())

# Create table of chance for each number
chance_36 = [1,2,3,4,5,6,7,8,9,10
             ,11,12,13,14,15,16,17,18,19,20
             ,21,22,23,24,25,26,27,28,29,30
             ,31,32,33,34,35,36]

num = COUNT
# chance for namber, based on "cold numbers" strategy
for index, row in (data.iterrows()):
    cur_row = [row['N1'], row['N2'], row['N3'], row['N4'], row['N5'], row['N6']]
    for number in range(36):
        if( (number+1) in cur_row):
            chance_36[number] -= num / COUNT
        else:
            chance_36[number] += num / COUNT
    num -= 1

print(chance_36)

# TODO: set chance for chossen tickets

