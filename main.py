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
#df_filter = df_filter[ (df_filter['N1'] != (df_filter['N2']-1)) | (df_filter['N1'] != (df_filter['N3']-2)) | (df_filter['N1'] != (df_filter['N3']-3)) ]
#df_filter = df_filter[ (df_filter['N2'] != (df_filter['N3']-1)) | (df_filter['N2'] != (df_filter['N4']-2)) | (df_filter['N2'] != (df_filter['N5']-3)) ]
#df_filter = df_filter[ (df_filter['N3'] != (df_filter['N4']-1)) | (df_filter['N3'] != (df_filter['N5']-2)) | (df_filter['N3'] != (df_filter['N6']-3)) ]
# Filter one by one numbers
df_filter = df_filter[ (df_filter['N1'] != (df_filter['N2']-1)) & (df_filter['N1'] != (df_filter['N3']-2)) & (df_filter['N1'] != (df_filter['N3']-3)) ]
df_filter = df_filter[ (df_filter['N2'] != (df_filter['N3']-1)) & (df_filter['N2'] != (df_filter['N4']-2)) & (df_filter['N2'] != (df_filter['N5']-3)) ]
df_filter = df_filter[ (df_filter['N3'] != (df_filter['N4']-1)) & (df_filter['N3'] != (df_filter['N5']-2)) & (df_filter['N3'] != (df_filter['N6']-3)) ]



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

# Нормируем вероятности к 100%
max_chance = max(chance_36)
print('max_chance: ', max_chance)

for i in range(36):
    chance_36[i] = round( chance_36[i]/max_chance * 100, 2)

print('CHANCE AFTER NORMALIZE')
print(chance_36)

map_of_chance = {1: chance_36[0], 2: chance_36[1], 3: chance_36[2], 4: chance_36[3], 5: chance_36[4], 6: chance_36[5], 7: chance_36[6], 8: chance_36[7], 9: chance_36[8], 10: chance_36[9],
                 11: chance_36[10], 12: chance_36[11], 13: chance_36[12], 14: chance_36[13], 15: chance_36[14], 16: chance_36[15], 17: chance_36[16], 18: chance_36[17], 19: chance_36[18], 20: chance_36[19],
                 21: chance_36[20], 22: chance_36[21], 23: chance_36[22], 24: chance_36[23], 25: chance_36[24], 26: chance_36[25], 27: chance_36[26], 28: chance_36[27], 29: chance_36[28], 30: chance_36[29],
                 31: chance_36[30], 32: chance_36[31], 33: chance_36[32], 34: chance_36[33], 35: chance_36[34], 36: chance_36[35]
                 }

df_filter['N1_chance'] = df_filter['N1'].map(map_of_chance)
df_filter['N2_chance'] = df_filter['N2'].map(map_of_chance)
df_filter['N3_chance'] = df_filter['N3'].map(map_of_chance)
df_filter['N4_chance'] = df_filter['N4'].map(map_of_chance)
df_filter['N5_chance'] = df_filter['N5'].map(map_of_chance)
df_filter['N6_chance'] = df_filter['N6'].map(map_of_chance)

df_filter['chance'] = df_filter['N1_chance'] + df_filter['N2_chance'] + df_filter['N3_chance'] + df_filter['N4_chance'] + df_filter['N5_chance']+ df_filter['N6_chance']
print(df_filter.head())

# Normalize chance
max_chance = max(df_filter['chance'])

df_filter['chance'] = df_filter['chance'] / max_chance * 100

print('After normalize')
print(df_filter.head())

df_filter = df_filter.sort_values(by=['chance'], ascending=False)
print('SORTERED')
print(df_filter.head())

print('TOP 5 BILETs')
print(df_filter['N1'].iloc[0], ' ', df_filter['N2'].iloc[0], ' ', df_filter['N3'].iloc[0], ' ', df_filter['N4'].iloc[0], ' ', df_filter['N5'].iloc[0], ' ', df_filter['N6'].iloc[0], ' ')
print(df_filter['N1'].iloc[1], ' ', df_filter['N2'].iloc[1], ' ', df_filter['N3'].iloc[1], ' ', df_filter['N4'].iloc[1], ' ', df_filter['N5'].iloc[1], ' ', df_filter['N6'].iloc[1], ' ')
print(df_filter['N1'].iloc[2], ' ', df_filter['N2'].iloc[2], ' ', df_filter['N3'].iloc[2], ' ', df_filter['N4'].iloc[2], ' ', df_filter['N5'].iloc[2], ' ', df_filter['N6'].iloc[2], ' ')
print(df_filter['N1'].iloc[3], ' ', df_filter['N2'].iloc[3], ' ', df_filter['N3'].iloc[3], ' ', df_filter['N4'].iloc[3], ' ', df_filter['N5'].iloc[3], ' ', df_filter['N6'].iloc[3], ' ')
print(df_filter['N1'].iloc[4], ' ', df_filter['N2'].iloc[4], ' ', df_filter['N3'].iloc[4], ' ', df_filter['N4'].iloc[4], ' ', df_filter['N5'].iloc[4], ' ', df_filter['N6'].iloc[4], ' ')

print('TILE 5 BILETs')
print(df_filter['N1'].iloc[-1], ' ', df_filter['N2'].iloc[-1], ' ', df_filter['N3'].iloc[-1], ' ', df_filter['N4'].iloc[-1], ' ', df_filter['N5'].iloc[-1], ' ', df_filter['N6'].iloc[-1], ' ')
print(df_filter['N1'].iloc[-2], ' ', df_filter['N2'].iloc[-2], ' ', df_filter['N3'].iloc[-2], ' ', df_filter['N4'].iloc[-2], ' ', df_filter['N5'].iloc[-2], ' ', df_filter['N6'].iloc[-2], ' ')
print(df_filter['N1'].iloc[-3], ' ', df_filter['N2'].iloc[-3], ' ', df_filter['N3'].iloc[-3], ' ', df_filter['N4'].iloc[-3], ' ', df_filter['N5'].iloc[-3], ' ', df_filter['N6'].iloc[-3], ' ')
print(df_filter['N1'].iloc[-4], ' ', df_filter['N2'].iloc[-4], ' ', df_filter['N3'].iloc[-4], ' ', df_filter['N4'].iloc[-4], ' ', df_filter['N5'].iloc[-4], ' ', df_filter['N6'].iloc[-4], ' ')
print(df_filter['N1'].iloc[-5], ' ', df_filter['N2'].iloc[-5], ' ', df_filter['N3'].iloc[-5], ' ', df_filter['N4'].iloc[-5], ' ', df_filter['N5'].iloc[-5], ' ', df_filter['N6'].iloc[-5], ' ')


