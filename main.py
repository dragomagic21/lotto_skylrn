import pandas as pd  # Пандас
import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy
from sklearn.metrics import mean_squared_error
from math import sqrt
#from sklearn.preprocessing import StandardScaler, MinMaxScaler #Нормировщики

# Get number in lottery
data = pd.read_csv('/home/vadim/python_loto/loto_stat.csv', encoding='cp1251', header=0, index_col=0)
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
    if (data['SUM'].iloc[i] > data['SUM'].iloc[i-1]):
        res.loc[data['lottery_num'].iloc[i]] = 1
    else:
        res.loc[data['lottery_num'].iloc[i]] = -1
#res = []
#for i in range(1, len(data)):
#    if (data['SUM'].iloc[i] > data['SUM'].iloc[i - 1]):
#        res.append(1)
#    else:
#        res.append(-1)


print('RESULT COLUM')
print(len(res))
print(res)
print('#################')

plt.plot(res)
plt.savefig('demo.png', bbox_inches='tight')

# Make new Data_Model - get indexing from previous
data_for_model = pd.DataFrame(index=data.index)
data_for_model['SUM'] = data['SUM']
# New Column, wich append column with % of change sum from previous
data_for_model['SUM_prc'] = data['SUM'].pct_change()*100.0
# Replace 0 in SUM_prc with small values
for i,x in enumerate(data_for_model['SUM_prc']):
    if (abs(x) < 0.0001):
        data_for_model['SUM_prc'][i] = 0.0001
#res.reset_index(inplace=True)

x = data_for_model
y = res

x_train = x[1:98]
print(x_train)
x_test = x[99:]
#print(x_test)

y_train = y[1:98]
#print(y_train)
#print('##### y_TRAIN #######')
#print(y_train)
y_test = y[99:]
#print(y_test)

d = pd.DataFrame(index=data_for_model.index)
d['FACT'] = res[1:]
d.drop(d.index[0], inplace=True)
#print(d.head())

#print(y_train.values.ravel())

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)     # Обучение (подбор параметров модели)
#d['Predict_LR'] = model1.predict(x_test) # Тест

# Считаем процент правильно предсказанных направлений изменения цены:
#d["Correct_LR"] = (1.0+d['Predict_LR']*d['FACT'])/2.0
#print(d)
#hit_rate1 = np.mean(d["Correct_LR"])
#print("Процент верных предсказаний: %.1f%%" % (hit_rate1*100))