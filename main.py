import pandas as pd  # Пандас
import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler, MinMaxScaler #Нормировщики

# Get number in lottery
data = pd.read_csv('D:\\temp\\loto_stat.csv', delimiter=';', encoding='cp1251', header=0, index_col=0)
print(range(len(data)))
print('############')

COUNT = 1000

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
res = res[1:]

#print('RESULT COLUM')
#print(len(res))
#print(res)
#print('#################')

plt.plot(data['SUM'])
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

x = data_for_model
y = res.values.ravel()
y = y.astype('int')

x_train = x[1:COUNT-2]
x_test = x[COUNT-1:]
y_train = y[1:COUNT-2]
y_test = y[COUNT-1:]

#print('X TRAIN')
#print(x_train)
#print('X TEST')
#print(x_test)

#print('Y TRAIN')
#print(y_train)
#print('Y TEST')
#print(y_test)

d = pd.DataFrame(index=x_test.index) # Set for check model
d['FACT'] = res[1:] # Real values

# Visual
print('##########')
print('##########')


# Логистическая регрессия
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)     # Обучение (подбор параметров модели)

d['Predict_LR'] = model1.predict(x_test) # Тест
# Считаем процент правильно предсказанных направлений изменения цены:
d["Correct_LR"] = (1.0+d['Predict_LR']*d['FACT'])/2.0
print(d)
hit_rate1 = np.mean(d["Correct_LR"])
print("Процент верных предсказаний: %.1f%%" % (hit_rate1*100))



# Линейный дискриминант
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
model2 = LDA()
model2.fit(x_train, y_train)     # Обучение (подбор параметров модели)
d['Predict_LDA'] = model2.predict(x_test) # Тест

# Считаем процент правильно предсказанных направлений изменения цены:
d["Correct_LDA"] = (1.0+d['Predict_LDA']*d["FACT"])/2.0
print(d)
hit_rate2 = np.mean(d["Correct_LDA"])
print("Процент верных предсказаний: %.1f%%" % (hit_rate2*100))


# Квадратичный дискриминант
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
model3 = QDA()
model3.fit(x_train, y_train)     # Обучение (подбор параметров модели)
d['Predict_QDA'] = model3.predict(x_test) # Тест

# Считаем процент правильно предсказанных направлений изменения цены:
d["Correct_QDA"] = (1.0+d['Predict_QDA']*d["FACT"])/2.0
print(d)
hit_rate3 = np.mean(d["Correct_QDA"])
print("Процент верных предсказаний: %.1f%%" % (hit_rate3*100))