import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import sleep


# загружаем и подготавляваем данные
df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи


# создаем матрицу весов скрытого слоя
Win = np.zeros((1+inputSize,hiddenSizes)) 
# пороги w0 задаем случайными числами
Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes))) 
# остальные веса  задаем случайно -1, 0 или 1 
Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes))) 

#Wout = np.zeros((1+hiddenSizes,outputSize))

# случайно инициализируем веса выходного слоя
Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
   
# функция прямого прохода (предсказания) 
def predict(Xp):
    # выходы первого слоя = входные сигналы * веса первого слоя
    hidden_predict = np.where((np.dot(Xp, Win[1:,:]) + Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
    # выходы второго слоя = выходы первого слоя * веса второго слоя
    out = np.where((np.dot(hidden_predict, Wout[1:,:]) + Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
    return out, hidden_predict


# обучение
# у перцептрона Розенблатта обучаются только веса выходного слоя 
# как и раньше обучаем подавая по одному примеру и корректируем веса в случае ошибки
eta = 0.001
errors = 1
prev_wout = np.zeros((1+hiddenSizes,outputSize))
max_it = 2000
it = 0
list = []
list.append(prev_wout)

# #Вариант один
# while errors > 0 and not((prev_wout == Wout).all()):
#     errors = 0
#     for xi, target, j in zip(X, y, range(X.shape[0])):
#         pr, hidden = predict(xi)
#         prev_wout = np.copy(Wout)
#         if(pr != target):
#             Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
#             Wout[0] += eta * (target - pr)
#             errors += 1
#     print(errors)
#     print(not((prev_wout == Wout).all()))
#     #sleep(0.1)
    
# #Вариант два
# while errors > 0 and not((prev_wout == Wout).all()):
#     errors = 0
#     for xi, target, j in zip(X, y, range(X.shape[0])):
#         pr, hidden = predict(xi)
#         prev_wout = np.copy(Wout)
#         Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
#         Wout[0] += eta * (target - pr)
#     errors=sum(abs(pr-y.reshape(-1, 1))/2)
#     print(errors)
#     print(not((prev_wout == Wout).all()))
#     # sleep(0.1)

#Вариант три
def check(items,value):
    flag = True
    for item in items:
        if(((item == value).all())):
            flag = False
            break
    return flag

while errors > 0 and check(list,Wout) and it < max_it:
    errors = 0
    it += 1
    list.append(np.copy(Wout))
    for xi, target, j in zip(X, y, range(X.shape[0])):
        pr, hidden = predict(xi)
        #prev_wout = np.copy(Wout)
        if(pr != target):
            Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
            Wout[0] += eta * (target - pr)
            errors += 1
        
# array = np.array([1,0,0,0,0,0,0,1,1,1,1])
# array = array.reshape(-1, 1)
# print(check(list,array))
print(errors)
#print(not((prev_wout == Wout).all()))
    
# print(not((prev_wout == Wout).all()))

# посчитаем сколько ошибок делаем на всей выборке
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
pr, hidden = predict(X)
#sum(pr-y.reshape(-1, 1))
print(sum(abs(pr-y.reshape(-1, 1))/2))