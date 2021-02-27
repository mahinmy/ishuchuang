import pandas as pd
from math import sin,cos,pi
import numpy as np
def read_one_month(year:int, month:int):
    data=pd.read_csv(f'data{year}/{year}-{month}月.csv',encoding='gbk')
    data['横向风速']=data[['汕尾风速','汕尾风向']].apply(lambda x:x['汕尾风速']*sin(pi/180*x['汕尾风向']),axis=1)
    data['纵向风速']=data[['汕尾风速','汕尾风向']].apply(lambda x:x['汕尾风速']*cos(pi/180*x['汕尾风向']),axis=1)
    return data

def read_all_data():
    data=pd.read_csv('data1993/1993-1月.csv',encoding='gbk')
    for month in range(2,13):
        file_name=f'data1993/1993-{month}月.csv'
        data=pd.concat([data,pd.read_csv(file_name,encoding='gbk')],ignore_index=True)
    for month in range(1,13):
        file_name=f'data1994/1994-{month}月.csv'
        data=pd.concat([data,pd.read_csv(file_name,encoding='gbk')],ignore_index=True)
    for month in range(1,13):
        file_name=f'data1995/1995-{month}月.csv'
        data=pd.concat([data,pd.read_csv(file_name,encoding='gbk')],ignore_index=True)
    from math import sin,cos,pi
    data['横向风速']=data[['汕尾风速','汕尾风向']].apply(lambda x:x['汕尾风速']*sin(pi/180*x['汕尾风向']),axis=1)
    data['纵向风速']=data[['汕尾风速','汕尾风向']].apply(lambda x:x['汕尾风速']*cos(pi/180*x['汕尾风向']),axis=1)
    return data
def read_one_year(year):
    data=pd.read_csv(f'data{year}/{year}-1月.csv',encoding='gbk')
    for month in range(2,13):
        file_name=f'data{year}/{year}-{month}月.csv'
        data=pd.concat([data, pd.read_csv(file_name,encoding='gbk')],ignore_index=True)
    data['横向风速']=data[['汕尾风速','汕尾风向']].apply(lambda x:x['汕尾风速']*sin(pi/180*x['汕尾风向']),axis=1)
    data['纵向风速']=data[['汕尾风速','汕尾风向']].apply(lambda x:x['汕尾风速']*cos(pi/180*x['汕尾风向']),axis=1)
    return data

def random_search(train_X,train_y,valid_X,valid_y,output_shape=1,cv=3):
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    def build_model(hidden_layers=1,layer_size=30,learning_rate=1e-3):
        model=Sequential()
        model.add(Dense(layer_size,activation='relu',input_shape=train_X.shape[1:]))
        for _ in range(hidden_layers-1):
            model.add(Dense(layer_size,activation='relu'))
        model.add(LSTM(layer_size,return_sequences=False))
        model.add(Dense(output_shape))
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mse',optimizer=optimizer)
        return model
    callbacks=[EarlyStopping(patience=5,min_delta=1e-3)]
    sklearn_model=keras.wrappers.scikit_learn.KerasRegressor(build_model)
    from scipy.stats import reciprocal
    param_distribution={
        'hidden_layers':[1,2,3],
        'layer_size':np.arange(10,100),
        'learning_rate':reciprocal(1e-4,1e-2),
    }
    from sklearn.model_selection import RandomizedSearchCV
    random_search_cv=RandomizedSearchCV(sklearn_model,param_distribution,n_iter=10,n_jobs=1,scoring='r2',cv=cv)
    random_search_cv.fit(train_X,train_y,epochs=30,callbacks=callbacks,validation_data=(valid_X,valid_y))
    return random_search_cv

def create_data(data,input_points=56,pred_points=8):
    X_data=[]
    y_data=[]
    for i in range(data.shape[0]-input_points-pred_points):
        X_data.append(data.iloc[i:i+input_points,[28,29,11]].values)
        y_data.append(data.iloc[i+input_points:i+input_points+pred_points,11].values)
    return np.array(X_data),np.array(y_data)

def svm_random_search_cv(train_X,train_y,cv=5):
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    def build_model(kernel='rbf',degree=3,C=1,epsilon=1e-2):
        model=SVR(kernel=kernel,degree=degree, C=C,epsilon=epsilon)
        model = MultiOutputRegressor(model)
        return model
    from scipy.stats import reciprocal
    param_distribution={
        'kernel':['rbf','poly','sigmoid','linear'],
        'degree':np.arange(2,5),
        'epsilon':reciprocal(1e-4,1e-2),
    }
    model=build_model()
    from sklearn.model_selection import RandomizedSearchCV
    random_search_cv=RandomizedSearchCV(model,param_distribution,n_iter=10,n_jobs=1,scoring='r2',cv=cv)
    random_search_cv.fit(train_X,train_y)
    return random_search_cv
