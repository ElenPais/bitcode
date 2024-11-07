import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import joblib


def train_model_function(source, df_to_model):

    df_to_model["Date"] = pd.to_datetime(df_to_model["Date"])

    df_to_model["Date_Unix"] = df_to_model["Date"].apply(lambda x: int(x.timestamp()))

    df_to_model = df_to_model[df_to_model["Date"] < "2024-05-01"]

    df_to_model.drop(columns=["Date"], inplace=True)

    df_to_model_source = df_to_model[df_to_model["Source"]==source]


    X = df_to_model_source[['Open', 'Volume', 'Date_Unix']]
    y = df_to_model_source[['Adj Close']]

        
    X_to_train = X[['Date_Unix', 'Open', 'Volume']]

    
    X_train, X_test, y_train, y_test = train_test_split(X_to_train, y, train_size=0.3)

    
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train, y_train)

    
    joblib.dump(rf_regressor, f'{source}_modelo_ml.pkl')  # Salva o modelo em um arquivo .pkl

train_model_function(source="bitcoin", df_to_model=pd.read_csv("df_to_model.csv"))

train_model_function(source="apple", df_to_model=pd.read_csv("df_to_model.csv"))

train_model_function(source="amazon", df_to_model=pd.read_csv("df_to_model.csv"))
