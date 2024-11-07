import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import joblib
from dash import Dash, html, dcc, callback, Output, Input

df_bitcoin = pd.read_csv("BTC-USD.csv")
df_amazon = pd.read_csv("AMZN.csv")
df_apple = pd.read_csv("AAPL.csv")

df_bitcoin["Source"]="bitcoin"
df_amazon["Source"]="amazon"
df_apple["Source"]="apple"

df_consolidated=pd.concat([df_bitcoin, df_amazon, df_apple])

list_df=[df_bitcoin, df_amazon, df_apple]

df_growth_rate=pd.DataFrame(columns=['Source', 'Year', 'GrowthRate'])

for source_type in list(set(df_consolidated['Source'].to_list())):
    
    df=df_consolidated[df_consolidated["Source"]==source_type]
    df_aux = df[["Date", "Adj Close"]].copy()
    df_aux["Date"] = pd.to_datetime(df_aux["Date"])
    df_aux["Year"] = df_aux["Date"].dt.year

    columns = ["Year", "First Value", "Last Value"]
    df_aux_year = pd.DataFrame(columns=columns)

    list_years = list(set(list(df_aux["Year"])))

    del list_years[0]

    aux_df_growth_rate = pd.DataFrame(columns=['Year', 'GrowthRate'])

    for x in list_years:
        df_aux_filter = df_aux[df_aux["Year"]==x]
        # first_value = df_aux_filter.head(1)['Adj Close'].iloc[0]
        # last_value = df_aux_filter.tail(1)['Adj Close'].iloc[0]
        first_value = df_aux_filter.iloc[0, 1] # primeira linha do Adj Close, mais eficiente
        last_value = df_aux_filter.iloc[-1, 1] # ultima linha do Adj Close, mais eficiente
        growth_rate = (last_value - first_value)/first_value
        dict_= {
            "Year":[x],
            "GrowthRate":[growth_rate]
        }
        aux = pd.DataFrame(dict_)
        aux_df_growth_rate = pd.concat([aux_df_growth_rate,aux])

    aux_df_growth_rate['Source']=source_type
    df_growth_rate=pd.concat([df_growth_rate,aux_df_growth_rate])

fig_gw = go.Figure()

for source in df_growth_rate['Source'].unique():
    dados_filtrados = df_growth_rate[df_growth_rate['Source'] == source]
    x = dados_filtrados['Year']
    y = dados_filtrados['GrowthRate']

    fig_gw.add_trace(go.Scatter(x=x, y=y, name=source))

fig_gw.update_layout(title='Taxa de Crescimento ao Ano',
                  xaxis_title='Ano',
                  yaxis_title='Taxa de Crescimento')

fig_gw.update_layout(colorway=['blue', 'green', 'red'])

df_consolidated["Month"]=pd.to_datetime(df_consolidated["Date"]).dt.to_period("M").astype(str)
df_consolidated["Year"]=pd.to_datetime(df_consolidated["Date"]).dt.year

df_per_month = df_consolidated[df_consolidated["Source"]=="bitcoin"][["Month", "Year", "Volume", "Source"]]
aux_agg_month = df_per_month.groupby("Month").agg(volume_agg=("Volume", "sum")).reset_index()

df_per_month = aux_agg_month.merge(df_per_month, how="left", on="Month")
df_per_month.drop("Volume", axis=1, inplace=True)
df_per_month.drop_duplicates(inplace=True)
df_per_month["volume_agg"]=(df_per_month["volume_agg"]/10**9).round(2)

df_per_month["Month"] = df_per_month["Month"].apply(lambda x: x[5:])

fig_vol = go.Figure()
df_per_month.sort_values(by=["Month", "Year"], ascending = [True, True], inplace=True)
years = sorted(df_per_month['Year'].unique())
years.remove(2019)
years.remove(2024)


for year in years:
    dados_filtrados = df_per_month[df_per_month['Year'] == year]
    x = dados_filtrados['Month']
    y = dados_filtrados['volume_agg']

    fig_vol.add_trace(go.Scatter(x=x, y=y, name=str(year)))

fig_vol.update_layout(title='Volumetria por Mês',
                  xaxis_title='Mês',
                  yaxis_title='Volumetria (x10^9)')

fig_vol.update_layout(colorway=['#D3D6FF', '#A7ACFF', '#7B83FF', '#4F59FF'])


modelo_bitcoin = joblib.load('bitcoin_modelo_ml.pkl')

modelo_apple = joblib.load('apple_modelo_ml.pkl')

modelo_amazon = joblib.load('amazon_modelo_ml.pkl')


df_to_model = df_consolidated[['Date', 'Open', 'Source', 'High', 'Low', 'Adj Close', 'Volume']]

df_to_model["Date"] = pd.to_datetime(df_to_model["Date"])

df_to_model["Date_Unix"] = df_to_model["Date"].apply(lambda x: int(x.timestamp()))

df_to_predict = df_to_model[df_to_model["Date"]>="2024-05-01"]

df_to_fig = df_to_model[df_to_model["Date"]<"2024-05-01"]


X_bitcoin = df_to_predict[df_to_predict["Source"]=="bitcoin"][['Date_Unix', 'Open', 'Volume']]

X_apple = df_to_predict[df_to_predict["Source"]=="apple"][['Date_Unix', 'Open', 'Volume']]

X_amazon = df_to_predict[df_to_predict["Source"]=="amazon"][['Date_Unix', 'Open', 'Volume']]


previsoes_bitcoin = modelo_bitcoin.predict(X_bitcoin)
X_bitcoin["Predict"]=previsoes_bitcoin

previsoes_apple = modelo_apple.predict(X_apple)
X_apple["Predict"]=previsoes_apple

previsoes_amazon = modelo_amazon.predict(X_amazon)
X_amazon["Predict"]=previsoes_amazon


X_bitcoin['Date']=X_bitcoin["Date_Unix"].apply(lambda x:datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))

X_apple['Date']=X_apple["Date_Unix"].apply(lambda x:datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))

X_amazon['Date']=X_amazon["Date_Unix"].apply(lambda x:datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))

X_bitcoin['Tipo']="bitcoin"

X_apple['Tipo']="apple"

X_amazon['Tipo']="amazon"

X_Consolidated = pd.concat([X_bitcoin, X_apple, X_amazon])


#Criar um gráfico comparando a entrada com a saída
#plot

fig_bitcoin = go.Figure()

fig_bitcoin.add_trace(go.Scatter(x=df_to_fig[(df_to_fig['Source']=="bitcoin")&(df_to_fig['Date']>="2024-01-01")]['Date'], y=df_to_fig[(df_to_fig['Source']=="bitcoin")&(df_to_fig['Date']>="2024-01-01")]['Adj Close'], name="Valor_Historico"))

fig_bitcoin.add_trace(go.Scatter(x=X_Consolidated[X_Consolidated["Tipo"]=="bitcoin"]['Date'], y=X_Consolidated[X_Consolidated["Tipo"]=="bitcoin"]['Predict'], mode='lines', name="Valor_Previsto"))

fig_bitcoin.update_layout(title='Acurácia do Modelo',
                  xaxis_title='Date',
                  yaxis_title='Valores')

fig_bitcoin.update_layout(colorway=['blue', 'red'])

#fig_bitcoin.show()



fig_apple = go.Figure()

fig_apple.add_trace(go.Scatter(x=df_to_fig[df_to_fig['Source']=="apple"]['Date'], y=df_to_fig[df_to_fig['Source']=="apple"]['Adj Close'], name="Valor_Historico"))

fig_apple.add_trace(go.Scatter(x=X_apple['Date'], y=X_apple['Predict'], mode='lines', name="Valor_Previsto"))

fig_apple.update_layout(title='Acurácia do Modelo',
                  xaxis_title='Date',
                  yaxis_title='Valores')

fig_apple.update_layout(colorway=['blue', 'red'])

#fig_apple.show()



fig_amazon = go.Figure()

fig_amazon.add_trace(go.Scatter(x=df_to_fig[df_to_fig['Source']=="amazon"]['Date'], y=df_to_fig[df_to_fig['Source']=="amazon"]['Adj Close'], name="Valor_Historico"))

fig_amazon.add_trace(go.Scatter(x=X_amazon['Date'], y=X_amazon['Predict'], mode='lines', name="Valor_Previsto"))

fig_amazon.update_layout(title='Acurácia do Modelo',
                  xaxis_title='Date',
                  yaxis_title='Valores')

fig_amazon.update_layout(colorway=['blue', 'red'])

#fig_amazon.show()



## Criar Dash  (Se possivel criar um Grafico plotly sendo um so para todas as 3 empresas)

app = Dash()

app.layout = [
    html.H1(children='Stocks of Dash App', style={'textAlign':'center'}),
    dcc.Graph(id='graph-gw', figure=fig_gw),
    dcc.Dropdown(df_consolidated.Source.unique(), 'bitcoin', id='dropdown-selection'),
    dcc.Graph(id='graph-vol'),
    dcc.Graph(id='graph-predict')
]


#@callback(
#   Output('graph-vol', 'figure'),
#    Input('dropdown-selection', 'value')
#)


@callback(
    [Output('graph-vol', 'figure'),
     Output('graph-predict', 'figure')],
    [Input('dropdown-selection', 'value')]
)

def update_graph(value):
    #dff = df[df.country==value]
    #return px.line(dff, x='year', y='pop')

    df_per_month = df_consolidated[df_consolidated["Source"]==value][["Month", "Year", "Volume", "Source"]]
    aux_agg_month = df_per_month.groupby("Month").agg(volume_agg=("Volume", "sum")).reset_index()

    df_per_month = aux_agg_month.merge(df_per_month, how="left", on="Month")
    df_per_month.drop("Volume", axis=1, inplace=True)
    df_per_month.drop_duplicates(inplace=True)
    df_per_month["volume_agg"]=(df_per_month["volume_agg"]/10**9).round(2)

    df_per_month["Month"] = df_per_month["Month"].apply(lambda x: x[5:])

    fig_vol = go.Figure()
    df_per_month.sort_values(by=["Month", "Year"], ascending = [True, True], inplace=True)
    years = sorted(df_per_month['Year'].unique())
    years.remove(2019)
    years.remove(2024)


    for year in years:
        dados_filtrados = df_per_month[df_per_month['Year'] == year]
        x = dados_filtrados['Month']
        y = dados_filtrados['volume_agg']

        fig_vol.add_trace(go.Scatter(x=x, y=y, name=str(year)))

    fig_vol.update_layout(title='Volumetria por Mês',
                    xaxis_title='Mês',
                    yaxis_title='Volumetria (x10^9)')

    fig_vol.update_layout(colorway=['#D3D6FF', '#A7ACFF', '#7B83FF', '#4F59FF'])

    fig_predict = go.Figure()

    fig_predict.add_trace(go.Scatter(x=df_to_fig[(df_to_fig['Source']==value)&(df_to_fig['Date']>="2024-01-01")]['Date'], y=df_to_fig[(df_to_fig['Source']=="bitcoin")&(df_to_fig['Date']>="2024-01-01")]['Adj Close'], name="Valor_Historico"))

    fig_predict.add_trace(go.Scatter(x=X_Consolidated[X_Consolidated["Tipo"]==value]['Date'], y=X_Consolidated[X_Consolidated["Tipo"]=="bitcoin"]['Predict'], mode='lines', name="Valor_Previsto"))

    fig_predict.update_layout(title='Acurácia do Modelo',
                    xaxis_title='Date',
                    yaxis_title='Valores')

    fig_predict.update_layout(colorway=['blue', 'red'])

    return fig_vol, fig_predict


if __name__ == '__main__':
    app.run(debug=True)

