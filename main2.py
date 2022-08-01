# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:01:25 2022

@author: Jony
"""

import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns
from unicodedata import normalize
import yfinance as yf
import datetime as dt
import plotly.express as px


# Extraemos la lista de las 500 empresas desde wikipedia.
tabla_500SP = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

#brk-b en vez de brk.b
print(f'Total tables: {len(tabla_500SP)}')

dfsp = tabla_500SP[0]
symbols = dfsp.Symbol.str.replace('.','-')
sp = [*symbols.values]

print(sp)
sp500data = pd.read_csv('sp500data.csv', index_col='Date', parse_dates=['Date'], infer_datetime_format=True)

'''
sp500data = yf.download(sp, start= '2000-01-01', end='2021-12-31', group_by='ticker')
sp500data = sp500data.reset_index()
sp500data['Date'] = pd.to_datetime(sp500data['Date'])
sp500data = sp500data.set_index('Date')
sp500data = sp500data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
sp500data['Day'] = sp500data.index.day_name()
sp500data.rename(columns={'Ticker':'Symbol'}, inplace=True)
sp500data = sp500data.reset_index()
sp500data = sp500data.merge(tabla_500SP[0], on='Symbol', how='outer')
#sp500data = sp500data.set_index('Date')
sp500data.drop(['SEC filings', 'Headquarters Location', 'Date first added', 'CIK','Founded'], axis=1, inplace=True)
sp500data = sp500data.assign(Retorno_Gap= lambda x: (np.log(x['Open']/x['Close']).shift(1)).fillna(0))
sp500data = sp500data.assign(Retorno_Intradiario= lambda x: (np.log(x['Close']/x['Open'])).fillna(0))
sp500data = sp500data.assign(Variaciones= lambda x: x['Adj Close'].pct_change())
sp500data = sp500data.assign(Volatilidad= lambda x: (x['Variaciones'].rolling(250).std()*100*(250)**0.5))
sp500data.to_csv('sp500data.csv')
'''
syp = yf.download(['^GSPC'], start= '2000-01-01', end='2021-12-31')
syp = syp.assign(Retorno_Gap= lambda x: (np.log(x.Open/x.Close).shift(1)).fillna(0))
syp = syp.assign(Retorno_Intradiario= lambda x: (np.log(x['Close']/x['Open'])).fillna(0))
syp = syp.assign(Variaciones= lambda x: x['Adj Close'].pct_change().fillna(0))
syp = syp.assign(Volatilidad= lambda x: (x['Variaciones'].rolling(250).std()*100*(250)**0.5))
syp = syp[['Retorno_Intradiario','Variaciones']].apply(sum)
syp[1]
# Pregunta 1 Cual es el mejor día para invertir teniendo en cuenta el retorno de los movimiento gap (Jueves)
mejor_dia_gap = sp500data.groupby('Day')['Retorno_Gap'].sum().sort_values().index[-1]
print(f'El mejor día para invertir teniendo en cuenta el retorno de los movimientos gaps es {mejor_dia_gap}')
# Pregunta 2 Cual es el mejor día para invertir teniendo en cuenta el retorno de los movimientos intradiarios
mejor_dia_intra = sp500data.groupby('Day')['Retorno_Intradiario'].sum().sort_values.index[-1]
print(f'El mejor día para invertir teniendo en cuenta el retorno de los movimientos intradiarios es {mejor_dia_intra}')
# Pregunta 3
industrias = sp500data.groupby('GICS Sector')[['Retorno_Intradiario', 'Retorno_Gap', 'Variaciones']].sum()
mejores_industrias = industrias.sort_values(['Retorno_Intradiario', 'Variaciones'], ascending=[False,False])[:3]
industrias.plot(kind='bar', title='Industrias por retornos intradiarios, de gap y variaciones')
print(f'Las mejores industrias según los retornos intradiarios y las variaciones son {mejores_industrias.index.values[:3]}')

plt.xticks(rotation=45)
sns.distplot(sp500data['Retorno_Gap'])
px.histogram(industrias, x="Retorno_Intradiario")

Retorno_Gap_dia = sp500data.groupby('Day')['Retorno_Intradiario','Retorno_Gap', 'Variaciones'].sum()
Retorno_Gap_dia.reset_index()
import plotly.graph_objects as go
import plotly
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = go.Figure(data=[go.Candlestick(x=sp500data.index,
                open=sp500data['Open'],
                high=sp500data['High'],
                low=sp500data['Low'],
                close=sp500data['Close'])])
plotly.offline.plot({'data':[fig], 'layout': go.Layout(showlegend=True, height=600, width=600)})
fig.show()

spintra=syp['Retorno_Intradiario'].sum()
mejores = sp500data[(sp500data['GICS Sector'] == 'Health Care') | (sp500data['GICS Sector'] == 'Industrials') | (sp500data['GICS Sector'] == 'Information Technology')]
mejores_9 = mejores.groupby('Symbol')[['Retorno_Intradiario', 'Retorno_Gap', 'Variaciones', 'Volatilidad']].sum().sort_values(['Variaciones', 'Retorno_Intradiario', 'Volatilidad'], ascending=[False, False, True])[:9]

volatilidad = sp500data['Volatilidad'].apply(np.mean)
volatilidad.plot.line(x=volatilidad.index, y='Volatilidad',lw=1)
volatil_por_anio = volatilidad.groupby(volatilidad.index.map(lambda x: x.year)).sum()
peores_años = volatil_por_anio.sort_values(ascending=False).index.values[0:3]
a.index.values[0:3]


volatilidad = sp500data['Volatilidad'].apply(np.mean)
sns.set_style('darkgrid')
sns.set(rc = {'figure.figsize':(15,8)})
fig, ax = sns.lineplot(data=volatilidad).set(title='Volatilidad a través del tiempo')

ax.text('2012-1-1', volatil_por_anio[0], "New Year's Day", dict(size=10, color='gray'))
#%%
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['ticker return'][0:10],
    name = 'Ticker Total Return')
trace2 = go.Scatter(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['SP Return'][0:10],
    name = 'SP500 Total Return')
    
data = [trace1, trace2]
layout = go.Layout(title = 'Total Return vs S&P 500'
    , barmode = 'group'
    , yaxis=dict(title='Returns', tickformat=".2%")
    , xaxis=dict(title='Ticker')
    , legend=dict(x=.8,y=1)
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)

#%%
volatilidad = sp500data['Volatilidad'].apply(np.mean)
volatil_por_anio = volatilidad.groupby(volatilidad.index.map(lambda x: x.year)).sum()
peores_años = volatil_por_anio.sort_values(ascending=False).index.values[0:3]

mejores = sp500data[(sp500data['GICS Sector'] == 'Health Care') | (sp500data['GICS Sector'] == 'Industrials') | (sp500data['GICS Sector'] == 'Information Technology')]
mejores_9 = mejores.groupby('Symbol')[['Retorno_Intradiario', 'Retorno_Gap', 'Variaciones', 'Volatilidad']].sum().sort_values(['Variaciones', 'Retorno_Intradiario', 'Volatilidad'], ascending=[False, False, True])[:9]
