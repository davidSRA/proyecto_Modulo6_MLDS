# Carga del dataset consolidado y preparado
df = pd.read_csv('dataset.csv')
df = df.drop(columns=['Volume'], errors='ignore')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()

print('Dimensiones del dataset:', df.shape)
display(df.head())

features_derived = ['RSI', 'Volatility_10', 'Range', 'LogVolume', 'Gap', 'Ratio']
features_price = ['Open', 'High', 'Low', 'Close']
features_mixed = features_price + features_derived

train = df.loc[:'2015-12-31']
val = df.loc['2016-01-01':'2021-12-31']
test = df.loc['2022-01-01':]

print('Tamaño de entrenamiento:', train.shape)
print('Tamaño de validación:', val.shape)
print('Tamaño de prueba:', test.shape)

