import pandas as pd
from pathlib import Path

columns_lst = ['date_time', 'label', 'predict_close', 'last', 'predict_high', 'high', 'predict_low', 'low']
file_path = Path(fr'../log_simulate_trade.txt')
# Загружаем файл с разделителем ';' в DF
df = pd.read_csv(file_path, delimiter=';', names=columns_lst)
df['previous_close'] = df['last'].shift(1)
df['rez'] = df['last'] - df['previous_close']
# print(df.info())
# print(df.describe().to_string(max_rows=8, max_cols=20))
print(df.nunique())

df = (
    df.query("label == 'CLOSE'")
)

df['rez_cum'] = df['rez'].cumsum()

print(df.to_string(max_rows=40, max_cols=22))
