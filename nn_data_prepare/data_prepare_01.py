"""

"""

from pathlib import Path
import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm  # <-- добавлено

# Название тикера (фьючерс)
tiker: str = 'RTS'

# Путь к базе данных с данными по фьючерсам и опционам
db_path: Path = Path(fr'c:\Users\Alkor\gd\data_quote_db\{tiker}_futures_options_day.db')

# Подключение к базе данных
conn = sqlite3.connect(db_path)

# Чтение данных из таблицы Futures в DataFrame df_f
df_f = pd.read_sql_query("SELECT `TRADEDATE`, `OPEN`, `LOW`, `HIGH`, `CLOSE`, `OPENPOSITION`, `SHORTNAME`, `LSTTRADE` FROM Futures", conn)
# Чтение данных из таблицы Options в DataFrame df_o
df_o = pd.read_sql_query("SELECT `TRADEDATE`, `OPENPOSITION`, `NAME`, `LSTTRADE`, `OPTIONTYPE`, `STRIKE` FROM Options", conn)

# Закрытие соединения с БД
conn.close()

# Преобразование столбцов TRADEDATE и LSTTRADE в формат datetime
df_f[['TRADEDATE', 'LSTTRADE']] = df_f[['TRADEDATE', 'LSTTRADE']].apply(pd.to_datetime)
df_o[['TRADEDATE', 'LSTTRADE']] = df_o[['TRADEDATE', 'LSTTRADE']].apply(pd.to_datetime)

# Удаление строк из df_o, где дата торгов (TRADEDATE) равна дате истечения опциона (LSTTRADE)
df_o = df_o[df_o.TRADEDATE < df_o.LSTTRADE]

# Сортировка DataFrame по дате торгов и сброс индекса
df_f = df_f.sort_values(by='TRADEDATE').reset_index(drop=True)
df_o = df_o.sort_values(by='TRADEDATE').reset_index(drop=True)

step_strike = 2500

# Создание пустого DataFrame для хранения результатов
df_rez = pd.DataFrame()

# === Добавлен tqdm к циклу для отображения прогресса ===
for row in tqdm(df_f.itertuples(), total=len(df_f), desc="Обработка строк фьючерсов", unit="строка"):
    # Выбор из DF с опционами только опционов на дату фьючерса
    df = df_o[df_o.TRADEDATE == row.TRADEDATE]

    # Проверка на пустоту
    if df.empty:
        continue

    # Формирование DataFrame с опционами put (P)
    df_p = (
        df.query('OPTIONTYPE ==  "P"')
        .groupby(['STRIKE'], as_index=False)
        .agg({'OPENPOSITION': 'sum'})
        .sort_values(['STRIKE'], ascending=True)
        .rename(columns={'OPENPOSITION': 'oi_p'})
    )

    # Формирование DataFrame с опционами call (C)
    df_c = (
        df.query('OPTIONTYPE ==  "C"')
        .groupby(['STRIKE'], as_index=False)
        .agg({'OPENPOSITION': 'sum'})
        .sort_values(['STRIKE'], ascending=True)
        .rename(columns={'OPENPOSITION': 'oi_c'})
    )

    # Генерация временного DataFrame со всеми возможными страйками
    df_tmp = pd.DataFrame(columns=['STRIKE'])
    for st in range(df.STRIKE.min(), df.STRIKE.max() + step_strike, step_strike):
        new_row = pd.DataFrame({'STRIKE': [st]})
        df_tmp = pd.concat([df_tmp, new_row], ignore_index=True)

    # Объединение DataFrame с опционами и временными страйками
    merged_df = pd.merge(df_p, df_c, on='STRIKE', how='outer')
    merged_df = pd.merge(merged_df, df_tmp, on='STRIKE', how='outer')
    merged_df = merged_df.infer_objects(copy=False)
    merged_df = merged_df.fillna(0)
    merged_df[['STRIKE', 'oi_c', 'oi_p']] = merged_df[['STRIKE', 'oi_c', 'oi_p']].astype(int)

    # Накопление суммы открытого интереса по call и put
    merged_df['oi_c'] = merged_df['oi_c'].cumsum()
    merged_df['oi_p'] = merged_df.iloc[::-1]['oi_p'].cumsum()[::-1]

    # Получение даты, цен и ближайшего страйка
    trade_date = (df_f.loc[df_f['TRADEDATE'] == row.TRADEDATE, 'TRADEDATE'].values[0]).astype('datetime64[D]')
    price_open = df_f.loc[df_f['TRADEDATE'] == row.TRADEDATE, 'OPEN'].values[0]
    price_close = df_f.loc[df_f['TRADEDATE'] == row.TRADEDATE, 'CLOSE'].values[0]
    price_high = df_f.loc[df_f['TRADEDATE'] == row.TRADEDATE, 'HIGH'].values[0]
    price_low = df_f.loc[df_f['TRADEDATE'] == row.TRADEDATE, 'LOW'].values[0]
    nearest_strike = round(price_close / step_strike) * step_strike

    # Получение подмножества строк вокруг ближайшего страйка
    index_lst = merged_df.index[merged_df['STRIKE'] == nearest_strike].tolist()
    if not index_lst:
        continue
    index_nearest = index_lst[0]
    start_index = max(0, index_nearest - 10)
    end_index = min(len(merged_df), index_nearest + 10 + 1)
    subset_df = merged_df.iloc[start_index:end_index]
    subset_df = subset_df.copy()

    # Расчёт разности открытого интереса между call и put
    subset_df['oi'] = subset_df.apply(
        lambda x: x.oi_p - x.oi_c if x.STRIKE < price_close else x.oi_c - x.oi_p, axis=1)

    # Нормализация значений
    scaler = MinMaxScaler()
    subset_df['oi_norm'] = scaler.fit_transform(subset_df[['oi']])

    # Добавление новых колонок
    subset_df['strike_abc'] = subset_df['STRIKE'] - nearest_strike
    subset_df = subset_df.set_index('strike_abc')
    subset_df = subset_df.sort_index(ascending=True)
    subset_df = subset_df[['oi_norm']].T
    subset_df.index = [trade_date]
    subset_df = subset_df.rename_axis('date')
    subset_df['high_abc'] = price_high - nearest_strike
    subset_df['low_abc'] = price_low - nearest_strike
    subset_df['close_abc'] = price_close - nearest_strike
    subset_df['central_strike'] = nearest_strike
    subset_df['open'] = price_open
    subset_df['high'] = price_high
    subset_df['low'] = price_low
    subset_df['close'] = price_close

    # Объединение результатов
    df_rez = pd.concat([df_rez, subset_df])

# Округление значений
df_rez.iloc[:, 0:21] = df_rez.iloc[:, 0:21].round(6)

# Добавление колонки с перцентилем
df_rez = df_rez.sort_values(by='date')
df_rez['body'] = (df_rez['close'] - df_rez['open'])

def rolling_percentile(series):
    result = [np.nan] * len(series)
    for i in range(20, len(series)):
        window = series.iloc[i-20:i]
        result[i] = (window < series.iloc[i]).sum() / 20
    return result

# df_rez['percentile_20'] = rolling_percentile(df_rez['ret'])

# Вывод
# print(df_rez.tail(15).to_string(max_rows=6, max_cols=20))
print(df_rez.to_string(max_rows=6, max_cols=20))
print("Количество колонок:", len(df_rez.columns))
print(df_rez.columns)
print("Форма DataFrame:", df_rez.shape)

# Сохранение
df_rez.to_pickle('features_and_target.pkl')
