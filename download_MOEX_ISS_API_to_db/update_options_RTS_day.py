"""
Получение исторических данных по опционам RTS с MOEX ISS API
После формирования базы удалить строк из БД командой:
DELETE FROM Options WHERE TRADEDATE > LSTTRADE
"""
# import sys
from pathlib import Path
import requests
from datetime import date, datetime
from typing import Any

import apimoex
import pandas as pd
import sqlite3

import sqlighter3_RTS_day


def get_info_security(session, security: str):
    """
    Запрашивает у MOEX информацию по инструменту
    :param session: Подключение к MOEX
    :param security: Тикер инструмента
    :return: Дата последних торгов бара (если её нет то возвращает 2130.01.01), страйк, тип опциона, код фьючерса
    """
    security_info = apimoex.find_security_description(session, security)
    df = pd.DataFrame(security_info)  # Полученную информацию по тикеру в DF
    name_lst = list(df['name'])
    if 'LSTTRADE' in name_lst:
        row_lst = ['NAME', 'LSTTRADE', 'OPTIONTYPE', 'STRIKE']  # Список необходимых строк из DF
        # Меняем значение в ячейке ['NAME', 'value'] на конец строки 'value', где прописан код фьючерса
        df.loc[df[df['name'] == 'NAME'].index, 'value'] = list(df.loc[df[df['name'] == 'NAME'].index]['value'])[
            0].split().pop()
        df = df[df['name'].isin(row_lst)]  # Выборка необходимых строк
        rez_lst: list = list(df.value)  # Колонку 'value' в список
        return pd.Series(rez_lst)
    else:
        return pd.Series([float('nan'), '2130-01-01', float('nan'), float('nan')])


def get_options_date_results(tradedate: date, shortname: str):
    df_rez = pd.DataFrame()
    arguments = {'securities.columns': (
        "BOARDID, TRADEDATE, SECID, OPEN, LOW, HIGH, CLOSE, OPENPOSITIONVALUE, VALUE, VOLUME, OPENPOSITION, SETTLEPRICE"
    )}
    # arguments = {'securities.columns': ("TRADEDATE, SECID, OPENPOSITION")}

    with requests.Session() as session:
        page = 0  # С какой записи стартовать запрос
        while True:  # В цикле отправляем запрос постранично и обрабатываем ответ
            request_url = (f'http://iss.moex.com/iss/history/engines/futures/markets/options/securities.json?'
                           f'date={tradedate}&assetcode=RTS&start={page}')
            print(f'{request_url=}')
            iss = apimoex.ISSClient(session, request_url, arguments)
            data = iss.get()
            df = pd.DataFrame(data['history'])  # Полученные исторические данные в DF
            # print(df.to_string(max_rows=20, max_cols=15), '\n')
            if len(df) == 0:  # Больше нет страниц в ответе
                break
            else:
                df = df[["TRADEDATE", "SECID", "OPENPOSITION"]]  # Оставляем нужные поля
                # df = df.drop(columns=['BOARDID', 'SETTLEPRICEDAY', 'WAPRICE'])  # Удаляем не нужные поля
                # Создаем новые колонки 'NAME', 'LSTTRADE', 'OPTIONTYPE', 'STRIKE' и заполняем
                df[['NAME', 'LSTTRADE', 'OPTIONTYPE', 'STRIKE']] = df.apply(
                    lambda x: get_info_security(session, x['SECID']), axis=1)
                # print(df.to_string(max_rows=20, max_cols=15), '\n')
                # Меняем формат колонки на дату
                df["LSTTRADE"] = pd.to_datetime(df["LSTTRADE"])
                # # Преобразуем указанные колонки в тип "дата"
                # df[["LSTTRADE", "TRADEDATE"]] = df[["LSTTRADE", "TRADEDATE"]].apply(pd.to_datetime)
                # Оставляем только строки, где дата экспирации опциона больше даты бара фьючерса(исключаем ОИ=0)
                df = df.loc[df['LSTTRADE'] > tradedate]
                df = df.loc[df['NAME'] == shortname]  # Выбор опционов текущего базового актива
                # Заполняем пропущенные значения в столбце OPENPOSITION значением 0.0
                df['OPENPOSITION'] = df['OPENPOSITION'].fillna(0.0)
                # df = df[df['LSTTRADE'] == df['LSTTRADE'].min()]  # Выборка строк с минимальной датой
                # df_rez = pd.concat([df_rez, df]).reset_index(drop=True)  # Слияние DF
                df_rez = pd.concat([df_rez.dropna(), df.dropna()]).reset_index(drop=True)
                print(df_rez.to_string(max_rows=6, max_cols=18), '\n')
                page += 100  # для запроса следующей страницы со 100 записями

    return df_rez


def add_row_options_table(connection, cursor, df):
    """
    Функция преобразует поле 'OPENPOSITION' в int и передает DF построчно для записи в БД
    """
    df['OPENPOSITION'] = df['OPENPOSITION'].fillna(0.0)  # Nan в 0.0
    df['OPENPOSITION'] = df['OPENPOSITION'].astype(int)  # float в int
    # print(df.to_string(max_rows=20, max_cols=25), '\n')
    # # Вызываем функцию sys.exit() для остановки выполнения кода
    # sys.exit()
    for row in df.itertuples():  # Перебираем опционы для занесения в БД
        # print(f'{row.TRADEDATE}, {row.SECID}, {int(row.OPENPOSITION)}, {row.NAME}, {row.LSTTRADE.date()}, '
        #       f'{row.OPTIONTYPE}, {row.STRIKE}')
        sqlighter3_RTS_day.add_tradedate_option(
            connection,
            cursor,
            row.TRADEDATE,
            row.SECID,
            int(row.OPENPOSITION),
            row.NAME,
            row.LSTTRADE.date(),
            row.OPTIONTYPE,
            row.STRIKE
        )
    print('Опционы за дату записаны в БД.')
    # Вызываем функцию sys.exit() для остановки выполнения кода
    # sys.exit()


if __name__ == '__main__':  # Точка входа при запуске этого скрипта
    tiker: str = 'RTS'
    path_db: Path = Path(fr'c:\Users\Alkor\gd\data_quote_db\{tiker}_futures_options_day.db')
    # Лучше брать последнюю дату в БД таблицы Options
    start_date: date = datetime.strptime('2024-07-01', "%Y-%m-%d").date()

    connection: Any = sqlite3.connect(path_db, check_same_thread=True)
    cursor: Any = connection.cursor()

    # Получаем в DF данные по фьючерсам из БД
    df_tradedate: pd = sqlighter3_RTS_day.get_tradedate_future_update(connection, start_date)
    df_tradedate.sort_values(by='TRADEDATE')
    # print(df_tradedate.to_string(max_rows=10, max_cols=20), '\n')
    # print(type(df_tradedate.TRADEDATE[0]))

    df = pd.DataFrame()
    for row in df_tradedate.itertuples():  # Перебираем даты для запроса торгуемых опционов на эту дату
        print(f'\nИндекс={row.Index}, TRADEDATE={row.TRADEDATE}, SHORTNAME={row.SHORTNAME}')
        # Нет записи с такой датой
        if not sqlighter3_RTS_day.tradedate_options_exists(connection, cursor, row.TRADEDATE):
            df = get_options_date_results(row.TRADEDATE, row.SHORTNAME)  # Получаем DF по опционам от МОЕХ
            # print(df.to_string(max_rows=10, max_cols=20), '\n')
            add_row_options_table(connection, cursor, df)  # Записываем в БД построчно DF по опционам
