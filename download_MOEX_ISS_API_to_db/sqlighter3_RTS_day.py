"""
Создание БД с таблицами Futures и Options при запуске скрипта.
При доступе из других модулей получает доступ к БД.
"""
from pathlib import Path
from typing import Any
import sqlite3

import pandas as pd


def create_tables(connection, cursor):
    """ Функция создания таблиц в БД если их нет"""
    try:
        with connection:
            cursor.execute('''CREATE TABLE if not exists Futures (
                            TRADEDATE         DATE PRIMARY KEY UNIQUE NOT NULL,
                            SECID             TEXT NOT NULL,
                            OPEN              REAL NOT NULL,
                            LOW               REAL NOT NULL,
                            HIGH              REAL NOT NULL,
                            CLOSE             REAL NOT NULL,
                            VOLUME            INTEGER NOT NULL,
                            OPENPOSITION      INTEGER NOT NULL,
                            SHORTNAME         TEXT NOT NULL,
                            LSTTRADE          DATE NOT NULL)'''
                           )
            cursor.execute('''CREATE TABLE if not exists Options (
                            ID                INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
                            TRADEDATE         DATE NOT NULL,
                            SECID             TEXT NOT NULL,
                            OPENPOSITION      INTEGER,
                            NAME              TEXT,
                            LSTTRADE          DATE,
                            OPTIONTYPE        TEXT,
                            STRIKE            INTEGER)'''
                           )
        print('Taблицы в БД созданы')
    except sqlite3.OperationalError as exception:
        print(f"Ошибка при создании БД: {exception}")


def non_empty_table_futures(connection, cursor):
    """Проверяем, есть ли записи в таблице 'Futures' в базе"""
    with connection:
        return cursor.execute("SELECT count(*) FROM (select 1 from Futures limit 1)").fetchall()[0][0]


def tradedate_futures_exists(connection, cursor, tradedate):
    """Проверяем, есть ли дата в таблице 'Futures' в базе"""
    with connection:
        result = cursor.execute('SELECT * FROM `Futures` WHERE `TRADEDATE` = ?', (tradedate,)).fetchall()
        return bool(len(result))


def tradedate_options_exists(connection, cursor, tradedate):
    """Проверяем, есть ли дата в таблице 'Options' в базе"""
    with connection:
        result = cursor.execute('SELECT * FROM `Options` WHERE `TRADEDATE` = ?', (tradedate,)).fetchall()
        return bool(len(result))


def add_tradedate_future(connection, cursor, tradedate, secid, open, low, high, close, volume, openposition, shortname,
                         lsttrade):
    """Добавляет строку в таблицу Futures """
    with connection:
        return cursor.execute(
            "INSERT INTO `Futures` (`TRADEDATE`, `SECID`, `OPEN`, `LOW`, `HIGH`, `CLOSE`, `VOLUME`, `OPENPOSITION`, "
            "`SHORTNAME`, `LSTTRADE`) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (tradedate, secid, open, low, high, close, volume, openposition, shortname, lsttrade))


def add_tradedate_option(connection, cursor, tradedate, secid, openposition, name, lsttrade, optiontype, strike):
    """Добавляет строку в таблицу Options """
    with connection:
        return cursor.execute(
            "INSERT INTO `Options` (`TRADEDATE`, `SECID`, `OPENPOSITION`, `NAME`, `LSTTRADE`, `OPTIONTYPE`, "
            "`STRIKE`) VALUES(?,?,?,?,?,?,?)",
            (tradedate, secid, openposition, name, lsttrade, optiontype, strike))


def get_tradedate_future(connection):  # Используется для перебора дат в
    """ Возвращает Dataframe с: дата торгов, короткое имя, последний день торгов из БД SQL по фьючерсам """
    with connection:
        return pd.read_sql('SELECT `TRADEDATE`, `SHORTNAME`, `LSTTRADE` FROM `Futures`', connection)


def get_tradedate_future_update(connection, start_date):
    """ Получение дат из обновленной таблицы Futures, для обновления таблицы Options """
    with connection:
        return pd.read_sql(f'SELECT TRADEDATE, SHORTNAME FROM Futures WHERE TRADEDATE >= "{start_date}"', connection)


def get_tradedate_future_date(connection, cursor, datedraw):
    """ Возвращает: дата торгов, low, high, close, короткое имя, последний день торгов из БД SQL на дату """
    with connection:
        return cursor.execute('SELECT LOW, HIGH, CLOSE, SHORTNAME, LSTTRADE '
                              'FROM Futures WHERE TRADEDATE = ?', (datedraw,)).fetchall()[0]


def get_df_datedraw(connection, datedraw):
    """ Возвращает выборку соответствующую дате построения графика """
    with connection:
        return pd.read_sql(f'SELECT * FROM Options '
                           f'WHERE TRADEDATE = "{datedraw}" AND TRADEDATE < LSTTRADE', connection)


def get_max_date_futures(connection, cursor):
    """ Получение максимальной даты по фьючерсам """
    with connection:
        return cursor.execute('SELECT MAX (TRADEDATE) FROM Futures').fetchall()[0][0]


def delete_options_bag(connection, cursor):
    """ Удаление опционов где дата торгов больше даты экспирации опционов """
    with connection:
        return cursor.execute('DELETE FROM Options WHERE TRADEDATE > LSTTRADE')


if __name__ == '__main__':  # Создание БД, если её не существует
    # Настройка базы данных
    tiker: str = 'RTS'
    path_bd: Path = Path(r'c:\Users\Alkor\gd\data_quote_db')  # Папка с БД
    file_bd: str = f'{tiker}_futures_options_day.db'

    if not path_bd.is_dir():  # Если не существует папка под БД
        try:
            path_bd.mkdir()  # Создание папки под БД
        except Exception as err:
            print(f'Ошибка создания каталога "{path_bd}": {err}')

    connection: Any = sqlite3.connect(fr'{path_bd}\{file_bd}', check_same_thread=True)
    cursor: Any = connection.cursor()
    create_tables(connection, cursor)  # Создание таблиц в БД если их нет
