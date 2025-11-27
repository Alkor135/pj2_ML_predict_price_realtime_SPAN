#!/usr/bin/env python3
"""
main.py

Единый скрипт:
- Создаёт/подключается к БД с таблицами Futures и Options
- Догружает фьючерсы с MOEX (начиная с 2015-01-01, если БД пустая)
- Затем догружает опционы только для новых дат (по SHORTNAME фьючерса)
- Логирование: rts_update.log + консоль (цветной ANSI вывод)
- Повторы для сетевых запросов: до 3 попыток, задержка 10 секунд

Несколько строк отредактировать вручную в таблице Futures с даты 2017-12-21.
В колонке SHORTMANE заменить все RIH8 на RTS-3.18, чтобы корректно закачивались опционы.
"""

from pathlib import Path
from datetime import datetime, timedelta, date
import sqlite3
from typing import Any, Tuple, Optional

import time
import logging
import requests
import apimoex
import pandas as pd

# Преобразование Python → SQLite
sqlite3.register_adapter(date, lambda d: d.isoformat())
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat(" "))

# Преобразование SQLite TEXT → Python
sqlite3.register_converter("DATE", lambda s: date.fromisoformat(s.decode()))
sqlite3.register_converter("DATETIME", lambda s: datetime.fromisoformat(s.decode()))

# ----------------------------- Настройки -----------------------------
TICKER = 'RTS'
DB_PATH = Path(r'c:\Users\Alkor\gd\data_quote_db')
DB_FILE = DB_PATH / f'{TICKER}_futures_options_day.db'
LOG_FILE = Path('rts_update.log')

# Стартовая дата, если БД пустая (жёстко в коде)
START_DATE_IF_EMPTY = datetime.strptime('2019-01-01', "%Y-%m-%d").date()

# Повтор запросов к MOEX
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10

# ------------------------ ANSI цвета для консоли -----------------------
ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_BLUE = "\033[34m"
ANSI_YELLOW = "\033[33m"

def cprint_info(msg: str):
    print(f"{ANSI_BLUE}[INFO] {msg}{ANSI_RESET}")

def cprint_success(msg: str):
    print(f"{ANSI_GREEN}[OK] {msg}{ANSI_RESET}")

def cprint_warn(msg: str):
    print(f"{ANSI_YELLOW}[WARN] {msg}{ANSI_RESET}")

def cprint_error(msg: str):
    print(f"{ANSI_RED}[ERROR] {msg}{ANSI_RESET}")

# ---------------------------- Logging setup ---------------------------
logger = logging.getLogger("rts_updater")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

# Файловый Хэндлер
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Консольный хэндлер (без цветов в файле — цвета для print)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# ------------------------- DB: функции работы -------------------------
def ensure_db_dir():
    """Убедиться, что папка для БД существует"""
    if not DB_PATH.is_dir():
        try:
            DB_PATH.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создан каталог БД: {DB_PATH}")
            cprint_info(f"Создан каталог БД: {DB_PATH}")
        except Exception as e:
            logger.error(f"Не удалось создать каталог БД {DB_PATH}: {e}")
            cprint_error(f"Не удалось создать каталог БД {DB_PATH}: {e}")
            raise

def get_connection() -> Tuple[Any, Any]:
    """Открыть соединение и курсор SQLite"""
    conn = sqlite3.connect(
            str(DB_FILE),
            detect_types = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread = True
        )
    cur = conn.cursor()
    return conn, cur

def create_tables(connection, cursor):
    """Создать таблицы Futures и Options, если их нет"""
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
        logger.info("Таблицы в БД созданы/проверены")
        cprint_success("Таблицы в БД созданы/проверены")
    except sqlite3.OperationalError as exception:
        logger.error(f"Ошибка при создании таблиц: {exception}")
        cprint_error(f"Ошибка при создании таблиц: {exception}")
        raise

def non_empty_table_futures(connection, cursor) -> bool:
    """Проверить, есть ли записи в таблице Futures"""
    with connection:
        cnt = cursor.execute("SELECT count(*) FROM (select 1 from Futures limit 1)").fetchall()[0][0]
        return bool(cnt)

def tradedate_futures_exists(connection, cursor, tradedate: date) -> bool:
    """Проверить наличие даты в таблице Futures"""
    with connection:
        result = cursor.execute('SELECT 1 FROM Futures WHERE TRADEDATE = ?', (tradedate,)).fetchall()
        return bool(len(result))

def tradedate_options_exists(connection, cursor, tradedate: date) -> bool:
    """Проверить наличие даты в таблице Options"""
    with connection:
        result = cursor.execute('SELECT 1 FROM Options WHERE TRADEDATE = ?', (tradedate,)).fetchall()
        return bool(len(result))

def add_tradedate_future(connection, cursor, tradedate, secid, open_, low, high, close, volume, openposition, shortname, lsttrade):
    """Вставка записи во Futures"""
    with connection:
        cursor.execute(
            "INSERT OR IGNORE INTO Futures (TRADEDATE, SECID, OPEN, LOW, HIGH, CLOSE, VOLUME, OPENPOSITION, SHORTNAME, LSTTRADE) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (tradedate, secid, open_, low, high, close, volume, openposition, shortname, lsttrade)
        )

def add_tradedate_option(connection, cursor, tradedate, secid, openposition, name, lsttrade, optiontype, strike):
    """Вставка записи в Options"""
    with connection:
        cursor.execute(
            "INSERT INTO Options (TRADEDATE, SECID, OPENPOSITION, NAME, LSTTRADE, OPTIONTYPE, STRIKE) VALUES(?,?,?,?,?,?,?)",
            (tradedate, secid, openposition, name, lsttrade, optiontype, strike)
        )

def get_max_date_futures(connection, cursor) -> Optional[str]:
    """Вернуть максимальную дату TRADEDATE в таблице Futures (строка)"""
    with connection:
        res = cursor.execute('SELECT MAX(TRADEDATE) FROM Futures').fetchall()[0][0]
        return res

def get_shortname_for_date(connection, cursor, tradedate: date) -> Optional[str]:
    """Получить SHORTNAME для particular TRADEDATE из таблицы Futures"""
    with connection:
        res = cursor.execute('SELECT SHORTNAME FROM Futures WHERE TRADEDATE = ?', (tradedate,)).fetchall()
        if res:
            return res[0][0]
        return None

def get_tradedates_after(connection, cursor, from_date: date):
    """Получить список TRADEDATE и SHORTNAME из Futures где TRADEDATE > from_date"""
    with connection:
        df = pd.read_sql(f'SELECT TRADEDATE, SHORTNAME FROM Futures WHERE TRADEDATE > "{from_date}"', connection)
        # конвертируем TRADEDATE к типу date
        if not df.empty:
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE']).dt.date
        return df

# ----------------------- Вспом. функции для MOEX ----------------------
def with_retries(func, *args, **kwargs):
    """
    Враппер для повторных вызовов функций, которые могут падать при сетевых ошибках.
    Попытки: MAX_RETRIES, задержка: RETRY_DELAY_SECONDS
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            logger.warning(f"Попытка {attempt}/{MAX_RETRIES} не удалась: {e}")
            cprint_warn(f"Попытка {attempt}/{MAX_RETRIES} не удалась: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"Ждем {RETRY_DELAY_SECONDS} сек перед повтором...")
                time.sleep(RETRY_DELAY_SECONDS)
    # после всех попыток
    logger.error(f"Все {MAX_RETRIES} попыток завершились неудачей. Ошибка: {last_exc}")
    cprint_error(f"Все {MAX_RETRIES} попыток завершились неудачей. Ошибка: {last_exc}")
    raise last_exc

def get_info_future(session: requests.Session, security: str) -> Tuple[Optional[str], date]:
    """
    Запрашивает у MOEX информацию по фьючерсному инструменту:
    возвращает (SHORTNAME, LSTTRADE (date))
    Если LSTTRADE не найден — ставим '2130-01-01'
    """
    security_info = apimoex.find_security_description(session, security)
    df = pd.DataFrame(security_info)

    # Защита: если нет нужных строк
    name_lst = list(df['name'])
    if 'SHORTNAME' in name_lst:
        shortname = df.loc[df[df['name'] == 'SHORTNAME'].index]['value'].values[0]
    else:
        shortname = None

    if 'LSTTRADE' in name_lst:
        lsttrade = df.loc[df[df['name'] == 'LSTTRADE'].index]['value'].values[0]
    elif 'LSTDELDATE' in name_lst:
        lsttrade = df.loc[df[df['name'] == 'LSTDELDATE'].index]['value'].values[0]
    else:
        lsttrade = '2130-01-01'

    try:
        lsttrade_date = pd.to_datetime(lsttrade).date()
    except Exception:
        lsttrade_date = datetime.strptime('2130-01-01', '%Y-%m-%d').date()

    return shortname, lsttrade_date

def get_info_security(session: requests.Session, security: str) -> Tuple[Optional[str], date, Optional[str], Optional[int]]:
    """
    Информация по опциону:
    возвращает (NAME, LSTTRADE (date), OPTIONTYPE, STRIKE)
    Если LSTTRADE нет — LSTTRADE = 2130-01-01
    """
    security_info = apimoex.find_security_description(session, security)
    df = pd.DataFrame(security_info)
    name_lst = list(df['name'])
    if 'LSTTRADE' in name_lst:
        # Для NAME интересует последний токен (в коде оригинала)
        try:
            # получаем value для NAME и берем последний токен
            name_val = list(df.loc[df[df['name'] == 'NAME'].index]['value'])[0].split().pop()
        except Exception:
            name_val = None
        try:
            lsttrade = df.loc[df[df['name'] == 'LSTTRADE'].index]['value'].values[0]
            lsttrade_date = pd.to_datetime(lsttrade).date()
        except Exception:
            lsttrade_date = datetime.strptime('2130-01-01', '%Y-%m-%d').date()
        # OPTIONTYPE и STRIKE
        try:
            optiontype = df.loc[df[df['name'] == 'OPTIONTYPE'].index]['value'].values[0]
        except Exception:
            optiontype = None
        try:
            strike_val = df.loc[df[df['name'] == 'STRIKE'].index]['value'].values[0]
            strike = int(float(strike_val))
        except Exception:
            strike = None
        return name_val, lsttrade_date, optiontype, strike
    else:
        return (None, datetime.strptime('2130-01-01', '%Y-%m-%d').date(), None, None)

# ------------------------ Загрузка фьючерсов --------------------------
def get_future_date_results(tradedate: date, tiker: str, connection, cursor):
    """
    Основная логика для получения исторических данных по фьючерсам от MOEX.
    Для каждой даты (tradedate) делает запрос и записывает в DB.
    """
    today_date = datetime.now().date()

    arguments = {'securities.columns': ("BOARDID, TRADEDATE, SECID, OPEN, LOW, HIGH, CLOSE, OPENPOSITIONVALUE, VALUE, "
                                        "VOLUME, OPENPOSITION, SETTLEPRICE")}

    # Открываем сессию requests и используем apimoex.ISSClient
    with requests.Session() as session:
        # пока tradedate != today_date, итерируем по дням
        while tradedate != today_date:
            # если запись за tradedate уже есть — пропускаем
            if not tradedate_futures_exists(connection, cursor, tradedate):
                request_url = (f'http://iss.moex.com/iss/history/engines/futures/markets/forts/securities.json?'
                               f'date={tradedate.strftime("%Y-%m-%d")}&assetcode={tiker}')
                logger.info(f"Запрос фьючерсов: {request_url}")
                cprint_info(f"Запрос фьючерсов: {tradedate}")

                # выполняем запрос с retry-логикой вокруг apimoex.ISSClient(session, ...).get()
                def fetch():
                    iss = apimoex.ISSClient(session, request_url, arguments)
                    return iss.get()

                data = with_retries(fetch)

                df = pd.DataFrame(data.get('history', []))
                if len(df) != 0:
                    # Добавляем SHORTNAME и LSTTRADE через вызов get_info_future
                    df[['SHORTNAME', 'LSTTRADE']] = df.apply(lambda x: get_info_future(session, x['SECID']), axis=1, result_type='expand')
                    df["LSTTRADE"] = pd.to_datetime(df["LSTTRADE"]).dt.date
                    # Оставляем только те строки, где LSTTRADE > tradedate (исключаем опции для прошлых экспираций)
                    df = df.loc[df['LSTTRADE'] > tradedate]
                    # Убираем строки с пустыми ценами
                    df.dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'], inplace=True)
                    # Выбираем строки с минимальной LSTTRADE (наиболее ближайшая экспирация)
                    if not df.empty:
                        df = df[df['LSTTRADE'] == df['LSTTRADE'].min()].reset_index(drop=True)

                    logger.info(f"Получено записей фьючерсов на {tradedate}: {len(df)}")
                    cprint_info(f"Получено записей фьючерсов на {tradedate}: {len(df)}")

                    if len(df) == 1:
                        row = df.loc[0]
                        # защитные преобразования типов
                        try:
                            add_tradedate_future(
                                connection, cursor,
                                row['TRADEDATE'],
                                row['SECID'],
                                float(row['OPEN']),
                                float(row['LOW']),
                                float(row['HIGH']),
                                float(row['CLOSE']),
                                int(row['VOLUME']),
                                int(row['OPENPOSITION']),
                                row['SHORTNAME'],
                                row['LSTTRADE']
                            )
                            logger.info(f"Строка фьючерса записана: {row['TRADEDATE']} {row['SECID']}")
                            cprint_success(f"Фьючерс: {row['TRADEDATE']} записан")
                        except Exception as e:
                            logger.error(f"Ошибка записи фьючерса в БД для {tradedate}: {e}")
                            cprint_error(f"Ошибка записи фьючерса в БД для {tradedate}: {e}")
                    else:
                        # Если вдруг больше одной строки — пробуем записать по каждой (защищаемся от дубликатов)
                        for idx, row in df.iterrows():
                            try:
                                add_tradedate_future(
                                    connection, cursor,
                                    row['TRADEDATE'],
                                    row['SECID'],
                                    float(row['OPEN']),
                                    float(row['LOW']),
                                    float(row['HIGH']),
                                    float(row['CLOSE']),
                                    int(row['VOLUME']),
                                    int(row['OPENPOSITION']),
                                    row['SHORTNAME'],
                                    row['LSTTRADE']
                                )
                                logger.info(f"Фьючерс записан: {row['TRADEDATE']} {row['SECID']}")
                            except Exception as e:
                                logger.error(f"Ошибка записи ряда фьючерса: {e}")
                                cprint_error(f"Ошибка записи ряда фьючерса: {e}")
            else:
                logger.info(f"Фьючерс за дату {tradedate} уже есть в БД — пропускаем")
            # увеличиваем tradedate на 1 день
            tradedate += timedelta(days=1)

# ------------------------ Загрузка опционов ---------------------------
def get_options_date_results(tradedate: date, shortname: str):
    """
    Получаем все опционы на дату tradedate из MOEX (пагинация).
    Возвращаем DataFrame c полями TRADEDATE, SECID, OPENPOSITION, NAME, LSTTRADE, OPTIONTYPE, STRIKE
    """
    df_rez = pd.DataFrame()
    arguments = {'securities.columns': (
        "BOARDID, TRADEDATE, SECID, OPEN, LOW, HIGH, CLOSE, OPENPOSITIONVALUE, VALUE, VOLUME, OPENPOSITION, SETTLEPRICE"
    )}

    with requests.Session() as session:
        page = 0
        while True:
            request_url = (f'http://iss.moex.com/iss/history/engines/futures/markets/options/securities.json?'
                           f'date={tradedate}&assetcode={TICKER}&start={page}')
            logger.info(f"Запрос опционов: {request_url}")
            cprint_info(f"Запрос опционов для {tradedate}, страница start={page}")

            def fetch():
                iss = apimoex.ISSClient(session, request_url, arguments)
                return iss.get()

            data = with_retries(fetch)
            df = pd.DataFrame(data.get('history', []))
            if len(df) == 0:
                break
            # Оставляем нужные поля
            df = df[["TRADEDATE", "SECID", "OPENPOSITION"]]
            # Добавляем описание опционов (NAME, LSTTRADE, OPTIONTYPE, STRIKE)
            df[['NAME', 'LSTTRADE', 'OPTIONTYPE', 'STRIKE']] = df.apply(
                lambda x: get_info_security(session, x['SECID']), axis=1, result_type='expand'
            )
            df["LSTTRADE"] = pd.to_datetime(df["LSTTRADE"]).dt.date
            # Оставляем только опционы где LSTTRADE > tradedate (то есть еще "живые")
            df = df.loc[df['LSTTRADE'] > tradedate]
            # Оставляем только те, что относятся к shortname (фьючерсу)
            df = df.loc[df['NAME'] == shortname]
            # Заполняем NAN в OPENPOSITION нулями
            df['OPENPOSITION'] = df['OPENPOSITION'].fillna(0.0)
            df_rez = pd.concat([df_rez, df.dropna()]).reset_index(drop=True)
            logger.info(f"Накоплено опционов: {len(df_rez)} (страница start={page})")
            page += 100

    return df_rez

def add_row_options_table(connection, cursor, df: pd.DataFrame):
    """
    Преобразует OPENPOSITION в int и записывает DF построчно в таблицу Options
    """
    if df.empty:
        logger.info("Нет опционов для записи")
        cprint_info("Нет опционов для записи")
        return
    df = df.copy()
    df['OPENPOSITION'] = df['OPENPOSITION'].fillna(0.0)
    try:
        df['OPENPOSITION'] = df['OPENPOSITION'].astype(int)
    except Exception:
        # на всякий случай — округление перед конвертацией
        df['OPENPOSITION'] = df['OPENPOSITION'].round(0).astype(int)

    # Запись построчно
    for row in df.itertuples(index=False):
        try:
            add_tradedate_option(
                connection,
                cursor,
                row.TRADEDATE,
                row.SECID,
                int(row.OPENPOSITION),
                row.NAME,
                row.LSTTRADE,
                row.OPTIONTYPE,
                row.STRIKE
            )
        except Exception as e:
            logger.error(f"Ошибка записи опциона {row.SECID} на {row.TRADEDATE}: {e}")
            cprint_error(f"Ошибка записи опциона {row.SECID} на {row.TRADEDATE}: {e}")
    logger.info(f"Опционы за дату записаны в БД. Всего записано: {len(df)}")
    cprint_success(f"Опционы за дату записаны в БД. Всего записано: {len(df)}")

def get_futures_without_options(connection):
    """
    Возвращает DF с датами фьючерсов, для которых ещё нет записей в Options.
    """
    query = """
        SELECT f.TRADEDATE, f.SHORTNAME
        FROM Futures f
        LEFT JOIN Options o ON f.TRADEDATE = o.TRADEDATE
        WHERE o.TRADEDATE IS NULL
        ORDER BY f.TRADEDATE
    """
    return pd.read_sql(query, connection, parse_dates=['TRADEDATE'])

# --------------------------- Основной процесс ------------------------
def main():
    cprint_info("Старт RTS updater")
    logger.info("=== START RTS UPDATER ===")

    ensure_db_dir()
    connection, cursor = get_connection()
    create_tables(connection, cursor)

    # Определяем стартовую дату
    if non_empty_table_futures(connection, cursor):
        # Если таблица не пуста — берём максимальную дату
        prev_max = get_max_date_futures(connection, cursor)
        if prev_max:
            start_date = datetime.strptime(prev_max, "%Y-%m-%d").date()
            logger.info(f"Таблица Futures не пустая. Начинаем с последней даты в БД: {start_date}")
            cprint_info(f"Начинаем с последней даты в БД: {start_date}")
        else:
            start_date = START_DATE_IF_EMPTY
            logger.info(f"Не удалось определить max date, используем стартовую дату: {start_date}")
    else:
        start_date = START_DATE_IF_EMPTY
        logger.info(f"Таблица Futures пуста. Используем стартовую дату: {start_date}")
        cprint_info(f"Таблица Futures пуста. Используем стартовую дату: {start_date}")

    # Сохраним prev_max для последующей фильтрации дат для опционов
    prev_max_before_run = get_max_date_futures(connection, cursor)
    if prev_max_before_run:
        prev_max_before_run_date = datetime.strptime(prev_max_before_run, "%Y-%m-%d").date()
    else:
        prev_max_before_run_date = START_DATE_IF_EMPTY - timedelta(days=1)  # чтобы взять всё от START_DATE_IF_EMPTY
    logger.info(f"prev_max_before_run_date = {prev_max_before_run_date}")

    # 1) Догружаем фьючерсы
    try:
        get_future_date_results(start_date, TICKER, connection, cursor)
    except Exception as e:
        logger.error(f"Ошибка при обновлении фьючерсов: {e}")
        cprint_error(f"Ошибка при обновлении фьючерсов: {e}")
        # В случае фатальной ошибки прервём выполнение (чтобы не писать опции без фьючерсов)
        return

    # 2) После обновления фьючерсов получаем DF с датами фьючерсов, для которых ещё нет записей в Options
    new_dates_df = get_futures_without_options(connection)

    if new_dates_df.empty:
        logger.info("Новых дат фьючерсов не найдено — нет необходимости обновлять опционы")
        cprint_info("Новых дат фьючерсов не найдено — нет необходимости обновлять опционы")
    else:
        logger.info(f"Найдено новых дат фьючерсов: {len(new_dates_df)}")
        cprint_info(f"Найдено новых дат фьючерсов: {len(new_dates_df)}")
        # Проходим по найденным датам и догружаем опционы, если их нет в таблице Options
        for idx, row in new_dates_df.iterrows():
            tradedate = row['TRADEDATE']
            # tradedate = tradedate.strftime('%Y-%m-%d')
            tradedate = tradedate.to_pydatetime().date()
            shortname = row['SHORTNAME']
            # tradedate уже в типе date согласно get_tradedates_after
            # проверяем, есть ли опции на эту дату в БД
            # if tradedate_options_exists(connection, cursor, tradedate):
            #     logger.info(f"Опционы за дату {tradedate} уже есть в БД — пропускаем")
            #     continue
            logger.info(f"Обновляем опционы для {tradedate} (SHORTNAME={shortname})")
            cprint_info(f"Обновляем опционы для {tradedate} (SHORTNAME={shortname})")
            try:
                df_opts = get_options_date_results(tradedate, shortname)
                add_row_options_table(connection, cursor, df_opts)
            except Exception as e:
                logger.error(f"Ошибка при обработке опционов за {tradedate}: {e}")
                cprint_error(f"Ошибка при обработке опционов за {tradedate}: {e}")

    # Закрываем соединение
    try:
        connection.commit()
        logger.info("Выполняется VACUUM для оптимизации БД...")
        cprint_info("Выполняется VACUUM для оптимизации БД...")
        cursor.execute("VACUUM")
        logger.info("VACUUM завершён успешно.")
        cprint_success("VACUUM завершён успешно.")
        connection.close()
        logger.info("Соединение с БД закрыто")
        cprint_success("Готово. Соединение с БД закрыто")
    except Exception as e:
        logger.error(f"Ошибка при закрытии БД или выполнении VACUUM: {e}")
        cprint_error(f"Ошибка при закрытии БД или выполнении VACUUM: {e}")

    logger.info("=== END RTS UPDATER ===")

if __name__ == '__main__':
    main()
