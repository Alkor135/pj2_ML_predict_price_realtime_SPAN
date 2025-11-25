#!/usr/bin/env python3
"""
test_db_futures_options.py
Читает из БД все таблицы в датафреймы.
Выводит в консоль 5 первых и 5 последних строк каждой таблицы.
Дополнительно выводит размер файла SQLite.
"""

from pathlib import Path
from datetime import datetime, timedelta, date
import sqlite3
import pandas as pd

# ----------------------------- Настройки -----------------------------
TICKER = 'RTS'
DB_PATH = Path(r'c:\Users\Alkor\gd\data_quote_db')
DB_FILE = DB_PATH / f'{TICKER}_futures_options_day.db'

# ----------------------------- Основной код -----------------------------
def main():
    # Проверяем существование файла
    if not DB_FILE.exists():
        print(f"Файл базы данных не найден: {DB_FILE}")
        return

    # Получаем размер файла в байтах
    file_size_bytes = DB_FILE.stat().st_size

    # Конвертируем в КБ или МБ для удобства
    if file_size_bytes < 1024:
        file_size = f"{file_size_bytes} B"
    elif file_size_bytes < 1024**2:
        file_size = f"{file_size_bytes / 1024:.2f} KB"
    else:
        file_size = f"{file_size_bytes / (1024**2):.2f} MB"

    print(f"\nРазмер файла базы данных: {file_size} ({file_size_bytes} байт)")
    print(f"Путь: {DB_FILE}\n")

    # Подключение к БД
    conn = sqlite3.connect(DB_FILE)

    # Получаем список всех таблиц
    # query_tables = "SELECT name FROM sqlite_master WHERE type='table';"
    query_tables = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    tables_df = pd.read_sql_query(query_tables, conn)
    table_names = tables_df['name'].tolist()

    print(f"Найдено таблиц: {len(table_names)}\n")

    # Устанавливаем опции pandas для широкого вывода
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    # Читаем каждую таблицу и выводим первые и последние 5 строк
    for table_name in table_names:
        print(f"=== Таблица: {table_name} ===")
        df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)

        print("Первые 5 строк:")
        print(df.head())
        print("\nПоследние 5 строк:")
        print(df.tail())
        print("\n" + "-" * 50 + "\n")

    conn.close()

if __name__ == '__main__':
    main()