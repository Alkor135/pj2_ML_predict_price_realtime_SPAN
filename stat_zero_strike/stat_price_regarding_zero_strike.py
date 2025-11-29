"""
Скрипт загружает финансовые данные из features_and_target.pkl.
Оставляет только ключевые столбцы: дату, close_abc и тело свечи.
Преобразует close_abc в бинарный признак (0 — ниже нуля, 1 — выше или равно).
Определяет направление движения цены (up/down) по знаку тела свечи.
Формирует целевую переменную next_direction со сдвигом на следующий период.
Балансирует выборку по классам up/down, отбирая равное количество последних данных.
Выводит статистику распределения close_abc для каждого направления.
"""
from pathlib import Path
import pandas as pd

pkl_path = Path(__file__).parent.parent / 'nn_data_prepare/features_and_target.pkl'

# Чтение данных из pkl файла
df = pd.read_pickle(pkl_path)

df = df.reset_index()
df = df[['date', 'close_abc', 'body']]

# Преобразование колонки 'date' в datetime и сортировка по ней
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Замена значений в 'close_abc': 0 если < 0, и 1 если >= 0
df['close_abc'] = (df['close_abc'] >= 0).astype(int)

# Создание колонки 'direction' на основе значения 'body'
df['direction'] = df['body'].apply(lambda x: 'up' if x > 0 else 'down')

# Создание колонки 'next_direction' — сдвиг значения 'direction' на следующий ряд
df['next_direction'] = df['direction'].shift(-1)

# Оставляем только нужные колонки
df = df[['date', 'close_abc', 'next_direction']]

# Удаляем строки с NaN в 'next_direction' (последняя строка)
df = df.dropna(subset=['next_direction']).reset_index(drop=True)

# Сортируем по дате по убыванию, чтобы оставить последние значения
df = df.sort_values('date', ascending=False)

# Разделяем на группы 'up' и 'down'
up_df = df[df['next_direction'] == 'up']
down_df = df[df['next_direction'] == 'down']

# Определяем минимальное количество между классами
n_samples = min(len(up_df), len(down_df))

# Берем по n_samples последних (т.е. самых свежих) строк из каждого класса
up_balanced = up_df.head(n_samples)
down_balanced = down_df.head(n_samples)

# Объединяем и сортируем обратно по возрастанию даты (если нужно хронологически)
df = pd.concat([up_balanced, down_balanced]).sort_values('date').reset_index(drop=True)

# === Статистика по vector_strike для 'up' и 'down' ===
print("\nСтатистика по значениям close_abc:")
print("\nРаспределение close_abc для next_direction = 'up':")
up_vectors = df[df['next_direction'] == 'up']['close_abc']
print(up_vectors.value_counts().to_string())

print("\nРаспределение close_abc для next_direction = 'down':")
down_vectors = df[df['next_direction'] == 'down']['close_abc']
print(down_vectors.value_counts().to_string())

# Общая сводка
print(f"\nВсего записей: {len(df)}")
print(f"up: {len(up_vectors)}, down: {len(down_vectors)}")

# Вывод в консоль
print("\nИтоговый DataFrame:")
print(df)