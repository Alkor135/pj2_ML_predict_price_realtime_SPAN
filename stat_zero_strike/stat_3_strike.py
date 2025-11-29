"""
Скрипт загружает финансовые данные из features_and_target.pkl.
Преобразует значения столбцов -2500, 0, 2500 в бинарные по порогу 0.000001.
Формирует признак vector_strike как комбинацию трёх бинарных флагов.
Определяет направление движения (up/down) по телу свечи и сдвигает его как целевую переменную.
Балансирует классы, оставляя равное количество последних записей каждого типа.
Выводит статистику распределения vector_strike по направлениям.
Строит и сохраняет столбчатую диаграмму сравнения распределений up и down.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля графиков
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

pkl_path = Path(__file__).parent.parent / 'nn_data_prepare/features_and_target.pkl'

# Чтение данных из pkl файла
df = pd.read_pickle(pkl_path)

df = df.reset_index()
df = df[['date', -2500, 0, 2500, 'open', 'close', 'body']]

# Преобразование колонки 'date' в datetime и сортировка по ней
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Замена значений в столбцах -2500, 0, 2500 на 1 или 0 в зависимости от порога
columns_to_update = [-2500, 0, 2500]
for col in columns_to_update:
    df[col] = (df[col] > 0.000001).astype(int)

# Создание новой колонки 'vector_strike' со списком значений из столбцов -2500, 0, 2500
df['vector_strike'] = df[[-2500, 0, 2500]].values.tolist()

# Создание колонки 'direction' на основе значения 'body'
df['direction'] = df['body'].apply(lambda x: 'up' if x > 0 else 'down')

# Создание колонки 'next_direction' — сдвиг значения 'direction' на следующий ряд
df['next_direction'] = df['direction'].shift(-1)

# Оставляем только нужные колонки
df = df[['date', 'vector_strike', 'next_direction']]

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
print("\nСтатистика по значениям vector_strike:")
print("\nРаспределение vector_strike для next_direction = 'up':")
up_vectors = df[df['next_direction'] == 'up']['vector_strike']
print(up_vectors.value_counts().to_string())

print("\nРаспределение vector_strike для next_direction = 'down':")
down_vectors = df[df['next_direction'] == 'down']['vector_strike']
print(down_vectors.value_counts().to_string())

# Общая сводка
print(f"\nВсего записей: {len(df)}")
print(f"up: {len(up_vectors)}, down: {len(down_vectors)}")

# === Подготовка данных для графика ===
up_vectors = df[df['next_direction'] == 'up']['vector_strike']
down_vectors = df[df['next_direction'] == 'down']['vector_strike']

# Преобразуем списки в кортежи, чтобы сделать их хешируемыми
up_counts = up_vectors.apply(tuple).value_counts().rename('up').reset_index()
up_counts.columns = ['vector', 'up']
down_counts = down_vectors.apply(tuple).value_counts().rename('down').reset_index()
down_counts.columns = ['vector', 'down']

# Объединение по вектору
plot_df = pd.merge(up_counts, down_counts, on='vector', how='outer').fillna(0)

# Преобразуем обратно в строковое представление для отображения
plot_df['vector'] = plot_df['vector'].astype(str)

# Преобразуем в длинный формат для seaborn
plot_df_long = plot_df.melt(id_vars='vector', value_vars=['up', 'down'],
                            var_name='direction', value_name='count')

# === Построение столбчатой диаграммы ===
plt.figure(figsize=(14, 7))
sns.barplot(data=plot_df_long, x='vector', y='count', hue='direction', palette={'up': 'green', 'down': 'red'})
plt.title('Распределение vector_strike по направлениям (up vs down)', fontsize=16)
plt.xlabel('Вектор strike', fontsize=12)
plt.ylabel('Количество', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Направление', loc='upper right')
plt.tight_layout()
plt.savefig('vector_strike_distribution.png', dpi=200, bbox_inches='tight')
plt.show()

# Вывод в консоль
print("\nИтоговый DataFrame:")
print(df)