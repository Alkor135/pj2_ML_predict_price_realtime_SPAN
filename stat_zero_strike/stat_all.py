"""
Скрипт анализирует финансовые данные из features_and_target.pkl.
Преобразует ценовые столбцы и close_abc в бинарные значения (0/1).
Формирует признак 'vector' из комбинации трёх strike-точек и close_abc.
Определяет направление движения (up/down) по телу свечи и сдвигает его как целевую переменную.
Балансирует классы up/down, отбирая равное количество последних наблюдений.
Выводит статистику распределения векторов для каждого направления.
Строит и сохраняет сравнительную столбчатую диаграмму распределений.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля графиков
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

pkl_path = Path(__file__).parent.parent / 'nn_data_prepare/features_and_target.pkl'

# Чтение данных из pkl файла
df = pd.read_pickle(pkl_path)

df = df.reset_index()
df = df[['date', -2500, 0, 2500, 'close_abc', 'body']]

# Преобразование колонки 'date' в datetime и сортировка по ней
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Замена значений в столбцах -2500, 0, 2500 на 1 или 0 в зависимости от порога
columns_to_update = [-2500, 0, 2500]
for col in columns_to_update:
    df[col] = (df[col] > 0.000001).astype(int)

# Замена значений в 'close_abc': 0 если < 0, и 1 если >= 0
df['close_abc'] = (df['close_abc'] >= 0).astype(int)

# Создание новой колонки 'vector' со списком значений из столбцов -2500, 0, 2500, 'close_abc'
df['vector'] = df[[-2500, 0, 2500, 'close_abc']].values.tolist()

# Создание колонки 'direction' на основе значения 'body'
df['direction'] = df['body'].apply(lambda x: 'up' if x > 0 else 'down')

# Создание колонки 'next_direction' — сдвиг значения 'direction' на следующий ряд
df['next_direction'] = df['direction'].shift(-1)

# Оставляем только нужные колонки
df = df[['date', 'vector', 'next_direction']]

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
print("\nСтатистика по значениям vector:")
print("\nРаспределение vector_strike для next_direction = 'up':")
up_vectors = df[df['next_direction'] == 'up']['vector']
print(up_vectors.value_counts().to_string())

print("\nРаспределение vector для next_direction = 'down':")
down_vectors = df[df['next_direction'] == 'down']['vector']
print(down_vectors.value_counts().to_string())

# Общая сводка
print(f"\nВсего записей: {len(df)}")
print(f"up: {len(up_vectors)}, down: {len(down_vectors)}")

# === Подготовка данных для графика ===
# Преобразуем списки в кортежи (чтобы сделать хешируемыми)
up_counts = up_vectors.apply(tuple).value_counts().rename('up').reset_index()
up_counts.columns = ['vector', 'up']

down_counts = down_vectors.apply(tuple).value_counts().rename('down').reset_index()
down_counts.columns = ['vector', 'down']

# Объединение по вектору
plot_df = pd.merge(up_counts, down_counts, on='vector', how='outer').fillna(0)

# Преобразуем в строковое представление для отображения на графике
plot_df['vector'] = plot_df['vector'].astype(str)

# Преобразуем в длинный формат для seaborn
plot_df_long = plot_df.melt(id_vars='vector', value_vars=['up', 'down'],
                            var_name='direction', value_name='count')

# === Построение столбчатой диаграммы ===
plt.figure(figsize=(16, 8))
sns.barplot(data=plot_df_long, x='vector', y='count', hue='direction', palette={'up': 'green', 'down': 'red'})
plt.title('Распределение vector по направлениям (up vs down)', fontsize=16)
plt.xlabel('Вектор (strike + close_abc)', fontsize=12)
plt.ylabel('Количество', fontsize=12)
plt.xticks(rotation=60)
plt.legend(title='Направление', loc='upper right')
plt.tight_layout()
plt.savefig('vector_distribution_up_vs_down.png', dpi=200, bbox_inches='tight')
plt.show()

# Вывод в консоль
print("\nИтоговый DataFrame:")
print(df)