from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_pl(row):
    """Создание колонки 'P/L' на основе условий"""
    vector = row['vector']
    next_direction = row['next_direction']
    body_next = row['body_next']

    if vector == [1, 0, 1, 0]:
        if next_direction == 'up':
            return abs(body_next)
        elif next_direction == 'down':
            return -abs(body_next)
    elif vector == [1, 0, 1, 1]:
        if next_direction == 'up':
            return -abs(body_next)
        elif next_direction == 'down':
            return abs(body_next)
    return 0

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

# Создание колонки 'body_next' на основе значения 'body'
df['body_next'] = df['body'].shift(-1)

# Оставляем только нужные колонки
df = df[['date', 'vector', 'next_direction', 'body_next']]

# Создание колонки 'P/L' на основе условий
df['P/L'] = df.apply(calculate_pl, axis=1)

# Создание колонки 'Cum P/L' как кумулятивной суммы значений 'P/L'
df['Cum P/L'] = df['P/L'].cumsum()

# Построение линейного графика для 'Cum P/L'
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['Cum P/L'], label='Cumulative P/L', color='green', linewidth=1)
plt.title('Cumulative P/L Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit/Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Сохранение графика
plot_path = Path(__file__).parent / 'cum_pl_plot.png'
plt.savefig(plot_path)
print(f"\nГрафик сохранён: {plot_path.resolve()}")

# Показать график
plt.show()

# Вывод в консоль
print("\nИтоговый DataFrame:")
print(df)
