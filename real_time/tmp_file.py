"""
Проба записи в файл и чтения параметра из последней строки
"""

import time
import random
from datetime import datetime
from pathlib import Path


def save_tmp_file(file_path, today_date, label, predict_close, last, predict_high, high, predict_low, low):
    with open(file_path, 'a') as f:
        f.write(f'{today_date};{label};{predict_close};{last};{predict_high};{high};{predict_low};{low}\n')
        print(today_date, label, last)


def read_tmp_file(file_path):
    with open(file_path, 'r') as f:
        last_line = f.readlines()[-1]
        # print(last_line.split(';')[0], last_line.split(';')[1])
    return last_line.split(';')[1]


if __name__ == '__main__':
    file_path = Path(fr'../log_tmp.txt')

    while True:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Метка последней записи

        if Path(file_path).exists():
            last_label_file = read_tmp_file(file_path)
        else:
            last_label_file = 'CLOSE'

        future_last = random.randrange(197500, 202500, 10)
        future_low = random.randrange(195000, 197500, 10)
        future_high = random.randrange(202500, 205000, 10)

        predict_close = random.randrange(197500, 202500, 10)
        predict_low = random.randrange(195000, 200000, 10)
        predict_high = random.randrange(200000, 205000, 10)

        if last_label_file == 'CLOSE':  # Нет открытых позиций
            if future_last < predict_low:  # Открываем BUY
                save_tmp_file(
                    file_path, current_datetime, 'BUY',
                    predict_close, future_last,
                    predict_high, future_high,
                    predict_low, future_low
                )
            elif future_last > predict_high:  # Открываем SELL
                save_tmp_file(
                    file_path, current_datetime, 'SELL',
                    predict_close, future_last,
                    predict_high, future_high,
                    predict_low, future_low
                )
        else:  # Есть открытые позиции
            if (last_label_file == 'BUY') and (future_last > predict_close):  # Закрываем BUY
                save_tmp_file(
                    file_path, current_datetime, 'CLOSE',
                    predict_close, future_last,
                    predict_high, future_high,
                    predict_low, future_low
                )
            elif (last_label_file == 'SELL') and (future_last < predict_close):  # Закрываем SELL
                save_tmp_file(
                    file_path, current_datetime, 'CLOSE',
                    predict_close, future_last,
                    predict_high, future_high,
                    predict_low, future_low
                )

        time.sleep(1)  # Ожидание 1 секунды перед следующей проверкой
