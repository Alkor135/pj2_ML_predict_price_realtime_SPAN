from datetime import datetime
from pathlib import Path


def save_tmp_file(file_path, today_date, label):
    with open(file_path, 'a') as f:
        f.write(f'{today_date};{label};PRICE_LAST;PRICE_HIGH;PRICE_LOW\n')


def read_tmp_file(file_path):
    with open(file_path, 'r') as f:
        last_line = f.readlines()[-1]
        print(last_line.split(';')[0])
    return last_line.split(';')[0], last_line.split(';')[1]


if __name__ == '__main__':
    today_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = Path(fr'../log_tmp.txt')

    # Дата и время из последней строки, а также метка записи
    time_str, label = read_tmp_file(file_path)
    # Преобразование строки в объект datetime
    time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    # Временная метка Unix (количество секунд с 1970-01-01)
    unix_timestamp = int(time_obj.timestamp())
    # Проверка на четность
    is_even = unix_timestamp % 2 == 0

    if (label != 'CLOSE') and is_even:
        save_tmp_file(file_path, today_date, 'CLOSE')
    if is_even:
        save_tmp_file(file_path, today_date, 'label')
