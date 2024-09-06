from multiprocessing import Queue
import threading
import sys  # Выход из точки входа
from datetime import datetime
from pathlib import Path

from QuikPy import QuikPy  # Работа с QUIK из Python через LUA скрипты QUIK#


def client():
    """
    Функция получает данные от сервера и помещает их в очередь.
    :return:
    """
    while True:  # "вечный" цикл
        is_connected = qp_provider.is_connected()['data']  # Состояние подключения терминала к серверу QUIK
        if is_connected == 0:  # Если нет подключения терминала QUIK к серверу
            # Перед выходом закрываем соединение для запросов и поток обработки функций обратного вызова
            qp_provider.close_connection_and_thread()
            sys.exit()  # Выходим, дальше не продолжаем

        # qp_provider.on_all_trade = lambda x: queue_stream.put(x)

        qp_provider.on_all_trade = lambda x: queue_stream.put(x) \
            if x['data']['seccode'] == sec_code and x['data']['class_code'] == class_code \
            else ''


def parser():
    """
    Функция парсит данные из очереди, записывает в файл, выводит на экран
    :return:
    """
    while True:
        parse_dic = {}
        parse = queue_stream.get()  # Получаем из очереди данные от клиента

        # date_time = datetime(2020, 2, 28, 23, 55, 52, 119673)
        parse_dic['date_time'] = datetime(
            parse['data']['datetime']['year'],
            parse['data']['datetime']['month'],
            parse['data']['datetime']['day'],
            parse['data']['datetime']['hour'],
            parse['data']['datetime']['min'],
            parse['data']['datetime']['sec'],
            parse['data']['datetime']['mcs']
        ).strftime("%Y-%m-%d %H:%M:%S.%f")
        parse_dic['price'] = parse['data']['price']
        parse_dic['quantity'] = int(parse['data']['qty'])
        if parse['data']['flags'] == 1026:
            parse_dic['flag'] = 'buy'
        elif parse['data']['flags'] == 1025:
            parse_dic['flag'] = 'sell'
        parse_dic['oi'] = parse['data']['open_interest']

        with open(file_path, 'a') as f:
            f.write(
                f'{parse_dic['date_time']};'
                f'{parse_dic['price']};'
                f'{parse_dic['quantity']};'
                f'{parse_dic['flag']};'
                f'{parse_dic['oi']}\n'
            )
        print(parse_dic)


if __name__ == '__main__':
    # Изменяемые настройки
    class_code = 'SPBFUT'  # Класс тикера
    sec_code = 'RIU4'  # Тикер

    today_datetime = datetime.now().strftime("%Y-%m-%d")  # Текущая дата и время
    file_path = Path(fr'../{today_datetime}_{sec_code}_ticks.csv')
    if not Path(file_path).exists():
        with open(file_path, 'a') as file:
            file.write(f'date_time;price;quantity;flag;oi\n')
            print(f'Файл: {file_path} создан')

    qp_provider = QuikPy()  # Подключение к локальному запущенному терминалу QUIK по портам по умолчанию

    queue_stream = Queue()  # Создаем очередь

    # Запускаем клиент в своем потоке
    threading_client = threading.Thread(name='client', target=client)
    threading_client.start()

    # Запускаем парсер в своем потоке
    threading_parser = threading.Thread(name='parser', target=parser)
    threading_parser.start()
