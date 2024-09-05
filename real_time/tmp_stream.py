from multiprocessing import Queue
import threading
import sys  # Выход из точки входа
from datetime import datetime

from QuikPy import QuikPy  # Работа с QUIK из Python через LUA скрипты QUIK#


def client():
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
    while True:
        parse_dic = {}
        parse = queue_stream.get()  # Получаем из очереди данные от клиента

        # date_time = datetime(2020, 10, 17, 10, 22, 14, 119673)
        parse_dic['date_time'] = datetime(
            parse['data']['datetime']['year'],
            parse['data']['datetime']['month'],
            parse['data']['datetime']['day'],
            parse['data']['datetime']['hour'],
            parse['data']['datetime']['min'],
            parse['data']['datetime']['sec'],
            parse['data']['datetime']['mcs']
        )
        parse_dic['price'] = parse['data']['price']
        parse_dic['quantity'] = int(parse['data']['qty'])
        if parse['data']['flags'] == 1026:
            parse_dic['flag'] = 'buy'
        elif parse['data']['flags'] == 1025:
            parse_dic['flag'] = 'sell'
        parse_dic['oi'] = parse['data']['open_interest']

        print(parse_dic)


if __name__ == '__main__':
    # Изменяемые настройки
    # ticker = 'RIU4'
    class_code = 'SPBFUT'  # Класс тикера
    sec_code = 'RIU4'  # Тикер

    qp_provider = QuikPy()  # Подключение к локальному запущенному терминалу QUIK по портам по умолчанию

    queue_stream = Queue()  # Создаем очередь

    # Запускаем клиент в своем потоке
    threading_client = threading.Thread(name='client', target=client)
    threading_client.start()

    # Запускаем парсер в своем потоке
    threading_parser = threading.Thread(name='parser', target=parser)
    threading_parser.start()
