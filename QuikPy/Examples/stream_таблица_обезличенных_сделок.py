import logging  # Выводим лог на консоль и в файл
from datetime import datetime  # Дата и время
import sys  # Выход из точки входа
import time  # Подписка на события по времени

from QuikPy import QuikPy  # Работа с QUIK из Python через LUA скрипты QUIK#


if __name__ == '__main__':  # Точка входа при запуске этого скрипта
    logger = logging.getLogger('QuikPy.Stream')  # Будем вести лог
    qp_provider = QuikPy()  # Подключение к локальному запущенному терминалу QUIK по портам по умолчанию

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Формат сообщения
                        datefmt='%d.%m.%Y %H:%M:%S',  # Формат даты
                        level=logging.DEBUG,  # Уровень логируемых событий NOTSET/DEBUG/INFO/WARNING/ERROR/CRITICAL
                        handlers=[logging.FileHandler('Stream.log'), logging.StreamHandler()])  # Лог записываем в файл и выводим на консоль
    logging.Formatter.converter = lambda *args: datetime.now(tz=qp_provider.tz_msk).timetuple()  # В логе время указываем по МСК

    # class_code = 'TQBR'  # Класс тикера
    # sec_code = 'SBER'  # Тикер

    class_code = 'SPBFUT'  # Класс тикера
    sec_code = 'RIU4'  # Тикер

    # class_code = 'SPBFUT'  # Класс тикера
    # sec_code = 'SiU4'  # Для фьючерсов: <Код тикера><Месяц экспирации: 3-H, 6-M, 9-U, 12-Z><Последняя цифра года>

    # # Подписка на обезличенные сделки. Чтобы получать, в QUIK открыть "Таблицу обезличенных сделок", указать тикер
    # qp_provider.on_all_trade = lambda data: logger.info(data)  # Обработчик получения обезличенной сделки
    # logger.info(f'Подписка на обезличенные сделки {class_code}.{sec_code}')
    # sleep_sec = 3  # Кол-во секунд получения обезличенных сделок
    # logger.info(f'Секунд обезличенных сделок: {sleep_sec}')
    # time.sleep(sleep_sec)  # Ждем кол-во секунд получения обезличенных сделок
    # logger.info(f'Отмена подписки на обезличенные сделки')
    # qp_provider.on_all_trade = qp_provider.default_handler  # Возвращаем обработчик по умолчанию
    #
    # # Просмотр изменений состояния соединения терминала QUIK с сервером брокера
    # qp_provider.on_connected = lambda data: logger.info(data)  # Нажимаем кнопку "Установить соединение" в QUIK
    # qp_provider.on_disconnected = lambda data: logger.info(data)  # Нажимаем кнопку "Разорвать соединение" в QUIK

    while True:  # "вечный" цикл
        is_connected = qp_provider.is_connected()['data']  # Состояние подключения терминала к серверу QUIK
        if is_connected == 0:  # Если нет подключения терминала QUIK к серверу
            # Перед выходом закрываем соединение для запросов и поток обработки функций обратного вызова
            qp_provider.close_connection_and_thread()
            sys.exit()  # Выходим, дальше не продолжаем

        # print(qp_provider.on_connected)
        # print(qp_provider.on_disconnected)
        # qp_provider.on_connected = lambda x: print(x)
        # qp_provider.on_disconnected = lambda x: print(x)

        # data_mess = qp_provider.on_all_trade  # Обработчик получения обезличенной сделки
        # print(data_mess)

        # qp_provider.on_all_trade = lambda data: data
        # print(data)

        # qp_provider.on_all_trade = lambda x: print(x)

        qp_provider.on_all_trade = lambda x: print(x) \
            if x['data']['seccode'] == 'RIU4' and x['data']['class_code'] == 'SPBFUT' \
            else ''

    # Выход
    # Закрываем соединение для запросов и поток обработки функций обратного вызова
    qp_provider.close_connection_and_thread()
