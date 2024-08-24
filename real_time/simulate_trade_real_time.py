"""
Симуляция реал-тайм торговли.
Простая запись в файл открытия позиций и закрытия
"""

import time
from pathlib import Path
from datetime import datetime, timedelta, date
import re
from collections import defaultdict
import pandas as pd
import numpy as np

import keyboard
from keras.src.saving import load_model
from sklearn.preprocessing import MinMaxScaler
from QuikPy import QuikPy  # Работа с QUIK из Python через LUA скрипты QuikSharp

from current_future import get_current_sec_id


class Future:
    def __init__(self, class_code: str, sec_code: str):
        self.class_code_future = class_code
        self.sec_code = sec_code
        self.last = 0.0
        self.low = 0.0
        self.high = 0.0

    def run(self):
        # Получение из Quik цены последней сделки по фьючерсу
        self.last = float(
            qp_provider.get_param_ex(self.class_code_future, self.sec_code, 'LAST')['data']['param_value']
        )
        # Получение из Quik минимальной цены по фьючерсу за торговую сессию
        self.low = float(
            qp_provider.get_param_ex(self.class_code_future, self.sec_code, 'LOW')['data']['param_value']
        )
        # Получение из Quik максимальной цены по фьючерсу за торговую сессию
        self.high = float(
            qp_provider.get_param_ex(self.class_code_future, self.sec_code, 'HIGH')['data']['param_value']
        )


class Options:
    call_code, put_code = 'ABCDEFGHIJKL', 'MNOPQRSTUVWX'

    def __init__(self, class_code_options: str, sec_code_future):
        self.predict_low = 0.0
        self.predict_high = 0.0
        self.predict_close = 0.0
        self.current_oi_dataset = None
        self.sec_code_future = sec_code_future
        self.class_code_options = class_code_options
        self.today_date = datetime.now().strftime("%Y%m%d")  # Текущая дата
        # self.df = pd.DataFrame()  # Data Frame для построения графика ТМВ всех дат истечения опционов
        self.step_strike = 0  # Шаг сетки для графика всех опционов равный шагу страйков
        self.zero_strike = 0

        # Все тикеры класса в список
        class_securities_opt_lst = qp_provider.get_class_securities(class_code_options)['data'][:-1].split(',')
        # print(class_securities_opt_lst)

        self.ba_options_lst = list()  # Список под опционы базового актива
        for sec_code in class_securities_opt_lst:  # Перебираем все тикеры
            # Получаем информацию о тикере (опционе)
            security_info = qp_provider.get_security_info(class_code_options, sec_code)["data"]
            ticker_ba = security_info['base_active_seccode']  # Базовый актив опциона
            exp_date = security_info['exp_date']  # Дата экспирации
            if ticker_ba == self.sec_code_future and int(self.today_date) < exp_date:
                self.ba_options_lst.append(sec_code)
        # print(self.ba_options_lst)

    def run(self):
        self.update()
        self.predict_close = self.predict_price(Path(fr'../close_best_model.keras'))[0][0] + self.zero_strike
        self.predict_close = round(self.predict_close, -1)
        # print(self.predict_close, future.last)
        self.predict_high = self.predict_price(Path(fr'../high_best_model.keras'))[0][0] + self.zero_strike
        self.predict_high = round(self.predict_high, -1)
        # print(self.predict_high, future.high)
        self.predict_low = self.predict_price(Path(fr'../low_best_model.keras'))[0][0] + self.zero_strike
        self.predict_low = round(self.predict_low, -1)
        # print(self.predict_low, future.low)

    def update(self):
        """

        """
        call_dic, put_dic = defaultdict(int), defaultdict(int)
        for sec_code in self.ba_options_lst:  # Перебираем тикеры
            numbers = re.findall(r'\d+', sec_code)  # Создание списка наборов цифр из тикера
            letters = re.findall(r'\D+', sec_code)  # Создание списка наборов букв из тикера
            if letters[1][1] in Options.call_code:
                call_dic[int(numbers[0])] += int(float(qp_provider.get_param_ex(
                    self.class_code_options, sec_code, 'NUMCONTRACTS')['data']['param_value']))
            else:
                put_dic[int(numbers[0])] += int(float(qp_provider.get_param_ex(
                    self.class_code_options, sec_code, 'NUMCONTRACTS')['data']['param_value']))
        # print(call_dic)
        # print(put_dic)

        # Словари с опционами в DF
        df_call = pd.DataFrame(list(call_dic.items()), columns=['strike', 'call'])
        df_put = pd.DataFrame(list(put_dic.items()), columns=['strike', 'put'])
        df = pd.merge(df_call, df_put, how='outer', on='strike')  # Склейка df по индексу
        df = df.fillna(0)  # Замена Nan на 0
        # print(df)

        self.step_strike = int(df['strike'].diff().min())  # Шаг между страйками
        # print(self.step_strike)
        # Временный DF со страйками (для последующего слияния), чтобы исключить пропуски страйков
        df_tmp = pd.DataFrame(columns=['strike'])
        for st in range(df.strike.min(), df.strike.max() + self.step_strike, self.step_strike):
            # Новое значение для добавления
            new_row = pd.DataFrame({'strike': [st]})
            # Добавление новой строки в DataFrame с помощью concat
            df_tmp = pd.concat([df_tmp, new_row], ignore_index=True)
        df = pd.merge(df, df_tmp, on='strike', how='outer').sort_values('strike')
        # Приведение типов с помощью infer_objects
        df = df.infer_objects(copy=False)
        # print(df.to_string(max_rows=4, max_cols=10))

        df = df.fillna(0)  # Пустые значения заполняем "0"
        # Приведение типов к int
        df[['strike', 'call', 'put']] = df[['strike', 'call', 'put']].astype(int)
        df = df.sort_values('strike').reset_index(drop=True)
        # print(df.to_string(max_rows=4, max_cols=10))
        # print(df.shape)

        # Переделываем колонку с ОИ по call в колонку с накопленной суммой ОИ
        df['call'] = df['call'].cumsum()
        # Переделываем колонку с ОИ по put в колонку с накопленной суммой ОИ в обратном порядке
        df['put'] = df.iloc[::-1]['put'].cumsum()[::-1]
        # print(df.to_string(max_rows=4, max_cols=10))

        # Расчитываем ближайший страйк к цене CLOSE
        self.zero_strike = round(future.last / self.step_strike) * self.step_strike
        # print(f'{trade_date=}, {price_open=}, {price_close=}, {price_high=}, {price_low=}, {nearest_strike=}')

        # Список индексов со значением nearest_strike из поля merged_df['STRIKE']
        index_lst = df.index[df['strike'] == self.zero_strike].tolist()
        # Известный индекс строки (берем 0)
        index_nearest = index_lst[0]
        # Определение диапазона строк
        start_index = max(0, index_nearest - 10)
        end_index = min(len(df), index_nearest + 10 + 1)
        # Получение 10 строк до и 10 строк после строки с известным индексом ближайшего страйка
        subset_df = df.iloc[start_index:end_index]
        subset_df = subset_df.copy()
        # Создание колонки соотношений (разницы) отрытого интереса Call и Put
        subset_df['oi'] = subset_df.apply(
            lambda x: x.put - x.call if x.strike < future.last else x.call - x.put, axis=1
        )
        # print(subset_df.to_string(max_rows=40, max_cols=10))

        # Создаем объект MinMaxScaler
        scaler = MinMaxScaler()
        # Применяем нормализацию 0-1 к колонке oi
        subset_df['oi_norm'] = scaler.fit_transform(subset_df[['oi']])
        # print(subset_df.to_string(max_rows=40, max_cols=10))

        subset_df['strike_abc'] = subset_df['strike'] - self.zero_strike
        subset_df = subset_df.set_index('strike_abc')
        subset_df = subset_df.sort_index(ascending=True)  # ascending=False - сортировка по убываю
        subset_df = subset_df[['oi_norm']].T
        # Округляем значения в столбцах с 0 по 20 до 6 знаков после запятой
        subset_df.iloc[:, 0:21] = subset_df.iloc[:, 0:21].round(6)

        # Преобразование в DataSet
        dataset = subset_df.values  # Dataframe преобразуем в Dataset для Keras
        self.current_oi_dataset = dataset.astype(np.float32)  # Смена типа для корректной работы Keras
        # print(subset_df.to_string(max_rows=4, max_cols=22))

    def predict_price(self, patch_model):
        model = load_model(patch_model)  # Загрузка сохраненной лучшей модели
        predictions = model.predict(self.current_oi_dataset)  # На выходе двумерный массив numpy с предсказаниями
        return predictions


def read_log_file(file_path):
    with open(file_path, 'r') as f:
        last_line = f.readlines()[-1]
        # print(last_line.split(';')[0])
    return last_line.split(';')[0], last_line.split(';')[1]


def save_log_file(file_path, today_date, label, predict_close, last, predict_high, high, predict_low, low):
    with open(file_path, 'a') as f:
        f.write(f'{today_date};{label};{predict_close};{last};{predict_high};{high};{predict_low};{low}\n')


if __name__ == '__main__':  # Точка входа при запуске этого скрипта
    qp_provider = QuikPy()  # Подключение к локальному запущенному терминалу QUIK

    ticker: str = 'RTS'
    class_code_future: str = 'SPBFUT'
    sec_code_future: str = get_current_sec_id(ticker)
    class_code_options: str = 'SPBOPT'
    file_path = Path(fr'../log_simulate_trade.txt')

    future = Future(class_code_future, sec_code_future)
    opt = Options(class_code_options, sec_code_future)

    while True:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        future.run()
        opt.run()
        print(opt.predict_close, future.last)
        print(opt.predict_high, future.high)
        print(opt.predict_low, future.low)
        last_datetime_file, last_label_file = read_log_file(file_path)  # Чтение из файла
        print(f'{last_datetime_file=}  {last_label_file=}')

        if last_label_file == 'CLOSE':  # Нет открытых позиций
            if future.last < opt.predict_low:  # Открываем BUY
                save_log_file(
                    file_path, current_datetime, 'BUY',
                    opt.predict_close, future.last,
                    opt.predict_high, future.high,
                    opt.predict_low, future.low
                )
            elif future.last > opt.predict_high:  # Открываем SELL
                save_log_file(
                    file_path, current_datetime, 'SELL',
                    opt.predict_close, future.last,
                    opt.predict_high, future.high,
                    opt.predict_low, future.low
                )
        else:  # Есть открытые позиции
            if (last_label_file == 'BUY') and (future.last > opt.predict_close):  # Закрываем BUY
                save_log_file(
                    file_path, current_datetime, 'CLOSE',
                    opt.predict_close, future.last,
                    opt.predict_high, future.high,
                    opt.predict_low, future.low
                )
            elif (last_label_file == 'SELL') and (future.last < opt.predict_close):  # Закрываем SELL
                save_log_file(
                    file_path, current_datetime, 'CLOSE',
                    opt.predict_close, future.last,
                    opt.predict_high, future.high,
                    opt.predict_low, future.low
                )

        time.sleep(30)  # Ожидание 30 секунду перед следующей проверкой
        if keyboard.is_pressed('q'):  # Замените 'q' на любую клавишу, которой хотите завершать цикл
            print("Завершение работы")
            break

    # print(f'{future.last=}, {future.low=}, {future.high=}')

    # Перед выходом закрываем соединение для запросов и поток обработки функций обратного вызова
    qp_provider.close_connection_and_thread()
