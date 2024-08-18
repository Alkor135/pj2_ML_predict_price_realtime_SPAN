"""
RTS
Скрипт обучает модели на основе нейронной сети и сохраняет в файлы.
Идея: Точка минимальных выплат SPAN
Модели для прогноза Close, High, Low
"""

import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.layers import Dense, BatchNormalization, Dropout
from keras.src.saving import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# import tensorflow as tf
#
# tf.random.set_seed(9)


class NeuralNetworkModel:
    def __init__(self, df, target='close') -> None:
        self.df = df
        self.file_name_save = fr'../{target}_best_model.keras'
        self.target = f'{target}_abc'  # название целевого предсказания (close, high, low)
        # self.file_name_save = fr"{path_model}\nn_model_{self.table}_{self.target}_drop_bn_es_mc_value_loss.h5"

    def get_model(self):
        column_names_features = self.df.columns.tolist()[0:21]
        # print(column_names_features)

        df = self.df[column_names_features + [self.target]]  # DF для обучения и тестирования
        df = df.fillna(0)  # Замена NaN на "0"
        # print(df.to_string(max_rows=6, max_cols=20))
        # print(df.shape)

        # Преобразование в DataSet
        dataset = df.values  # Dataframe преобразуем в Dataset для Keras
        dataset = dataset.astype(np.float32)  # Смена типа для корректной работы Keras
        # Все, что стоит перед запятой, относится к строкам массива, а все, что стоит после запятой,
        # относится к столбцам массивов.
        X = dataset[:, 0:-1]  # Срез массива по фичам
        y = dataset[:, -1]  # Срез массива по labels

        # Разделение данных на обучающую, тестовую и валидационную выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, shuffle=True)

        # Архитектура сети
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        for _ in range(2):
            model.add(Dense(256, activation='relu', ))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
        model.add(Dense(1))

        # Параметры модели
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Определение функций обратного вызова
        # Функция обратного вызова ранней остановки
        es = EarlyStopping(monitor='val_loss', mode='min', patience=2000, verbose=0)
        # Функция обратного вызова для сохранения лучшей модели
        mc = ModelCheckpoint(self.file_name_save, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

        # Тренировка модели
        num_epochs = 10000
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, validation_split=0.20, callbacks=[es, mc],
                  verbose=1)

        del model  # deletes the existing model
        model = load_model(self.file_name_save)  # Загрузка сохраненной лучшей модели

        predictions = model.predict(X_test)  # На выходе двумерный массив numpy с предсказаниями

        mae: int = int(mean_absolute_error(predictions, y_test))  # Средняя абсолютная ошибка для данных
        r2: float = r2_score(y_test, predictions)  # Коэффициент детерминации

        print(f'Таргет: {self.target}.')
        print(f'Средняя абсолютная ошибка {mae=:,} R2_score={r2:.3f}. Эпох: {num_epochs}')
        # print(model.summary())

        # # Сохранение модели в Keras
        # model.save(
        #     self.file_name_save,  # Путь и название файла(папки) для сохранения
        #     include_optimizer=True
        # )

        print(f'Модель сохранена в: {self.file_name_save}\n')


if __name__ == '__main__':
    # Загружаем файл с разделителем ';' в DF
    df = pd.read_csv(fr'../nn_features_and_target.csv', delimiter=';', index_col='date')
    # print(df.to_string(max_rows=6, max_cols=20))
    # print(df.shape)

    model_all_close = NeuralNetworkModel(df, "close")
    model_all_close.get_model()

    # model_all_high = NeuralNetworkModel(path_model, "All_opt", connection, "high")
    # model_all_high.get_model()
    #
    # model_all_low = NeuralNetworkModel(path_model, "All_opt", connection, "low")
    # model_all_low.get_model()
    #
    # model_nearest_close = NeuralNetworkModel(path_model, "Nearest", connection, "close")
    # model_nearest_close.get_model()
    #
    # model_nearest_high = NeuralNetworkModel(path_model, "Nearest", connection, "high")
    # model_nearest_high.get_model()
    #
    # model_nearest_low = NeuralNetworkModel(path_model, "Nearest", connection, "low")
    # model_nearest_low.get_model()
