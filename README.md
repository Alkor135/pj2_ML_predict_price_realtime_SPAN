# pj2_ML_predict_price_realtime_SPAN

Предсказание цены фьючерса RTS по открытому интересу опционов (SPAN) с использованием нейронных сетей.

## Описание

Система загружает данные по фьючерсам и опционам с MOEX ISS API, подготавливает признаки на основе открытого интереса (Open Interest) опционов, обучает нейросеть для предсказания направления цены и позволяет применять модель в реальном времени через QUIK.

## Структура проекта

```
├── download_MOEX_ISS_API_to_db/
│   ├── download_futures_and_options.py   # Загрузка фьючерсов и опционов с MOEX ISS API
│   └── test_db_futures_options.py        # Тестирование загруженных данных
├── nn_data_prepare/
│   ├── data_prepare.py                   # Подготовка данных для обучения
│   ├── data_prepare_01.py               # Альтернативная подготовка
│   ├── data_prepare.ipynb               # Подготовка (ноутбук)
│   └── options_1min.ipynb               # Анализ минутных опционных данных
├── nn_train_and_save_model/
│   └── nn_save_model_RTS.py             # Обучение и сохранение нейросети
├── real_time/
│   ├── simulate_trade_real_time.py       # Симуляция торговли в реальном времени
│   └── current_future.py                # Получение текущих данных фьючерса
├── stat_zero_strike/
│   ├── stat_price_regarding_zero_strike.py  # Статистика цены относительно центрального страйка
│   ├── stat_all.py                      # Общая статистика
│   ├── stat_3_strike.py                 # Статистика по 3 страйкам
│   └── simulate_trade_all.py            # Симуляция торговли
├── result_analysis/
│   └── log_analysis.py                  # Анализ логов торговли
├── QuikPy/                              # Библиотека подключения к QUIK
└── requirements.txt
```

## Пайплайн

1. **Загрузка данных** — фьючерсы и опционы с MOEX ISS API в SQLite
2. **Подготовка признаков** — фичи из открытого интереса опционов (SPAN)
3. **Обучение модели** — нейронная сеть (PyTorch)
4. **Применение** — предсказание в реальном времени через QUIK

## Зависимости

- Python 3.10+
- PyTorch, pandas, numpy
- Терминал QUIK (для реальной торговли)
