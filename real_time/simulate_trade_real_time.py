"""
Симуляция реал-тайм торговли.
"""

from pathlib import Path
import requests
from datetime import datetime, timedelta, date
from typing import Any

import apimoex
import pandas as pd


def get_current_sec_id(ticker: str) -> str:
    """
    Функция обращается к MOEX API и возвращает текущий sec_id фьючерса на запрошенный инструмент
    :param ticker: str
    :return: str
    """
    trade_date: date = datetime.now().date()  # Текущая дата
    request_url = (f'http://iss.moex.com/iss/engines/futures/markets/forts/securities.json?assets={ticker}')
    arguments = {'securities.columns': ('SECID, LASTTRADEDATE, LASTDELDATE, SHORTNAME, ASSETCODE')}
    with requests.Session() as session:
        iss = apimoex.ISSClient(session, request_url, arguments)
        data = iss.get()
        df = pd.DataFrame(data['securities'])

    df[['LASTTRADEDATE', 'LASTDELDATE']] = df[['LASTTRADEDATE', 'LASTDELDATE']].apply(pd.to_datetime)
    min_date = df['LASTTRADEDATE'].min()
    sec_id = (
        df.query('LASTTRADEDATE > @trade_date')
        .query('LASTTRADEDATE == @min_date')
        .reset_index(drop=True)
        .loc[0, 'SECID']
    )

    return sec_id


if __name__ == '__main__':  # Точка входа при запуске этого скрипта
    ticker: str = 'RTS'

    sec_id: str = get_current_sec_id(ticker)
    print(sec_id)

