from datetime import datetime
import math


timestamp_with_micros = 1629788400.123456  # Example timestamp with microseconds
print(f"{timestamp_with_micros=}")

# timestamp в datetime формат
dt_object_with_micros = datetime.fromtimestamp(timestamp_with_micros)
print("Datetime with Microseconds:", dt_object_with_micros)

# datetime формат в string (строку)
dt_string = dt_object_with_micros.strftime("%Y-%m-%d %H:%M:%S.%f")
print("Datetime в форматированную строку:", dt_string)

# Строку с датой и временем в datetime формат (объект)
dt_object_with_micros_new = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S.%f")
print(f"{dt_object_with_micros_new=}")

# datetime формат в timestamp
time_stamp_new = dt_object_with_micros_new.timestamp()
print(f'{time_stamp_new=}')

# Округление timestamp до секунд в большую сторону
time_stamp_sec = math.ceil(time_stamp_new)
print(f'{time_stamp_sec=}')

