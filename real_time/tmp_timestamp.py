# import datetime
from datetime import datetime

# time_stamp_value = 1725443109361.9
#
# date_time = datetime.datetime.fromtimestamp(time_stamp_value)
# print(date_time)

timestamp_with_micros = 1629788400.123456 # Example timestamp with microseconds
# timestamp_with_micros = 1725443109361.9 # Example timestamp with microseconds
dt_object_with_micros = datetime.fromtimestamp(timestamp_with_micros)
print("Datetime with Microseconds:", dt_object_with_micros)