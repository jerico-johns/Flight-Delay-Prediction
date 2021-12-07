# Databricks notebook source
# !find / -name "phase1*"

# COMMAND ----------

# import common
import pprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql.functions import col, isnan, when, count, mean, min, max, stddev, variance, isnull, isnan, log10
# from IPython.display import display, Markdown

pp = pprint.PrettyPrinter(indent=2)

# COMMAND ----------

# spark = common.get_spark()
root_data_path = '/mnt/mids-w261/datasets_final_project'

# COMMAND ----------

df_airlines = spark.read.parquet(f'{root_data_path}/parquet_airlines_data_3m/')

# COMMAND ----------

df_airlines_fields = { 
  'ACTUAL_ELAPSED_TIME': {'description': 'Elapsed Time of Flight, in Minutes', 'type': 'numerical', 'is_feature': True},
  'AIR_TIME': {'description': 'Minutes in the air. Wheels up -> Wheels down', 'type': 'numerical', 'is_feature': True},
  'ARR_DEL15': {'description': 'Arrival Delay Indicator, 15 Minutes or More (1=Yes)', 'type': 'categorical'},
  'ARR_DELAY': {'description': 'Arrival Delay in minutes ', 'type': 'numerical'},
  'ARR_DELAY_GROUP': {'description': 'Arrival Delay intervals, every (15-minutes from <-15 to >180)', 'type': 'categorical'},
  'ARR_DELAY_NEW': {'description': 'Difference in minutes between scheduled and actual arrival time. Early arrivals set to 0.', 'type': 'numerical', 'is_feature': True},
  'ARR_TIME': {'description': 'Actual Arrival Time (local time: hhmm)', 'type': 'numerical'},
  'ARR_TIME_BLK': {'description': 'CRS Arrival Time Block, Hourly Intervals', 'type': 'categorical'},
  'CANCELLATION_CODE': {'description': 'Specifies The Reason For Cancellation', 'type': 'categorical'},
  'CANCELLED': {'description': 'Cancelled Flight Indicator (1=Yes)', 'type': 'categorical'},
  'CARRIER_DELAY': {'description': 'Carrier Delay, in Minutes', 'type': 'numerical', 'is_feature': True},
  'CRS_ARR_TIME': {'description': 'CRS Arrival Time (local time: hhmm)', 'type': 'numerical'},
  'CRS_DEP_TIME': {'description': 'CRS Departure Time (local time: hhmm)', 'type': 'numerical'},
  'CRS_ELAPSED_TIME': {'description': 'CRS Elapsed Time of Flight, in Minutes', 'type': 'numerical'},
  'DAY_OF_MONTH': {'description': 'Day of Month', 'type': 'numerical', 'is_feature': True},
  'DAY_OF_WEEK': {'description': 'Day of Week', 'type': 'numerical', 'is_feature': True},
  'DEP_DEL15': {'description': 'Departure Delay Indicator, 15 Minutes or More (1=Yes)', 'type': 'categorical'},
  'DEP_DELAY': {'description': 'Difference in minutes between scheduled and actual departure time. Early departures show negative numbers.', 'type': 'numerical'},
  'DEP_DELAY_GROUP': {'description': 'Departure Delay intervals, every (15 minutes from <-15 to >180)', 'type': 'categorical'},
  'DEP_DELAY_NEW': {'description': 'Difference in minutes between scheduled and actual departure time. Early departures set to 0.', 'type': 'numerical', 'is_feature': True},
  'DEP_TIME': {'description': 'Actual Departure Time (local time: hhmm)', 'type': 'numerical'},
  'DEP_TIME_BLK': {'description': 'CRS Departure Time Block, Hourly Intervals', 'type': 'categorical'},
  'DEST': {'description': 'Destination Airport', 'type': 'categorical'},
  'DEST_AIRPORT_ID': {'description': 'Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.', 'type': 'categorical', 'is_feature': True},
  'DEST_AIRPORT_SEQ_ID': {'description': 'Destination Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time.', 'type': 'categorical'},
  'DEST_CITY_MARKET_ID': {'description': 'Destination Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market.', 'type': 'categorical', 'is_feature': True},
  'DEST_CITY_NAME': {'description': 'Destination Airport, City Name.', 'type': 'categorical'},
  'DEST_STATE_ABR': {'description': 'Destination Airport, State Abbreviation.', 'type': 'categorical'},
  'DEST_STATE_FIPS': {'description': 'Destination Airport, FIPS code.', 'type': 'categorical'},
  'DEST_STATE_NM': {'description': 'Destination Airport, State Number.', 'type': 'categorical'},
  'DEST_WAC': {'description': 'Destination Airport, World Area Code.', 'type': 'categorical'},
  'DISTANCE': {'description': 'Distance travelled.', 'type': 'numerical', 'is_feature': True},
  'DISTANCE_GROUP': {'description': 'Distance Intervals, every 250 Miles, for Flight Segment', 'type': 'categorical'},
  'DIV1_AIRPORT': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV1_AIRPORT_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV1_AIRPORT_SEQ_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV1_LONGEST_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV1_TAIL_NUM': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV1_TOTAL_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV1_WHEELS_OFF': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV1_WHEELS_ON': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV2_AIRPORT': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV2_AIRPORT_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV2_AIRPORT_SEQ_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV2_LONGEST_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV2_TAIL_NUM': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV2_TOTAL_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV2_WHEELS_OFF': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV2_WHEELS_ON': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV3_AIRPORT': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV3_AIRPORT_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV3_AIRPORT_SEQ_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV3_LONGEST_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV3_TAIL_NUM': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV3_TOTAL_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV3_WHEELS_OFF': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV3_WHEELS_ON': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV4_AIRPORT': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV4_AIRPORT_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV4_AIRPORT_SEQ_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV4_LONGEST_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV4_TAIL_NUM': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV4_TOTAL_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV4_WHEELS_OFF': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV4_WHEELS_ON': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV5_AIRPORT': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV5_AIRPORT_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV5_AIRPORT_SEQ_ID': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV5_LONGEST_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV5_TAIL_NUM': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV5_TOTAL_GTIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV5_WHEELS_OFF': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV5_WHEELS_ON': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIVERTED': {'description': 'Indicates if the flight was diverted (1 = Yes).', 'type': 'categorical'},
  'DIV_ACTUAL_ELAPSED_TIME': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV_AIRPORT_LANDINGS': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV_ARR_DELAY': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV_DISTANCE': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'DIV_REACHED_DEST': {'description': 'Diverted Airport Information - Unused.', 'type': None},
  'FIRST_DEP_TIME': {'description': 'First Gate Departure Time at Origin Airport', 'type': 'string'},
  'FLIGHTS': {'description': 'Number of Flights', 'type': 'numerical'},
  'FL_DATE': {'description': 'Flight Date (yyyymmdd)', 'type': 'categorical'},
  'LATE_AIRCRAFT_DELAY': {'description': 'Late Aircraft Delay, in Minutes', 'type': 'numerical', 'is_feature': True},
  'LONGEST_ADD_GTIME': {'description': 'Longest Time Away from Gate for Gate Return or Cancelled Flight', 'type': 'numerical'},
  'MONTH': {'description': 'Month', 'type': 'numerical', 'is_feature': True},
  'NAS_DELAY': {'description': 'National Air System Delay, in Minutes', 'type': 'numerical', 'is_feature': True},
  'OP_CARRIER': {'description': 'Commerical Operator.', 'type': 'categorical'},
  'OP_CARRIER_AIRLINE_ID': {'description': 'Commerical Operator, ID', 'type': 'categorical'},
  'OP_CARRIER_FL_NUM': {'description': 'Commerical Operator Flight Number', 'type': 'string'},
  'OP_UNIQUE_CARRIER': {'description': 'Commerical Operator, Unique Carrier Code.', 'type': 'string'},
  'ORIGIN': {'description': 'Origin Airport', 'type': ''},
  'ORIGIN_AIRPORT_ID': {'description': 'Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.', 'type': 'cateogrical', 'is_feature': True},
  'ORIGIN_AIRPORT_SEQ_ID': {'description': 'Origin Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time.', 'type': 'categorical'},
  'ORIGIN_CITY_MARKET_ID': {'description': 'Origin Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market.', 'type': 'categorical', 'is_feature': True},
  'ORIGIN_CITY_NAME': {'description': 'Origin Airport, City Name', 'type': 'categorical'},
  'ORIGIN_STATE_ABR': {'description': 'Origin Airport, State Code', 'type': 'categorical'},
  'ORIGIN_STATE_FIPS': {'description': 'Origin Airport, State Fips', 'type': 'categorical'},
  'ORIGIN_STATE_NM': {'description': 'Origin Airport, State Name', 'type': 'categorical'},
  'ORIGIN_WAC': {'description': 'Origin Airport, World Area Code', 'type': 'categorical'},
  'QUARTER': {'description': 'Quarter (1-4)', 'type': 'categorical'},
  'SECURITY_DELAY': {'description': 'Security Delay, in Minutes', 'type': 'numerical'},
  'TAIL_NUM': {'description': 'Tail Number', 'type': 'categorical'},
  'TAXI_IN': {'description': 'Taxi In Time, in Minutes', 'type': 'numerical', 'is_feature': True},
  'TAXI_OUT': {'description': 'Taxi Out Time, in Minutes', 'type': 'numerical', 'is_feature': True},
  'TOTAL_ADD_GTIME': {'description': 'Total Ground Time Away from Gate for Gate Return or Cancelled Flight', 'type': 'numerical'},
  'WEATHER_DELAY': {'description': 'Weather Delay, in Minutes', 'type': 'numerical', 'is_feature': True},
  'WHEELS_OFF': {'description': 'Wheels Off Time (local time: hhmm)', 'type': 'numerical'},
  'WHEELS_ON': {'description': 'Wheels On Time (local time: hhmm)', 'type': 'numerical'},
  'YEAR': {'description': 'Years', 'type': 'numerical'}
}

# COMMAND ----------

def generate_row(df, column_name, info, row_count):
    
    column_type = info['type']
    column_desc = info['description']
    exprs = [count(when(isnan(column_name) | col(column_name).isNull(), column_name)).alias('null_count')]
    stats = df.agg(*exprs).first()
    null_percent = round((stats['null_count'] / row_count) * 100, 2)
    
    if column_type == 'numerical':
        exprs = [mean(col(column_name)).alias('avg'), min(col(column_name)), 
                 max(col(column_name)), variance(col(column_name)), 
                 stddev(col(column_name))]
        stats = df.agg(*exprs).first()
        mean_val = round(stats['avg'], 4)
        min_val = round(stats[f'min({column_name})'], 4)
        max_val = round(stats[f'max({column_name})'], 4)
        variance_val = round(stats[f'var_samp({column_name})'], 4)
        std_dev_val = round(stats[f'stddev_samp({column_name})'], 4)

    else:
        mean_val = ''
        min_val = ''
        max_val = ''
        variance_val = ''
        std_dev_val = ''

    return f"<tr><td>{column_name}</td><td>{column_desc}</td><td>{column_type}</td><td>{mean_val}</td><td>{min_val}</td><td>{max_val}</td><td>{variance_val}</td><td>{std_dev_val}</td><td>{null_percent}</td></tr>"

def generate_stats_table(df, fields):
    
    column_table = [
      '<table><thead><tr><th>Column</th><th>Description</th><th>Type</th><th>Mean</th><th>Min</th><th>Max</th><th>Var</th><th>Std Dev</th><th>Null %</th></tr></thead>'
      '<tbody>'
    ]

    funcs = [mean, min, max, variance, stddev]
    row_count = df.count()    

    for column_name, info in fields.items():

        column_type = info['type']

        if column_type is None:
            continue

        row = generate_row(df, column_name, info, row_count)
        column_table.append(row)
        
    column_table.append('</tbody></table')
    
    return ''.join(column_table)

# COMMAND ----------

html = generate_stats_table(df_airlines, df_airlines_fields)
displayHTML(html)

# COMMAND ----------

airline_numerical_fields = [column_name for column_name, info in df_airlines_fields.items() if info['type'] == 'numerical']

df_airlines = df_airlines.withColumn(f'ACTUAL_ELAPSED_TIME_LOG', log10(col("ACTUAL_ELAPSED_TIME")))

for airline_numerical_field in airline_numerical_fields:
  df_airlines = df_airlines.withColumn(f'{airline_numerical_field}_LOG', log10(col(airline_numerical_field)))
#   df_airlines[f'{airline_numerical_field}_log'] = np.log(df_airlines[airline_numerical_field])

# COMMAND ----------


df_sample_values = df_airlines.select(airline_feature_fields).sample(fraction=0.25).toPandas()


# COMMAND ----------


  

# COMMAND ----------

df_sample_values.hist(figsize=(25,25), bins=15)
plt.show()

# COMMAND ----------

df_weather = spark.read.parquet(f"{root_data_path}/weather_data/*")

# COMMAND ----------

# print(df_weather.count())
# sfo = df_weather.where(df_weather.NAME.contains(' KSFO '))

df_weather.take(2)

# COMMAND ----------

df_weather_fields = { 
  'ACTUAL_ELAPSED_TIME': {'description': 'Elapsed Time of Flight, in Minutes', 'type': 'numerical'}
}

# COMMAND ----------

# https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat
