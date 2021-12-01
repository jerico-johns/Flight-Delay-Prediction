# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Team 2, Mid-term Presentation

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC - Create baseline pipelines and do experiments on all the data 
# MAGIC   - Use 2019 data as a blind test set that is never consulted during training.
# MAGIC   - Report evaluation metrics in terms of 2019 dataset
# MAGIC - Create at least one time-based feature, e.g., recency, frequency, monetary, (RFM)
# MAGIC - Create a baseline model (ie, logistic regression, ensembles) and write a gap analysis against the Leaderboard (TODO).
# MAGIC - Fine tune your baseline pipeline using gridsearch
# MAGIC   - Is there a difference in performance? Is it related to features? Is it related to noise? What is impacting the model performance?
# MAGIC 
# MAGIC Presentation
# MAGIC - Introduce the business case
# MAGIC - Introduce the dataset
# MAGIC - Summarize EDA and feature engineering
# MAGIC - Summarize algorithms tried, and justify final algorithm choice
# MAGIC - Discuss evaluation metrics in light of the business case
# MAGIC - Discuss performance (describe cluster, and clock times) and scalability concerns
# MAGIC - Interesting pipeline errors, debugging experiences
# MAGIC - Summarize limitations, challenges, and future work.

# COMMAND ----------

# Notebook Initialization and setup

import pprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import radians, cos, sin, asin, sqrt
from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType, IntegerType, TimestampType
from pyspark.sql.window import Window
from pyspark.ml.feature import Imputer
from datetime import datetime, timedelta

pp = pprint.PrettyPrinter(indent=2)
root_data_path = '/mnt/mids-w261/datasets_final_project'
blob_container = "w261" # The name of your container created in https://portal.azure.com
storage_account = "w261josh" # The name of your Storage account created in https://portal.azure.com
secret_scope = "joshscope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "joshkey" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

INITIALIZE_DATASETS = False
WEATHER_AGGREGATE_WINDOW_SECONDS = 60 * 30 # 30 minutes


def generate_eda_table(df_spark, sample_fraction=0.1, fields={}):
  
    df_pandas = df_spark.sample(fraction=sample_fraction).toPandas()
    row_count = len(df_pandas)
    
    column_table = [
      '<table border="1"><thead>'
      f'<tr><td>Sample #</td><td colspan=8>{row_count}</td></tr>'
      '<tr><th>Column</th><th>Description</th><th>Type</th><th>Mean</th><th>Min</th><th>Max</th><th>Var</th><th>Std Dev</th><th>Null %</th></tr></thead>'
      '<tbody>'
    ]
    
    means = df_pandas.mean()
    mins = df_pandas.min()
    maxs = df_pandas.max()
    variances = df_pandas.var()
    stds = df_pandas.std()
    nulls = df_pandas.count()
    row_count = len(df_pandas)

    for column_name in df_pandas:
      info = fields.get(column_name, {})
      column_desc = info.get('description', None) or ''
      column_type = str(df_spark.schema[column_name].dataType)
      is_numeric = column_type == 'DoubleType'
      
      mean_val = round(means[column_name], 2) if is_numeric else 0
      min_val = round(mins[column_name], 2) if is_numeric else 0
      max_val = round(maxs[column_name], 2) if is_numeric else 0
      variance_val = round(variances[column_name], 2) if is_numeric else 0
      std_dev_val = round(stds[column_name], 2) if is_numeric else 0
      null_percent = round((1 - nulls[column_name] / row_count) * 100, 2)

      row = f"<tr><td>{column_name}</td><td>{column_desc}</td><td>{column_type}</td><td>{mean_val}</td><td>{min_val}</td><td>{max_val}</td><td>{variance_val}</td><td>{std_dev_val}</td><td>{null_percent}</td></tr>"

      column_table.append(row)

    column_table.append(f'</tbody></table>')
    
    return ''.join(column_table), df_pandas

# COMMAND ----------

# display(dbutils.fs.ls(f"{mount_path}"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Business Case

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Airport Info

# COMMAND ----------

if INITIALIZE_DATASETS:
  df_airports = spark.createDataFrame(pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat', 
                                                 header=0, names=['of_id', 'name', 'city', 'country', 'iata', 'icao', 'lat', 'lon', 
                                                                  'altitude', 'utc_offset', 'dst', 'timezone', 'type', 'source']))
  df_airports.write.mode('overwrite').parquet(f"{blob_url}/df_airports")
else:
  df_airports = spark.read.parquet(f'{blob_url}/df_airports/')

df_airports = df_airports.select('name', 'iata', 'icao', 'lat', 'lon', 'timezone', sf.col('utc_offset').cast(IntegerType())) \
                         .filter((df_airports.country == 'United States') & (df_airports.type == 'airport'))

html, _ = generate_eda_table(df_airports, sample_fraction=1.0)
displayHTML(html)

# COMMAND ----------

df_airports.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Weather Stations

# COMMAND ----------

df_stations = spark.read.parquet(f"{root_data_path}/stations_data/*")

html, _ = generate_eda_table(df_stations, sample_fraction=0.01)
displayHTML(html)

# COMMAND ----------

df_stations.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Joining Airports to Weather Stations
# MAGIC 
# MAGIC There's a lot of weather stations and a ton of unnecessary data in the weather dataset.

# COMMAND ----------

def get_distance(lat_a, lon_a, lat_b, lon_b):
  lat_a, lon_a, lat_b, lon_b = map(radians, [lat_a, lon_a, lat_b, lon_b])
  dist_lon = lon_b - lon_a
  dist_lat = lat_b - lat_a

  area = sin(dist_lat/2)**2 + cos(lat_a) * cos(lat_b) * sin(dist_lon / 2)**2
 
  central_angle = 2 * asin(sqrt(area))
  radius = 6371

  distance = central_angle * radius
  
  return abs(distance)

udf_get_distance = udf(get_distance)

if INITIALIZE_DATASETS:
  df_unique_stations = df_stations.groupBy('station_id', sf.col('lat').alias('station_lat'), sf.col('lon').alias('station_lon')).count()
  
  df_airport_station_distances = df_airports.select('iata', 'icao', 'utc_offset', 'timezone',
                                                    sf.col('lat').alias('airport_lat'), 
                                                    sf.col('lon').alias('airport_lon')).join(df_unique_stations)
  
  df_airport_station_distances = df_airport_station_distances.select('*', udf_get_distance(df_airport_station_distances.station_lat, 
                                                                                           df_airport_station_distances.station_lon,
                                                                                           df_airport_station_distances.airport_lat, 
                                                                                      df_airport_station_distances.airport_lon).alias('distance').cast(DoubleType()))
  
  df_airport_station_shortest_distance = df_airport_station_distances.groupBy('icao').agg(sf.min(sf.col('distance')).alias('distance'))
  
  df_closest_airport_station = df_airport_station_distances.join(df_airport_station_shortest_distance, on=['icao', 'distance'])
  df_closest_airport_station = df_closest_airport_station.where('distance <= 5')
  df_closest_airport_station = df_closest_airport_station.drop('airport_lat', 'airport_lon', 'station_lat', 'station_lon', 'count', 'distance')
  
  df_closest_airport_station.write.mode('overwrite').parquet(f"{blob_url}/df_closest_airport_station")
  
else:
  df_closest_airport_station = spark.read.parquet(f'{blob_url}/df_closest_airport_station/')  


# COMMAND ----------

df_closest_airport_station.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Historical Flight Data

# COMMAND ----------

df_flights = spark.read.parquet(f'{root_data_path}/parquet_airlines_data/*')
df_flights = df_flights.toDF(*[c.lower() for c in df_flights.columns])

print(df_flights.count())

# COMMAND ----------

df_flights_fields = {
    'actual_elapsed_time': {'description': 'Elapsed Time of Flight, in Minutes'},
    'air_time': {'description': 'Minutes in the air. Wheels up -> Wheels down'},
    'arr_del15': {'description': 'Arrival Delay Indicator, 15 Minutes or More (1=Yes)'},
    'arr_delay': {'description': 'Arrival Delay in minutes.'},
    'arr_delay_group': {'description': 'Arrival Delay intervals, every (15-minutes from <-15 to >180)'},
    'arr_delay_new': {'description': 'Difference in minutes between scheduled and actual arrival time. Early arrivals set to 0.'},
    'arr_time': {'description': 'Actual Arrival Time (local time: hhmm)'},
    'arr_time_blk': {'description': 'CRS Arrival Time Block, Hourly Intervals'},
    'cancellation_code': {'description': 'Specifies The Reason For Cancellation'},
    'cancelled': {'description': 'Cancelled Flight Indicator (1=Yes)'},
    'carrier_delay': {'description': 'Carrier Delay, in Minutes'},
    'crs_arr_time': {'description': 'CRS Arrival Time (local time: hhmm)'},
    'crs_dep_time': {'description': 'CRS Departure Time (local time: hhmm)'},
    'crs_elapsed_time': {'description': 'CRS Elapsed Time of Flight, in Minutes'},
    'day_of_month': {'description': 'Day of Month'},
    'day_of_week': {'description': 'Day of Week'},
    'dep_del15': {'description': 'Departure Delay Indicator, 15 Minutes or More (1=Yes)'},
    'dep_delay': {'description': 'Difference in minutes between scheduled and actual departure time. Early departures show negative numbers.'},
    'dep_delay_group': {'description': 'Departure Delay intervals, every (15 minutes from <-15 to >180)'},
    'dep_delay_new': {'description': 'Difference in minutes between scheduled and actual departure time. Early departures set to 0.'},
    'dep_time': {'description': 'Actual Departure Time (local time: hhmm)'},
    'dep_time_blk': {'description': 'CRS Departure Time Block, Hourly Intervals'},
    'dest': {'description': 'Destination Airport'},
    'dest_airport_id': {'description': 'Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.'},
    'dest_airport_seq_id': {'description': 'Destination Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time.'},
    'dest_city_market_id': {'description': 'Destination Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market.'},
    'dest_city_name': {'description': 'Destination Airport, City Name.'},
    'dest_state_abr': {'description': 'Destination Airport, State Abbreviation.'},
    'dest_state_fips': {'description': 'Destination Airport, FIPS code.'},
    'dest_state_nm': {'description': 'Destination Airport, State Number.'},
    'dest_wac': {'description': 'Destination Airport, World Area Code.'},
    'distance': {'description': 'Distance travelled.'},
    'distance_group': {'description': 'Distance Intervals, every 250 Miles, for Flight Segment'},
    'diverted': {'description': 'Indicates if the flight was diverted (1 = Yes).'},
    'first_dep_time': {'description': 'First Gate Departure Time at Origin Airport'},
    'flights': {'description': 'Number of Flights'},
    'fl_date': {'description': 'Flight Date (yyyymmdd)'},
    'late_aircraft_delay': {'description': 'Late Aircraft Delay, in Minutes'},
    'longest_add_gtime': {'description': 'Longest Time Away from Gate for Gate Return or Cancelled Flight'},
    'month': {'description': 'Month'},
    'nas_delay': {'description': 'National Air System Delay, in Minutes'},
    'op_carrier': {'description': 'Commerical Operator.'},
    'op_carrier_airline_id': {'description': 'Commerical Operator, ID'},
    'op_carrier_fl_num': {'description': 'Commerical Operator Flight Number'},
    'op_unique_carrier': {'description': 'Commerical Operator, Unique Carrier Code.'},
    'origin': {'description': 'Origin Airport'},
    'origin_airport_id': {'description': 'Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.'},
    'origin_airport_seq_id': {'description': 'Origin Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time.'},
    'origin_city_market_id': {'description': 'Origin Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market.'},
    'origin_city_name': {'description': 'Origin Airport, City Name'},
    'origin_state_abr': {'description': 'Origin Airport, State Code'},
    'origin_state_fips': {'description': 'Origin Airport, State Fips'},
    'origin_state_nm': {'description': 'Origin Airport, State Name'},
    'origin_wac': {'description': 'Origin Airport, World Area Code'},
    'quarter': {'description': 'Quarter (1-4)'},
    'security_delay': {'description': 'Security Delay, in Minutes'},
    'tail_num': {'description': 'Tail Number'},
    'taxi_in': {'description': 'Taxi In Time, in Minutes'},
    'taxi_out': {'description': 'Taxi Out Time, in Minutes'},
    'total_add_gtime': {'description': 'Total Ground Time Away from Gate for Gate Return or Cancelled Flight'},
    'weather_delay': {'description': 'Weather Delay, in Minutes'},
    'wheels_off': {'description': 'Wheels Off Time (local time: hhmm)'},
    'wheels_on': {'description': 'Wheels On Time (local time: hhmm)'},
    'year': {'description': 'Year'},
 }

html, _ = generate_eda_table(df_flights, 0.001, df_flights_fields)
displayHTML(html)

# COMMAND ----------

junk_features = {
    'cancellation_code',
    'carrier_delay',
    'carrier_delay',
    'div1_airport',
    'div1_airport_id',
    'div1_airport_seq_id',
    'div1_longest_gtime',
    'div1_tail_num',
    'div1_total_gtime',
    'div1_wheels_off',
    'div1_wheels_on',
    'div2_airport',
    'div2_airport_id',
    'div2_airport_seq_id',
    'div2_longest_gtime',
    'div2_tail_num',
    'div2_total_gtime',
    'div2_wheels_off',
    'div2_wheels_on',
    'div3_airport',
    'div3_airport_id',
    'div3_airport_seq_id',
    'div3_longest_gtime',
    'div3_tail_num',
    'div3_total_gtime',
    'div3_wheels_off',
    'div3_wheels_on',
    'div4_airport',
    'div4_airport_id',
    'div4_airport_seq_id',
    'div4_longest_gtime',
    'div4_tail_num',
    'div4_total_gtime',
    'div4_wheels_off',
    'div4_wheels_on',
    'div5_airport',
    'div5_airport_id',
    'div5_airport_seq_id',
    'div5_longest_gtime',
    'div5_tail_num',
    'div5_total_gtime',
    'div5_wheels_off',
    'div5_wheels_on',
    'div_actual_elapsed_time',
    'div_airport_landings',
    'div_arr_delay',
    'div_distance',
    'div_reached_dest',
    'first_dep_time',
    'longest_add_gtime',
    'late_aircraft_delay',
    'nas_delay',
    'security_delay',
    'total_add_gtime',
    'weather_delay'}


df_flights = df_flights.drop(*junk_features)

# COMMAND ----------

pre_filtered_count = df_flights.count()
df_valid_flights = df_flights.where((df_flights.cancelled == 0) & 
                                    (df_flights.diverted == 0) & 
                                    (df_flights.arr_time.isNotNull()) & 
                                    (df_flights.tail_num.isNotNull()))

window = Window.partitionBy('tail_num', 'fl_date', 'origin_airport_id').orderBy('crs_dep_time')

df_valid_flights = df_valid_flights.withColumn('rank', sf.rank().over(window))
df_valid_flights = df_valid_flights.filter('rank = 1')

# df_valid_flights.count()
# df_unique_flights = df_valid_flights.groupby().agg(sf.count('tail_num').alias('count'), 
#                                                                       sf.min('crs_dep_time').alias('crs_dep_time'))
# df_unique_flights = df_unique_flights.filter('count > 1')

# print(df_unique_flights.count())

# df_valid_flights = df_valid_flights.join(df_unique_flights, on=['tail_num', 'fl_date', 'origin_airport_id', 'crs_dep_time'])

filtered_count = df_valid_flights.count()

print(f'The original dataset had {pre_filtered_count:,} records, {filtered_count:,} records remain after filtering.')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Connect flights to weather stations and UTC offsets

# COMMAND ----------

total_flights_count = df_flights.count()
valid_flights_count = df_valid_flights.count()
excluded_flights_count = total_flights_count - valid_flights_count

html = f"{excluded_flights_count:,} flights that were excluded. This will leave us with {valid_flights_count:,} flights that actually flew the scheduled flight plan, representing {1 - excluded_flights_count / total_flights_count:%} of the dataset."

displayHTML(html)

df_valid_flights = df_valid_flights.join(df_closest_airport_station.select(sf.col('iata').alias('origin'), 
                                                                           sf.col('utc_offset').alias('origin_utc_offset'), 
                                                                           sf.col('station_id').alias('origin_station_id')), on='origin')

df_valid_flights = df_valid_flights.join(df_closest_airport_station.select(sf.col('iata').alias('dest'), 
                                                                           sf.col('utc_offset').alias('destination_utc_offset'),
                                                                           sf.col('station_id').alias('destination_station_id')), on='dest')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Augment existing fields

# COMMAND ----------

def utc_time(flight_date, local_time, utc_offset=0):
  flight_date = str(flight_date)
  local_time = str(local_time).zfill(4)
  hour = int(local_time[:2])
  
  # Sometimes the flight data has 2400, which I'm assuming is midnight.
  if hour == 24:
    hour = 0
  
  minute = local_time[2:4]
  dt = datetime.strptime(f'{flight_date} {str(hour).zfill(2)}:{minute}', '%Y-%m-%d %H:%M') + timedelta(hours=utc_offset)
  
  return str(dt)

udf_utc_time = udf(utc_time)

df_valid_flights = df_valid_flights.withColumn('arr_datetime_utc', udf_utc_time('fl_date', 'arr_time', 'origin_utc_offset').cast(TimestampType()))
df_valid_flights = df_valid_flights.withColumn('crs_arr_datetime_utc', udf_utc_time('fl_date', 'crs_arr_time', 'origin_utc_offset').cast(TimestampType()))
df_valid_flights = df_valid_flights.withColumn('dep_datetime_utc', udf_utc_time('fl_date', 'dep_time', 'destination_utc_offset').cast(TimestampType()))
df_valid_flights = df_valid_flights.withColumn('crs_dep_datetime_utc', udf_utc_time('fl_date', 'crs_dep_time', 'destination_utc_offset').cast(TimestampType()))

df_valid_flights = df_valid_flights.withColumn('arr_datetime_local', udf_utc_time('fl_date', 'arr_time').cast(TimestampType()))
df_valid_flights = df_valid_flights.withColumn('crs_arr_datetime_local', udf_utc_time('fl_date', 'crs_arr_time').cast(TimestampType()))
df_valid_flights = df_valid_flights.withColumn('dep_datetime_local', udf_utc_time('fl_date', 'dep_time').cast(TimestampType()))
df_valid_flights = df_valid_flights.withColumn('crs_dep_datetime_local', udf_utc_time('fl_date', 'crs_dep_time').cast(TimestampType()))

# Add percent rank to aid in cross validation/splitting
df_valid_flights = df_valid_flights.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('dep_datetime_utc')))

print(df_valid_flights.count())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Previous Flight Info

# COMMAND ----------

previous_flight_features = {'origin_airport_id', 'dep_delay_new', 'air_time'}
windowSpec = Window.partitionBy('tail_num', sf.year('dep_datetime_utc'), sf.dayofyear('dep_datetime_utc')).orderBy('dep_datetime_utc')

for previous_flight_feature in previous_flight_features:
  df_valid_flights = df_valid_flights.withColumn(f'previous_flight_{previous_flight_feature}', sf.lag(previous_flight_feature, 1).over(windowSpec))
  

# COMMAND ----------

html, _ = generate_eda_table(df_valid_flights, 0.001, df_flights_fields)
displayHTML(html)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Weather Data

# COMMAND ----------

def split_data(row):
  read_datetime = row[1]
  result = [row[0], read_datetime] 
  result = result + row[2] + row[3] + row[4] + row[5] + row[6] + row[7]
  return result

def udf_get_value(null_value):
  def _udf(feature_value, feature_quality):
    return feature_value if feature_quality in ['1', '5'] and float(feature_value) != float(null_value) else None
  return udf(_udf)

if INITIALIZE_DATASETS:
    
  df_weather = spark.read.parquet(f"{root_data_path}/weather_data/*")
  
  df_weather = df_weather.withColumn('id', sf.concat_ws('-', 'station_id', 'read_date'))
  df_weather = df_weather.groupby('id').count()
  df_weather = df_weather.filter('count > 1')  
  
  df_weather = df_weather.withColumn("station_id", sf.col("STATION")).join(df_closest_airport_station, on='station_id').select(
    'STATION', # 0
    'DATE', #1
    sf.split(sf.col('WND'), ','), #2 "wind_direction", "wind_direction_quality", "wind_code", "wind_speed", "wind_speed_quality"
    sf.split(sf.col('CIG'), ','), #3 "ceiling_height", "ceiling_height_quality", "ceiling_height_determination", "ceiling_height_cavok"
    sf.split(sf.col('VIS'), ','), #4 "visibility_distance", "visibility_distance_quality", "visibility_code", "visibility_code_quality"
    sf.split(sf.col('TMP'), ','), #5 "temperature", "temperature_quality"
    sf.split(sf.col('DEW'), ','), #6 "temperature_dewpoint", "temperature_dewpoint_quality"
    sf.split(sf.col('SLP'), ',')  #7 "air_pressure", "air_pressure_quality"
  ).rdd.map(split_data).toDF(schema=['station_id', 'read_date',
                                     "wind_direction", "wind_direction_quality", "wind_code", "wind_speed", "wind_speed_quality",
                                     "ceiling_height", "ceiling_height_quality", "ceiling_height_determination", "ceiling_height_cavok",
                                     "visibility_distance", "visibility_distance_quality", "visibility_code", "visibility_code_quality",
                                     "temperature", "temperature_quality",
                                     "temperature_dewpoint", "temperature_dewpoint_quality",
                                     "air_pressure", "air_pressure_quality"])
  
  df_weather = df_weather.withColumn("wind_direction", udf_get_value('999')("wind_direction", "wind_direction_quality").cast(DoubleType()))
  df_weather = df_weather.withColumn("wind_speed", udf_get_value('9999')("wind_speed", "wind_speed_quality").cast(DoubleType()))
  
  df_weather = df_weather.withColumn("ceiling_height", udf_get_value('99999')("ceiling_height", "ceiling_height_quality").cast(DoubleType()))
  
  df_weather = df_weather.withColumn("visibility_distance", udf_get_value('999999')("visibility_distance", "visibility_distance_quality").cast(DoubleType()))
  
  df_weather = df_weather.withColumn("temperature", udf_get_value('9999')("temperature", "temperature_quality").cast(DoubleType()))
  df_weather = df_weather.withColumn("temperature_dewpoint", udf_get_value('9999')("temperature_dewpoint", "temperature_dewpoint_quality").cast(DoubleType()))
  
  df_weather = df_weather.withColumn("air_pressure", udf_get_value('99999')("air_pressure", "air_pressure_quality").cast(DoubleType()))
  
  df_weather.write.mode('overwrite').parquet(f"{blob_url}/df_weather")

else:
  df_weather = spark.read.parquet(f"{blob_url}/df_weather")


# COMMAND ----------

df_weather_fields = {
  'station_id': {'description': 'The unique identifier of the weather station.'}, 
  'read_date': {'description': 'The datetime of the weather reading.'}, 
  'year': {'description': 'The year of the weather reading.'}, 
  'month': {'description': 'The month of the weather reading.'}, 
  'day': {'description': 'The day of the month of the weather reading.'}, 
  'hour': {'description': 'The hour of the weather reading.'},
  'minute': {'description': 'The minute of the weather reading.'},
  "wind_direction": {'description': 'The direction of the wind, 0 to 360'}, 
  "wind_direction_quality": {'description': 'The quality of the wind direction reading.'}, 
  "wind_code": {'description': ' The code for the type of wind was measured.'}, 
  "wind_speed": {'description': 'The speed of the wind that was measured.'}, 
  "wind_speed_quality": {'description': 'The quality level of the wind speed reading.'},
  "ceiling_height": {'description': 'The height above ground level (AGL) of the lowest cloud or obscuring phenomena layer aloft with 5/8 or more summation total sky cover, which may be predominantly opaque, or the vertical visibility into a surface-based obstruction. Unlimited = 22000'}, 
  "ceiling_height_quality": {'description': 'The code that denotes a quality status of a reported ceiling height dimension.'}, 
  "ceiling_height_determination": {'description': 'The code that denotes the method used to determine the ceiling.'}, 
  "ceiling_height_cavok": {'description': 'The code that represents whether the "Ceiling and Visibility Okay" (CAVOK) condition has been reported.'},
  "visibility_distance": {'description': 'The horizontal distance at which an object can be seen and identified'}, 
  "visibility_distance_quality": {'description': 'The code that denotes a quality status of a reported distance of a visibility observation.'}, 
  "visibility_code": {'description': 'The code that denotes whether or not the reported visibility is variable.'}, 
  "visibility_code_quality": {'description': 'The code that denotes a quality status of a reported VISIBILITY-OBSERVATION variability code'},
  "temperature": {'description': 'The temperature of the air, in Celsius.'},
  "temperature_quality": {'description': 'The code that denotes a quality status of an AIR-TEMPERATURE-OBSERVATION'},
  "temperature_dewpoint": {'description': 'The temperature to which a given parcel of air must be cooled at constant pressure and water vapor content in order for saturation to occur'},
  "temperature_dewpoint_quality": {'description': 'The code that denotes a quality status of the reported dew point temperature.'},
  "air_pressure": {'description': 'The air pressure relative to Mean Sea Level (MSL).'},
  "air_pressure_quality": {'description': 'The code that denotes a quality status of the sea level pressure.'}
}

# print(df_weather.count())

html, _ = generate_eda_table(df_weather, 0.002, df_weather_fields)
displayHTML(html)

# COMMAND ----------

spark.read.parquet(f"{root_data_path}/weather_data/*").count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Joining Flights to Weather Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Summarize Weather Data

# COMMAND ----------

numeric_weather_features = ['wind_direction', 
                            'wind_speed', 
                            'ceiling_height', 
                            'visibility_distance', 
                            'temperature', 
                            'temperature_dewpoint', 
                            'air_pressure']

expressions = list()
for numeric_weather_feature in numeric_weather_features:  
  expressions = expressions + [sf.mean(numeric_weather_feature).alias(f'{numeric_weather_feature}_mean'),
                               sf.min(numeric_weather_feature).alias(f'{numeric_weather_feature}_min'), 
                               sf.max(numeric_weather_feature).alias(f'{numeric_weather_feature}_max')]

seconds_window = sf.from_unixtime(sf.unix_timestamp('read_date') - sf.unix_timestamp('read_date') % WEATHER_AGGREGATE_WINDOW_SECONDS)
df_weather_summary = df_weather.withColumn('aggregated_datetime', 
                                           seconds_window.cast(TimestampType())).groupBy('station_id', 
                                                                                         'aggregated_datetime').agg(*expressions)

numeric_weather_agg_features = [f'{feature}_{func}' for feature in numeric_weather_features for func in ['mean', 'min', 'max']]
imputer = Imputer(inputCols=numeric_weather_agg_features, outputCols=numeric_weather_agg_features).setStrategy("mean")
df_weather_summary = imputer.fit(df_weather_summary).transform(df_weather_summary)

df_weather_summary = df_weather_summary.cache()

html, _ = generate_eda_table(df_weather_summary, 0.002, df_weather_fields)
displayHTML(html)

# COMMAND ----------

df_weather_summary.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Connect Weather Data to Flight Data

# COMMAND ----------

seconds_window = sf.from_unixtime(sf.unix_timestamp('crs_dep_datetime_utc') - sf.unix_timestamp('crs_dep_datetime_utc') % WEATHER_AGGREGATE_WINDOW_SECONDS)

df_joined = df_valid_flights.withColumn('aggregated_datetime', seconds_window.cast(TimestampType()) - sf.expr("INTERVAL 2 HOURS")).join(
  df_weather_summary.withColumn('origin_station_id', sf.col('station_id')),
  on=['origin_station_id', 'aggregated_datetime'], how='left'
)

df_joined = df_joined.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Calculate the Log10 of all numerical features

# COMMAND ----------

numerical_features = [column_name for column_name, column_type in df_joined.dtypes if str(column_type) == 'double']

for numerical_feature in numerical_features:
  df_joined = df_joined.withColumn(f'{numerical_feature}_log', sf.log10(numerical_feature))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Final Dataset

# COMMAND ----------

if INITIALIZE_DATASETS:
  df_joined.write.mode('overwrite').parquet(f"{blob_url}/df_joined")
else:
  print('loading from storage...')
  df_joined = spark.read.parquet(f'{blob_url}/df_joined/*')

# COMMAND ----------

df_joined.count()

# COMMAND ----------

html, _ = generate_eda_table(df_joined, 0.001, {})
displayHTML(html)

# COMMAND ----------

# df_joined_sample = df_joined.sample(fraction=0.1).toPandas()

features = {
  'actual_elapsed_time',
  'actual_elapsed_time_log',
#   'aggregated_datetime',
  'air_pressure_max',
  'air_pressure_max_log',
  'air_pressure_mean',
  'air_pressure_mean_log',
  'air_pressure_min',
  'air_pressure_min_log',
  'air_time',
  'air_time_log',
  'arr_datetime_local',
  'arr_datetime_utc',
  'arr_del15',
  'arr_del15_log',
  'arr_delay',
#   'arr_delay_group',
  'arr_delay_log',
  'arr_delay_new',
  'arr_delay_new_log',
#   'arr_time',
#   'arr_time_blk',
#   'cancelled',
#   'cancelled_log',
  'ceiling_height_max',
  'ceiling_height_max_log',
  'ceiling_height_mean',
  'ceiling_height_mean_log',
  'ceiling_height_min',
  'ceiling_height_min_log',
  'crs_arr_datetime_local',
  'crs_arr_datetime_utc',
#   'crs_arr_time',
  'crs_dep_datetime_local',
  'crs_dep_datetime_utc',
#   'crs_dep_time',
  'crs_elapsed_time',
  'crs_elapsed_time_log',
  'day_of_month',
  'day_of_week',
  'dep_datetime_local',
  'dep_datetime_utc',
  'dep_del15',
#   'dep_del15_log',
#   'dep_delay',
#   'dep_delay_group',
#   'dep_delay_log',
#   'dep_delay_new',
#   'dep_delay_new_log',
#   'dep_time',
#   'dep_time_blk',
  'dest',
  'dest_airport_id',
#   'dest_airport_seq_id',
  'dest_city_market_id',
#   'dest_city_name',
#   'dest_state_abr',
#   'dest_state_fips',
#   'dest_state_nm',
#   'dest_wac',
#   'destination_station_id',
#   'destination_utc_offset',
  'distance',
#   'distance_group',
  'distance_log',
#   'diverted',
#   'diverted_log',
#   'fl_date',
#   'flights',
#   'flights_log',
#   'late_aircraft_delay',
#   'late_aircraft_delay_log',
  'month',
#   'op_carrier',
  'op_carrier_airline_id',
#   'op_carrier_fl_num',
  'op_unique_carrier',
  'origin',
  'origin_airport_id',
#   'origin_airport_seq_id',
  'origin_city_market_id',
#   'origin_city_name',
#   'origin_state_abr',
#   'origin_state_fips',
#   'origin_state_nm',
#   'origin_station_id',
#   'origin_utc_offset',
#   'origin_wac',
  'quarter',
  'rank',
#   'rank_log',
#   'station_id',
#   'tail_num',
  'taxi_in',
  'taxi_in_log',
  'taxi_out',
  'taxi_out_log',
  'temperature_dewpoint_max',
  'temperature_dewpoint_max_log',
  'temperature_dewpoint_mean',
  'temperature_dewpoint_mean_log',
  'temperature_dewpoint_min',
  'temperature_dewpoint_min_log',
  'temperature_max',
  'temperature_max_log',
  'temperature_mean',
  'temperature_mean_log',
  'temperature_min',
  'temperature_min_log',
  'visibility_distance_max',
  'visibility_distance_max_log',
  'visibility_distance_mean',
  'visibility_distance_mean_log',
  'visibility_distance_min',
  'visibility_distance_min_log',
#   'wheels_off',
#   'wheels_on',
  'wind_direction_max',
  'wind_direction_max_log',
  'wind_direction_mean',
  'wind_direction_mean_log',
  'wind_direction_min',
  'wind_direction_min_log',
  'wind_speed_max',
  'wind_speed_max_log',
  'wind_speed_mean',
  'wind_speed_mean_log',
  'wind_speed_min',
  'wind_speed_min_log',
  'year'
}

# COMMAND ----------

# df_joined_sample = df_joined_sample[features]

# COMMAND ----------

# MAGIC %md 
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Day of Year

# COMMAND ----------

df_joined = df_joined.withColumn('dep_day_of_year', sf.dayofyear('dep_datetime_local'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Minute of Day

# COMMAND ----------

df_joined = df_joined.withColumn("dep_minute_of_day", sf.hour('dep_datetime_local') * 60 + sf.minute('dep_datetime_local'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Previous Flight Delay

# COMMAND ----------

df_joined.withColumn('dep_day_of_year', sf.dayofyear('dep_datetime_local'))

# COMMAND ----------

html, df_joined_sample = generate_eda_table(df_joined, 0.001, df_flights_fields)
displayHTML(html)

df_joined_sample.hist(figsize=(35,35), bins=15)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Feature Selection

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(50,50))         # Sample figsize in inches

corrMatrix = df_joined_sample.corr()
sns.heatmap(corrMatrix, annot=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Cross Validation
# MAGIC 
# MAGIC https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9

# COMMAND ----------

# df_joined = spark.read.parquet(f'{blob_url}/df_joined/')
#Starting with a simple implementation that will simply split train/val/test by the rank (percentile). 
# We must cache the dataframe at this point so that our random split on the 30% of test data (into 15% validation / 15% test) does not change upon rerun (non-deterministic). 
df_joined_pre2019 = df_joined.where("year < 2019").cache()
#Check that we don't have any 2019+ data
df_joined_pre2019.where("year >= 2019").show()


# COMMAND ----------

#Allocate 70% of pre-2019 data to training set. 
train_df = df_joined_pre2019.where("rank <= .7").cache()
train_df.head()

# COMMAND ----------

#Allocate 70% of pre-2019 data to training set. 
#Again we cache to maintain consistency on re-run. 
test_val_df = df_joined_pre2019.where("rank > .7").cache()

#Now we split the test_val_df into validation and test sets using a 50% / 50% random split. 
val_df, test_df = test_val_df.randomSplit([0.5, 0.5], seed = 1234)

# COMMAND ----------

count_df_joined = df_joined.count()
count_df_joined_pre2019 = df_joined_pre2019.count()
count_train_df = train_df.count()
count_test_val_df = test_val_df.count()
count_val_df = val_df.count()
count_test_df = test_df.count()

# COMMAND ----------

print(f'The number of rows in the df_joined: {count_df_joined}')
print(f'The number of rows in the df_joined_pre2019: {count_df_joined_pre2019}')
print(f'The number of rows in the train_df: {count_train_df}, percent: {count_train_df/count_df_joined_pre2019}')
print(f'The number of rows in the test_val_df: {count_test_val_df}, percent: {count_test_val_df/count_df_joined_pre2019}')
print(f'The number of rows in the val_df: {count_val_df}, percent: {count_val_df / count_df_joined_pre2019}')
print(f'The number of rows in the test_df: {count_test_df}, percent: {count_test_df / count_df_joined_pre2019}')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Baseline Model

# COMMAND ----------

# logistic regression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#df_joined = spark.read.parquet(f'{blob_url}/df_joined/')
#train, test = df_joined.randomSplit([.8,.2], seed=42)

# clean up null values from train and test
train = train_df.dropna(how='any', subset=['dep_del15','year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean'])
val = val_df.dropna(how='any', subset = ['dep_del15','year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean'])

# combine all features into single vector collumn
assembler = VectorAssembler().setInputCols(['year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean']).setOutputCol('features')
# apply vector assembler to train and test
assembler_train = assembler.transform(train)
assembler_val = assembler.transform(val)

# initiallize model
model = LogisticRegression(featuresCol = 'features', labelCol='dep_del15', maxIter=5)

# fit model
lr_model = model.fit(assembler_train)

# generate predictions
predictions = lr_model.transform(assembler_val)

# evaluate predictions

evaluator = MulticlassClassificationEvaluator(labelCol='dep_del15', predictionCol='prediction')
precision = evaluator.evaluate(predictions, {evaluator.metricName: 'weightedPrecision'})
print(f'Precision score: {precision}')

# COMMAND ----------

# decision tree
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#df_joined = spark.read.parquet(f'{blob_url}/df_joined/')
#train, test = df_joined.randomSplit([.8,.2], seed=42)

# clean up null values from train and test
train = train_df.dropna(how='any', subset=['dep_del15','year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean'])
val = val_df.dropna(how='any', subset = ['dep_del15','year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean'])

# combine all features into single vector collumn
assembler = VectorAssembler().setInputCols(['year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean']).setOutputCol('features')
# apply vector assembler to train and test
assembler_train = assembler.transform(train)
assembler_val = assembler.transform(val)

# initiallize model
model = DecisionTreeClassifier(featuresCol = 'features', labelCol='dep_del15', maxDepth=5)

# fit model
dt_model = model.fit(assembler_train)

# generate predictions
predictions = dt_model.transform(assembler_val)

# evaluate predictions

evaluator = MulticlassClassificationEvaluator(labelCol='dep_del15', predictionCol='prediction')
precision = evaluator.evaluate(predictions, {evaluator.metricName: 'weightedPrecision'})
print(f'Precision score: {precision}')

# COMMAND ----------

# random forest
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#df_joined = spark.read.parquet(f'{blob_url}/df_joined/')
#train, test = df_joined.randomSplit([.8,.2], seed=42)

# clean up null values from train and test
train = train_df.dropna(how='any', subset=['dep_del15','year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean'])
val = val_df.dropna(how='any', subset = ['dep_del15','year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean'])

# combine all features into single vector collumn
assembler = VectorAssembler().setInputCols(['year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean']).setOutputCol('features')
# apply vector assembler to train and test
assembler_train = assembler.transform(train)
assembler_val = assembler.transform(val)

# initiallize model
model = RandomForestClassifier(featuresCol = 'features', labelCol='dep_del15', numTrees=10)

# fit model
rf_model = model.fit(assembler_train)

# generate predictions
predictions = rf_model.transform(assembler_val)

# evaluate predictions

evaluator = MulticlassClassificationEvaluator(labelCol='dep_del15', predictionCol='prediction')
precision = evaluator.evaluate(predictions, {evaluator.metricName: 'weightedPrecision'})
print(f'Precision score: {precision}')

# COMMAND ----------

# linear svm
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#df_joined = spark.read.parquet(f'{blob_url}/df_joined/')
#train, test = df_joined.randomSplit([.8,.2], seed=42)

# clean up null values from train and test
train = train_df.dropna(how='any', subset=['dep_del15','year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean'])
val = val_df.dropna(how='any', subset = ['dep_del15','year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean'])

# combine all features into single vector collumn
assembler = VectorAssembler().setInputCols(['year', 'quarter', 'month', 'day_of_month', 'day_of_week', 'op_carrier_airline_id', 'origin_airport_id', 'dest_airport_id', 'dep_time', 'air_time', 'distance', 'previous_flight_actual_elapsed_time', 'previous_flight_dep_delay_new', 'previous_flight_origin_airport_id', 'previous_flight_dep_time', 'wind_direction_mean', 'wind_speed_mean', 'visibility_distance_mean', 'temperature_mean', 'air_pressure_mean', 'temperature_dewpoint_mean']).setOutputCol('features')
# apply vector assembler to train and test
assembler_train = assembler.transform(train)
assembler_val = assembler.transform(val)

# initiallize model
model = LinearSVC(featuresCol = 'features', labelCol='dep_del15', maxIter=10, regParam=0.1)

# fit model
svc_model = model.fit(assembler_train)

# generate predictions
predictions = svc_model.transform(assembler_val)

# evaluate predictions

evaluator = MulticlassClassificationEvaluator(labelCol='dep_del15', predictionCol='prediction')
precision = evaluator.evaluate(predictions, {evaluator.metricName: 'weightedPrecision'})
print(f'Precision score: {precision}')

# COMMAND ----------


