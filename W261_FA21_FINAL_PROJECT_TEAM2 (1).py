# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Team 2
# MAGIC 
# MAGIC David Ristau, Jerico Johns, Josh Jonte, Mohamed Gesalla

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Presentation	
# MAGIC - 15 minutes presenting, 5 minutes for Q&A. All team members should speak. You should cover the following points, but try to focus on the most interesting aspects of the project.
# MAGIC   - Introduce the business case
# MAGIC   - Introduce the dataset
# MAGIC   - Summarize EDA and feature engineering
# MAGIC   - Summarize algorithms tried, and justify final algorithm choice
# MAGIC   - Discuss evaluation metrics in light of the business case
# MAGIC   - Discuss performance and scalability concerns

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
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

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
RENDER_EDA_TABLES = False
WEATHER_AGGREGATE_WINDOW_SECONDS = 60 * 30 # 30 minutes


def generate_eda_table(df_spark, sample_fraction=0.1, fields={}):
  
    total_row_count = df_spark.count()
    df_pandas = df_spark.sample(fraction=sample_fraction).toPandas()
    sample_row_count = len(df_pandas)
    
    column_table = [
      '<table border="1"><thead>'
      f'<tr><td>Sample #</td><td colspan=8>{sample_row_count:,} / {total_row_count:,} ({sample_row_count / total_row_count:%})</td></tr>'
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

# MAGIC %md
# MAGIC # Question Formulation
# MAGIC 
# MAGIC You should refine the question formulation based on the general task description you’ve been given, ie, predicting flight delays. This should include some discussion of why this is an important task from a business perspective, who the stakeholders are, etcz. Some literature review will be helpful to figure out how this problem is being solved now, and the State Of The Art (SOTA) in this domain. Introduce the goal of your analysis. What questions will you seek to answer, why do people perform this kind of analysis on this kind of data? Preview what level of performance your model would need to achieve to be practically useful. Discuss evaluation metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Business Case
# MAGIC 
# MAGIC Business case goes here

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Stakeholders
# MAGIC 
# MAGIC Stakeholders case go here

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Reviewed Literature
# MAGIC 
# MAGIC Goes here

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Goals
# MAGIC 
# MAGIC Goes here

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Questions to be answered

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Performance & Evaluation Metrics
# MAGIC 
# MAGIC Goes here

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # EDA & Discussion of Challenges
# MAGIC 
# MAGIC Determine a handful of relevant EDA tasks that will help you make decisions about how you implement the algorithm to be scalable. Discuss any challenges that you anticipate based on the EDA you perform.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Airport Data

# COMMAND ----------

if INITIALIZE_DATASETS:
  df_airports = spark.createDataFrame(pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat', 
                                                  header=0, names=['of_id', 'name', 'city', 'country', 'iata', 'icao', 'lat', 'lon', 
                                                                   'altitude', 'utc_offset', 'dst', 'timezone', 'type', 'source']))
  df_airports.write.mode('overwrite').parquet(f"{blob_url}/df_airports")
else:
  df_airports = spark.read.parquet(f'{blob_url}/df_airports/')

df_airports = df_airports.select('name', 'iata', 'icao', 'lat', 'lon', 'timezone', sf.col('utc_offset').cast(IntegerType())) \
                         .filter((df_airports.country == 'United States') & (df_airports.type == 'airport') & (df_airports.utc_offset.isNotNull()))

if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_airports, sample_fraction=1.0)
  displayHTML(html)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Weather Station Data

# COMMAND ----------

df_stations = spark.read.parquet(f"{root_data_path}/stations_data/*")

if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_stations, sample_fraction=0.01)
  displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Joining Airports to Weather Stations
# MAGIC There's a lot of weather stations and a ton of unnecessary data in the weather dataset.

# COMMAND ----------

@sf.udf
def udf_get_distance(lat_a, lon_a, lat_b, lon_b):
  lat_a, lon_a, lat_b, lon_b = map(radians, [lat_a, lon_a, lat_b, lon_b])
  dist_lon = lon_b - lon_a
  dist_lat = lat_b - lat_a

  area = sin(dist_lat/2)**2 + cos(lat_a) * cos(lat_b) * sin(dist_lon / 2)**2
 
  central_angle = 2 * asin(sqrt(area))
  radius = 6371

  distance = central_angle * radius
  
  return abs(distance)

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
  
df_closest_airport_station.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Historical Flight Data

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

df_flights = spark.read.parquet(f'{root_data_path}/parquet_airlines_data/*')
df_flights = df_flights.toDF(*[c.lower() for c in df_flights.columns])

if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_flights, 0.001, df_flights_fields)
  displayHTML(html)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Remove Invalid & Duplicate Flights

# COMMAND ----------

pre_filtered_count = df_flights.count()
df_valid_flights = df_flights.where((df_flights.cancelled == 0) & # flight hasn't been cancelled
                                    (df_flights.diverted == 0) & # flight hasn't been diverted
                                    (df_flights.arr_time.isNotNull()) & # flight has an arrival time 
                                    (df_flights.tail_num.isNotNull()) & # flight has a tail number
                                    (df_flights.air_time > 30) & # flight was in the air more than 30 minutes
                                    (df_flights.dep_delay.isNotNull()) &
                                    (df_flights.arr_delay.isNotNull())
                                   )

window = Window.partitionBy('tail_num', 'fl_date', 'origin_airport_id').orderBy(sf.col('crs_dep_time').desc())
df_valid_flights = df_valid_flights.withColumn('rank', sf.rank().over(window).cast(IntegerType())).filter(sf.col('rank') == 1).drop('rank')

filtered_count = df_valid_flights.count()

print(f'The original dataset had {pre_filtered_count:,} records, {filtered_count:,} records remain after filtering.')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Join Flight Data to Weather Station Data

# COMMAND ----------

df_valid_flights = df_valid_flights.join(df_closest_airport_station.select(sf.col('iata').alias('origin'), 
                                                                           sf.col('utc_offset').alias('origin_utc_offset'), 
                                                                           sf.col('station_id').alias('origin_station_id')), on='origin')

df_valid_flights = df_valid_flights.join(df_closest_airport_station.select(sf.col('iata').alias('dest'), 
                                                                           sf.col('utc_offset').alias('destination_utc_offset'),
                                                                           sf.col('station_id').alias('destination_station_id')), on='dest')

# COMMAND ----------

@sf.udf
def udf_utc_time(flight_date, local_time, utc_offset=0):
  flight_date = str(flight_date)
  local_time = str(local_time).zfill(4)
  hour = int(local_time[:2])
  
  # Sometimes the flight data has 2400, which I'm assuming is midnight.
  if hour == 24:
    hour = 0
  
  minute = local_time[2:4]
  dt = datetime.strptime(f'{flight_date} {str(hour).zfill(2)}:{minute}', '%Y-%m-%d %H:%M') + timedelta(hours=utc_offset)
  
  return str(dt)

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

# print(df_valid_flights.count())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Previous Flight Data

# COMMAND ----------

previous_flight_features = {'origin_airport_id', 'dep_delay_new', 'crs_elapsed_time', 'dep_del15', 'crs_dep_datetime_utc', 'dep_datetime_utc'}
windowSpec = Window.partitionBy('tail_num', 'fl_date').orderBy(sf.col('dep_datetime_utc').desc())

for previous_flight_feature in previous_flight_features:
  df_valid_flights = df_valid_flights.withColumn(f'previous_flight_{previous_flight_feature}', sf.lag(previous_flight_feature, 1).over(window))

# COMMAND ----------

if RENDER_EDA_TABLES:
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
  
#   df_weather = df_weather.withColumn('id', sf.concat_ws('-', 'station_id', 'read_date'))
#   df_weather = df_weather.groupby('id').count()
#   df_weather = df_weather.filter('count > 1')  
  
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

if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_weather, 0.002, df_weather_fields)
  displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Weather Data Aggregation

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

# numeric_weather_agg_features = [f'{feature}_{func}' for feature in numeric_weather_features for func in ['mean', 'min', 'max']]
# imputer = Imputer(inputCols=numeric_weather_agg_features, outputCols=numeric_weather_agg_features).setStrategy("mean")
# df_weather_summary = imputer.fit(df_weather_summary).transform(df_weather_summary)

df_weather_summary = df_weather_summary.cache()


if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_weather_summary, 0.002, df_weather_fields)
  displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Join Flight Data to Weather Data

# COMMAND ----------

seconds_window = sf.from_unixtime(sf.unix_timestamp('crs_dep_datetime_utc') - sf.unix_timestamp('crs_dep_datetime_utc') % WEATHER_AGGREGATE_WINDOW_SECONDS)

df_joined = df_valid_flights.withColumn('aggregated_datetime', seconds_window.cast(TimestampType()) - sf.expr("INTERVAL 2 HOURS")).join(
  df_weather_summary.withColumn('origin_station_id', sf.col('station_id')),
  on=['origin_station_id', 'aggregated_datetime'], how='left'
)

# COMMAND ----------

junk_features = {
    'cancellation_code',
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
    'weather_delay',
    'station_id'
}


df_joined = df_joined.drop(*junk_features)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Balance the Dataset

# COMMAND ----------

ALPHA = 0.9

positive_sample_count = df_joined.filter('dep_del15 == 1').count()
negative_sample_count = df_joined.filter('dep_del15 == 0').count()

df_positive_sample = df_joined.filter('dep_del15 == 1')
df_negative_sample = df_joined.filter('dep_del15 == 0').sample(False, positive_sample_count / (negative_sample_count * ALPHA))

df_joined = df_negative_sample.union(df_positive_sample)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Impute missing weather features

# COMMAND ----------

numeric_weather_agg_features = [f'{feature}_{func}' for feature in numeric_weather_features for func in ['mean', 'min', 'max']]
imputer = Imputer(inputCols=numeric_weather_agg_features, outputCols=numeric_weather_agg_features).setStrategy("mean")

df_joined = imputer.fit(df_joined).transform(df_joined)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Persist & Load

# COMMAND ----------

if INITIALIZE_DATASETS:
  df_joined.write.mode('overwrite').parquet(f"{blob_url}/df_joined")

df_joined = spark.read.parquet(f'{blob_url}/df_joined/*')

if RENDER_EDA_TABLES:
  html, df_joined_sample = generate_eda_table(df_joined, 0.001, {})
  displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Feature Engineering
# MAGIC Apply relevant feature transformations, dimensionality reduction if needed, interaction terms, treatment of categorical variables, etc.. Justify your choices.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Day of Year

# COMMAND ----------

df_joined = df_joined.withColumn('dep_day_of_year', sf.dayofyear('dep_datetime_local').cast(DoubleType()))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Minute of Day

# COMMAND ----------

df_joined = df_joined.withColumn("dep_minute_of_day", (sf.hour('dep_datetime_local') * 60 + sf.minute('dep_datetime_local')).cast(DoubleType()))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Previous Flight Delayed

# COMMAND ----------

@sf.udf
def udf_calculate_previous_flight_delay(previous_scheduled_elapsed_time, previous_dep_delay_new):
  if previous_scheduled_elapsed_time is not None and previous_scheduled_elapsed_time > 120:
    return previous_dep_delay_new
  else:
    return 0
    

df_joined = df_joined.withColumn('previous_flight_dep_delay_new_2', 
                                  udf_calculate_previous_flight_delay('previous_flight_crs_elapsed_time', 'previous_flight_dep_delay_new').cast(DoubleType()))
  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Handle NULL values

# COMMAND ----------

zero_fills = ['previous_flight_crs_elapsed_time', 'previous_flight_dep_delay_new']

df_joined = df_joined.na.fill(value=0, subset=zero_fills)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Log10 

# COMMAND ----------

numerical_features = [column_name for column_name, column_type in df_joined.dtypes 
                      if str(column_type) == 'double' or str(column_type) == 'integer']

for numerical_feature in numerical_features:
  df_joined = df_joined.withColumn(f'{numerical_feature}_log', sf.log10(numerical_feature))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Load or Persist Final Dataset

# COMMAND ----------

# print(df_joined.count())
# print(df_joined.take(5))

if INITIALIZE_DATASETS:
  df_joined.write.mode('overwrite').parquet(f"{blob_url}/df_joined_final")

print('loading from storage...')
df_joined = spark.read.parquet(f'{blob_url}/df_joined_final/*')
  
if RENDER_EDA_TABLES:
  
  html, df_joined_sample = generate_eda_table(df_joined, 0.001, {})
  displayHTML(html)

  df_joined_sample.hist(figsize=(35,35), bins=15)
  plt.show()    

# COMMAND ----------

print(df_joined.count())

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Feature Selection

# COMMAND ----------

target_feature = 'dep_del15'

categorical_features = list({'dest',
                        'origin',
                        'op_unique_carrier',
                        'month',
                        'day_of_week',
                        'previous_flight_origin_airport_id'})

continuous_features = list({'day_of_month',
                      'dep_minute_of_day',
                      'dep_day_of_year_log',
                      'taxi_out_log',
                      'taxi_in_log',
                      'crs_elapsed_time_log',
                      'distance_log',
                      'previous_flight_crs_elapsed_time_log',
                      'wind_speed_mean_log',
                      'wind_speed_max_log',
                      'ceiling_height_mean_log',
                      'visibility_distance_mean_log',
                      'visibility_distance_max_log',
                      'temperature_mean_log',
                      'temperature_dewpoint_mean_log',
                      'air_pressure_mean_log',
                      'previous_flight_dep_delay_new_2_log'})

all_features = categorical_features + continuous_features
all_features.append(target_feature)

df_raw_features = df_joined.select(*all_features)

df_raw_features = df_raw_features.na.fill(value=-1, subset=['previous_flight_origin_airport_id'])
df_raw_features = df_raw_features.na.fill(value=0, subset=['previous_flight_dep_delay_new_2_log'])

imputer = Imputer(inputCols=continuous_features, outputCols=continuous_features).setStrategy("mean")
df_raw_features = imputer.fit(df_raw_features).transform(df_raw_features)

if True: #RENDER_EDA_TABLES:
  
  html, df_raw_features_sample = generate_eda_table(df_raw_features, 0.001, {})
  displayHTML(html)

  df_raw_features_sample.hist(figsize=(35,35), bins=15)
  plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Build Pipeline

# COMMAND ----------

# from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
# from pyspark.ml.feature import RobustScaler

pipeline_steps = list()

target_indexer = StringIndexer(inputCol=target_feature, outputCol=f'{target_feature}_index')
pipeline_steps += [target_indexer]

for feature in categorical_features:
  indexer = StringIndexer(inputCol=feature, outputCol=f'{feature}_index')
  indexer.setHandleInvalid('skip')
  encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()], outputCols=[f'{feature}_oh'])
  pipeline_steps += [indexer, encoder]

assembler = VectorAssembler(inputCols=continuous_features, outputCol='vectors')
assembler.setHandleInvalid('skip')
scaler = StandardScaler(inputCol='vectors', 
                        outputCol='scaled_vectors')

pipeline_steps += [assembler, scaler]

assemblerInputs = [f'{feature}_oh' for feature in categorical_features] + ['scaled_vectors']
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol='features')
pipeline_steps += [assembler]

train_model = Pipeline(stages=pipeline_steps).fit(df_raw_features).transform(df_raw_features)

train_model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Algorithm Exploration
# MAGIC Apply 2 to 3 algorithms to the training set, and discuss expectations, trade-offs, and results. These will serve as your baselines - do not spend too much time fine tuning these. You will want to use this process to select a final algorithm which you will spend your efforts on fine tuning.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Algorithm Implementation
# MAGIC Create your own toy example that matches the dataset provided and use this toy example to explain the math behind the algorithm that you will perform. Apply your algorithm to the training dataset and evaluate your results on the test set. 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Conclusions 
# MAGIC Report results and learnings for both the ML as well as the scalability.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Application of Course Concepts
# MAGIC Pick 3-5 key course concepts and discuss how your work on this assignment illustrates an understanding of these concepts.

# COMMAND ----------


