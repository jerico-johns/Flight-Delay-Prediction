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
from pyspark.sql.functions import col,lit
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
RENDER_EDA_TABLES = True
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
      is_numeric = column_type == 'DoubleType' or column_type == 'IntegerType'
      
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
# MAGIC You should refine the question formulation based on the general task description you???ve been given, ie, predicting flight delays. This should include some discussion of why this is an important task from a business perspective, who the stakeholders are, etcz. Some literature review will be helpful to figure out how this problem is being solved now, and the State Of The Art (SOTA) in this domain. Introduce the goal of your analysis. What questions will you seek to answer, why do people perform this kind of analysis on this kind of data? Preview what level of performance your model would need to achieve to be practically useful. Discuss evaluation metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Business Case
# MAGIC 
# MAGIC In business, time is money and unexpected delays mean unexpected costs. CAL Airlines & Freight  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Stakeholders
# MAGIC 
# MAGIC In business, time is money and unexpected delays 

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
# MAGIC [Source](https://machinelearningmastery.com/fbeta-measure-for-machine-learning/)
# MAGIC 
# MAGIC The buisness case for this project dictates that we should emphasize the avoidance of predicting a flight to be delayed when it is on time over predicting a flight is not delayed when it is. This rationale is supported by the theory that a pasenger that is notified a flight is delayed when it is not may miss their flight. Inversely, if a passanger is not notified of a delay when there is one, the outcome is added idle time for the passenger. Our team is operating under the belief that the latter is a prefered failure mode over the prior. 
# MAGIC 
# MAGIC This translates into an emphasized adversion to false positives over false negatives. However, we intend to build a prediction model to perform well in both contexts. As such we will use f-beta as our primary evaluation metric. F-beta is similar to F1-score, in that it takes into account both the recall and precision of the model, but f-beta takes a parameter, beta, to tune the sensitivity towards either precision or recall. The equaiton for f-beta is shown below:
# MAGIC 
# MAGIC \\( FBeta = \frac{(1+ \beta^2)\bullet Precision \bullet Recall}{\beta^2 \bullet Precision \bullet Recall}\\)
# MAGIC 
# MAGIC \\(\beta\\) 
# MAGIC is a parameter to be specified based on which is more important to the application, precision or recall. Commonly, if precision is to be emphasized then \\(\beta = 0.5\\) and if recall is to be emphasized \\(\beta = 2.0\\).
# MAGIC 
# MAGIC The equations for precision and recall are shown below, where FP = false positives, FN = false negatives, TP = true positives, and TN = true negatives.
# MAGIC 
# MAGIC \\(Precision = \frac{TP}{TP + FP} \\)  
# MAGIC 
# MAGIC \\(Recall = \frac{TP}{TP + FN}\\)
# MAGIC 
# MAGIC 
# MAGIC In this context we are more adverse to false positives. Given the equations for precision and recall above, it is apparent that Precision must be emphasized in this context. Our implementation of F-Beta will therefore use \\(\beta = 0.5\\).
# MAGIC 
# MAGIC 
# MAGIC Our Implemenation for F-Beta is in the cell below:

# COMMAND ----------

def f_beta(prediction_df, beta = 0.5):
  """ 
  F-Beta implementation with beta = 0.5 to emphasize precision (avoid false positives)
  
  prediction_df - dataframe with predictions generated in 'prediction' and ground truth in 'dep_del15'
  
  f_beta - float value of f_beta score
  
  """

  # True Positives:
  tp = prediction_df.where('dep_del15 = 1.0').where('prediction = 1.0').count()
  # False Positives:
  fp = prediction_df.where('dep_del15 = 0.0').where('prediction = 1.0').count()
  # True Negatives:
  tn = prediction_df.where('dep_del15 = 0.0').where('prediction = 0.0').count()
  # False Negatives:
  fn = prediction_df.where('dep_del15 = 1.0').where('prediction = 0.0').count()
  # Precision:
  precision = tp / (tp + fp)
  # Recall:
  recall = tp / (tp + fn)
  # F-Beta:
  f_beta = (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))
  return f_beta

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # EDA & Discussion of Challenges

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Airport Data
# MAGIC 
# MAGIC The lat/lon locations and timezone offsets for airports is downloaded from OpenFlights.org, the entire dataset is 7,697 rows. The historical flight data only contains domestic flights so we only keep airports that are in the US and that have a UTC offset, this reduces the dataset down to 1,512 rows.

# COMMAND ----------

if INITIALIZE_DATASETS:
  df_airports = spark.createDataFrame(pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat', 
                                                  header=0, names=['of_id', 'name', 'city', 'country', 'iata', 'icao', 'lat', 'lon', 
                                                                   'altitude', 'utc_offset', 'dst', 'timezone', 'type', 'source']))
  df_airports.write.mode('overwrite').parquet(f"{blob_url}/df_airports")
else:
  df_airports = spark.read.parquet(f'{blob_url}/df_airports/')

df_airports = df_airports.select('name', 'iata', 'icao', 'lat', 'lon', 'timezone', sf.col('utc_offset').cast(IntegerType())) \
                         .filter((df_airports.country == 'United States') & 
                                 (df_airports.type == 'airport') & 
                                 (df_airports.utc_offset.isNotNull()))

if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_airports, sample_fraction=1.0)
  displayHTML(html)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Weather Station Data
# MAGIC 
# MAGIC The Weather Station dataset contains data about 5,004,169 NOAA weather stations.

# COMMAND ----------

df_stations = spark.read.parquet(f"{root_data_path}/stations_data/*")

if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_stations, sample_fraction=0.01)
  displayHTML(html)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Joining Airports to Weather Stations
# MAGIC 
# MAGIC We are only interested in the weather stations closest to the airports, so we are able to eliminate 99.9742% of the weather stations by joining it to the Airport dataset, so there's only 1,291 rows. The join is more complex than a simple key-based join, we use the distance of the weather station from the airport as the predicate for selecting which weather station is the closest. If an airport does not have a weather station within 5 kilometers then the airport is eliminated from the list of candidate airports because weather is hyperlocal and anything more than 5 kilometers aways will have a minimal effect on the airport's departure. 

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
  
  dbutils.fs.rm(f"{blob_url}/df_closest_airport_station", True)
  df_closest_airport_station.write.mode('overwrite').parquet(f"{blob_url}/df_closest_airport_station")
  
else:
  df_closest_airport_station = spark.read.parquet(f'{blob_url}/df_closest_airport_station/')
  
if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_closest_airport_station, sample_fraction=1.0)
  displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ![Weather Station EDA Table](https://raw.githubusercontent.com/UCB-w261/w261-f21-finalproject-team-02/master/images/eda_weather_station_airport.png?token=AOG7ANXPAEWB4VKIELP2OJDBW2D6Y)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Historical Flights Dataset
# MAGIC 
# MAGIC The Historical Flight dataset contains 63,493,682 commerical domestic flights from years 2015 to 2019.

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

# Convert all features to lowercase names to reduce confusion
df_flights = df_flights.toDF(*[c.lower() for c in df_flights.columns])

if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_flights, 0.001, df_flights_fields)
  displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ![EDA Table](https://raw.githubusercontent.com/UCB-w261/w261-f21-finalproject-team-02/master/images/eda_flight_1.png?token=AOG7ANSGRYCWHLR7HZRJ2TDBW2EKK)
# MAGIC 
# MAGIC ![EDA Table](https://raw.githubusercontent.com/UCB-w261/w261-f21-finalproject-team-02/master/images/eda_flight_2.png?token=AOG7ANTIGOTBV7VJNDPHVM3BW2EKU)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Remove Invalid & Duplicate Flights
# MAGIC 
# MAGIC Flights that have been cancelled, diverted, lack an arrival time, lack a tail number, have a flight time less than 30 minutes, lack a depature time, or lack an arrival delay are considered invalid. Additionally, flights that have the plane visit the same airport more than once in the same day should be considered duplicates. This reduces this dataset from 63,493,682 down to 48,844,778.

# COMMAND ----------

pre_filtered_count = df_flights.count()
df_valid_flights = df_flights.where((df_flights.cancelled == 0) & # flight hasn't been cancelled
                                    (df_flights.diverted == 0) & # flight hasn't been diverted
                                    (df_flights.arr_time.isNotNull()) & # flight has an arrival time 
                                    (df_flights.tail_num.isNotNull()) & # flight has a tail number
                                    (df_flights.air_time > 30) & # flight was in the air more than 30 minutes
                                    (df_flights.dep_delay.isNotNull()) & # departure delay was indicated
                                    (df_flights.arr_delay.isNotNull()) # arrival delay was indicated
                                   )

# Find all duplicate flights by windowing by tail number, flight date, and origin. Sort by the scheduled departure time and only use the latest one
window = Window.partitionBy('tail_num', 'fl_date', 'origin_airport_id').orderBy(sf.col('crs_dep_time').desc())
df_valid_flights = df_valid_flights.withColumn('rank', sf.rank().over(window).cast(IntegerType())).filter(sf.col('rank') == 1).drop('rank')

filtered_count = df_valid_flights.count()

print(f'The original dataset had {pre_filtered_count:,} records, {filtered_count:,} records remain after filtering.')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Join Flight Data to Weather Station Data
# MAGIC 
# MAGIC We join the airport information containing the closest weather station identifier to the historical flight data, this gets us the nearest weather station and UTC offsets for the origin and destination of each historical flight.

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
# MAGIC 
# MAGIC We believe that certain pieces of previous flight information for a tail number in the same day, in other words, the previous leg of a multi-leg flight, contains valuable information for predicting late departures. 

# COMMAND ----------

previous_flight_features = {'origin_airport_id', 'dep_delay_new', 'crs_elapsed_time', 'dep_del15', 'crs_dep_datetime_utc', 'dep_datetime_utc'}
windowSpec = Window.partitionBy('tail_num', 'fl_date').orderBy(sf.col('dep_datetime_utc').desc())

for previous_flight_feature in previous_flight_features:
  df_valid_flights = df_valid_flights.withColumn(f'previous_flight_{previous_flight_feature}', sf.lag(previous_flight_feature, 1).over(window))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Weather Data
# MAGIC 
# MAGIC The original weather dataset contains 630,904,436 rows. The fields WND, CIG, VIS, TMP, DEW, and SLP contain comma-delimited information regarding the raw values, qauality of the reading, additional coded values, and "magic" values mean NULL.
# MAGIC 
# MAGIC We join the weather data to the just the weather stations that are closest to the airports and we end up with 93,145,915 rows of weather data.

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
# MAGIC 
# MAGIC We aggregate the weather data into 30 minute windows, recording the mean, min and max of all numeric weather features in that timeframe. The aggregation of the weather data allows us to match this bucketed timeframe to bucketed timeframes for the fights, it takes the dataset from 93,145,915 rows to 70,539,940. 

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

# We bucket the read_date of the into 30 minute buckets
seconds_window = sf.from_unixtime(sf.unix_timestamp('read_date') - sf.unix_timestamp('read_date') % WEATHER_AGGREGATE_WINDOW_SECONDS)
df_weather_summary = df_weather.withColumn('aggregated_datetime', 
                                           seconds_window.cast(TimestampType())).groupBy('station_id', 
                                                                                         'aggregated_datetime').agg(*expressions)

if RENDER_EDA_TABLES:
  html, _ = generate_eda_table(df_weather_summary, 0.002, df_weather_fields)
  displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Join Flight Data to Weather Data
# MAGIC 
# MAGIC We join the flight dataset with the weather dataset by bucketing the `crs_dep_datetime_utc` of the flight dataset into 30 minute buckets just like we did for the weather data and then perform a left join on the bucketed timestamps. We perform a left-join to keep flights that do not have any weather data associated.

# COMMAND ----------

# Window the data by the scheduled departure date into 30-minute buckets. 
seconds_window = sf.from_unixtime(sf.unix_timestamp('crs_dep_datetime_utc') - sf.unix_timestamp('crs_dep_datetime_utc') % WEATHER_AGGREGATE_WINDOW_SECONDS)

# Match the bucketed weather datetime with the bucketed scheduled departure date subtracted by two hours. We perform a left-join, rather than inner, so as to keep flights that don't have corresponding weather data. 
df_joined = df_valid_flights.withColumn('aggregated_datetime', seconds_window.cast(TimestampType()) - sf.expr("INTERVAL 2 HOURS")).join(
  df_weather_summary.withColumn('origin_station_id', sf.col('station_id')),
  on=['origin_station_id', 'aggregated_datetime'], how='left'
)

# COMMAND ----------

# There are several features that are 100% NULL or would not be known two hours before a departure, we eliminate those features to reduce complexity.

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
# MAGIC 
# MAGIC The dataset contains MANY more flights that are not delayed by 15 minutes or more than flights that are, so for the training set we balance the examples for both so the model does not become overfit.

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
# MAGIC 
# MAGIC There are XXXXX flights that do not contain weather information, however, we do not wish to exclude these from training. We impute 
# MAGIC 
# MAGIC Rather than setting the NULLs to zero, which for some features be an extreme value, we set the NULLs to the mean.

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
  dbutils.fs.rm(f"{blob_url}/df_joined", True)
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
# MAGIC 
# MAGIC There are certain times of the year where people travel more than others; summer break, Thanksgiving, July 4th, etc. Depending on the days of the week and school schedules, these popular travel days can vary year-to-year but generally fall into the same 10-day range year-after-year. We have added the `day of year` feature to capture these busy travel days and the potential congestion they cause.  

# COMMAND ----------

df_joined = df_joined.withColumn('dep_day_of_year', sf.dayofyear('dep_datetime_local').cast(DoubleType()))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Minute of Day
# MAGIC 
# MAGIC Much like the `day of year`, `minute of day` is designed to capture the approximate time of day that people generally prefer to travel with more granularity than simply the hour.  

# COMMAND ----------

df_joined = df_joined.withColumn("dep_minute_of_day", (sf.hour('dep_datetime_local') * 60 + sf.minute('dep_datetime_local')).cast(DoubleType()))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Previous Flight Delayed
# MAGIC 
# MAGIC Domestic flights typically have multiple legs throughout the day, meaning they drop passengers off at different stops. Airlines build buffer into the flight schedule to accomodate delays, however, that buffer is very limited. If the previous flight is delayed by more than the buffer then the next flight is going to be delayed. However, because we are predicting flight delays by two-hours ahead of time, only if the previous flight is longer than two hours will we know if it was delayed.

# COMMAND ----------

@sf.udf
def udf_calculate_previous_flight_delay(previous_scheduled_elapsed_time, previous_dep_delay_new):
  if previous_scheduled_elapsed_time is not None and previous_scheduled_elapsed_time > 120:
    return previous_dep_delay_new
  else:
    return 0
    

df_joined = df_joined.withColumn('previous_flight_dep_delay_new_2', 
                                  udf_calculate_previous_flight_delay('previous_flight_crs_elapsed_time', 'previous_flight_dep_delay_new').cast(DoubleType()))
#TODO: Add back previous_flight_crs_elapsed_time

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Handle NULL values
# MAGIC 
# MAGIC Flights that did not have a previous leg will, by defintion, will not be delayed by a previous leg, so we set the `previous_flight_crs_elapsed_time` and `previous_flight_dep_delay_new` features to zero.

# COMMAND ----------

zero_fills = ['previous_flight_crs_elapsed_time', 'previous_flight_dep_delay_new']

df_joined = df_joined.na.fill(value=0, subset=zero_fills)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Log10 
# MAGIC 
# MAGIC We perform a log10 on all the continuious fields, this has the affect of sacrificing data fidelity in exchange for usually normalizing the data. 

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

if INITIALIZE_DATASETS:
  dbutils.fs.rm(f"{blob_url}/df_joined_final", True)
  df_joined.write.mode('overwrite').parquet(f"{blob_url}/df_joined_final")

print('loading from storage...')
df_joined = spark.read.parquet(f'{blob_url}/df_joined_final/*')

print(df_joined.count())
  
if RENDER_EDA_TABLES:
  
  html, df_joined_sample = generate_eda_table(df_joined, 0.001, {})
  displayHTML(html)

  df_joined_sample.hist(figsize=(35,35), bins=15)
  plt.show()    

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Feature Selection

# COMMAND ----------

target_feature = 'dep_del15'
split_feature = 'crs_dep_datetime_utc'
categorical_features = list({'dest',
                             'origin',
                             'op_unique_carrier',
                             'year', 
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
all_features.append(split_feature)

df_raw_features = df_joined.select(*all_features)

df_raw_features = df_raw_features.na.fill(value=-1, subset=['previous_flight_origin_airport_id'])
df_raw_features = df_raw_features.na.fill(value=0, subset=['previous_flight_dep_delay_new_2_log'])

imputer = Imputer(inputCols=continuous_features, outputCols=continuous_features).setStrategy("mean")
df_raw_features = imputer.fit(df_raw_features).transform(df_raw_features)

if RENDER_EDA_TABLES:
  
  html, df_raw_features_sample = generate_eda_table(df_raw_features, 0.001, {})
  displayHTML(html)

  df_raw_features_sample.hist(figsize=(35,35), bins=15)
  plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Build Pipeline

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import when

# Separate our training data (pre-2019) and blind test data (2019), which we will use in final step for testing. 
df_raw_features_train = df_raw_features.where("year < 2019").cache()
df_raw_features_test = df_raw_features.where("year >= 2019").cache()

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

featureTransform = Pipeline(stages=pipeline_steps).fit(df_raw_features_train)

train_model = featureTransform.transform(df_raw_features_train)
test_model = featureTransform.transform(df_raw_features_test)

train_model

# COMMAND ----------

# Add percent rank to aid in cross validation/splitting
train_model = train_model.withColumn('rank', sf.percent_rank().over(Window.partitionBy().orderBy('crs_dep_datetime_utc')))
# Get 10 folds (20 train + val splits) to generate distribution for statistical testing. 
# Cache for faster access times. 
train_model = train_model.withColumn("foldNumber", when((train_model.rank < .07), lit(0)) \
                                           .when((train_model.rank < .1), lit(1)) \
                                           .when((train_model.rank < .17), lit(2)) \
                                           .when((train_model.rank < .2), lit(3)) \
                                           .when((train_model.rank < .27), lit(4)) \
                                           .when((train_model.rank < .3), lit(5)) \
                                           .when((train_model.rank < .37), lit(6)) \
                                           .when((train_model.rank < .4), lit(7)) \
                                           .when((train_model.rank < .47), lit(8)) \
                                           .when((train_model.rank < .5), lit(9)) \
                                           .when((train_model.rank < .57), lit(10)) \
                                           .when((train_model.rank < .6), lit(11)) \
                                           .when((train_model.rank < .67), lit(12)) \
                                           .when((train_model.rank < .7), lit(13)) \
                                           .when((train_model.rank < .77), lit(14)) \
                                           .when((train_model.rank < .8), lit(15)) \
                                           .when((train_model.rank < .87), lit(16)) \
                                           .when((train_model.rank < .9), lit(17)) \
                                           .when((train_model.rank < .97), lit(18)) \
                                           .otherwise(lit(19))).cache()

# COMMAND ----------

#Check that everything is working as intended (i.e. no data leakage, foldNumbers correctly allocated, features transformed appropriately, etc.)
train_model.sample(fraction=0.0001).toPandas()

# COMMAND ----------

#Create a validation dataframe for our majority class prediction model. This will be all rows with odd numbers in "foldNumber" column (i.e i % 2 != 0). 
validation_data = train_model.where("foldNumber % 2 != 0")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Algorithm Exploration
# MAGIC Apply 2 to 3 algorithms to the training set, and discuss expectations, trade-offs, and results. These will serve as your baselines - do not spend too much time fine tuning these. You will want to use this process to select a final algorithm which you will spend your efforts on fine tuning.

# COMMAND ----------

from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier

from pyspark.sql import functions as f

import random

from scipy import stats
import matplotlib.pyplot as plt

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# COMMAND ----------

# MAGIC %md Define Experimental Framework Constraints, based on performance limitations observed on results here: https://docs.google.com/spreadsheets/d/1aqF0zf6Fm9QCM_mi0E-NS_F1LLFPrZAXy-ICT6-ycn4/edit?usp=sharing

# COMMAND ----------

# Ideal parallelism param for CV observed to be 35 for our cluster. 
PARELLELISM = 35

# Ideal number of PCA features observed to be 5, given just 5 features explain ~100% of variance. 
PCA_FEATURES = 5 

# Assuming 2 hyperparameters to search over, we limit ourselves to 2 values for each, 
# given an optimal param search of 4, given our parallelism and config. 
NUM_PARAM_VALUES = 2

# Ideal sample size observed to be 20% for our experimentation framework, given time constraint of less than 120 minutes for model selection. 
SAMPLE_SIZE = 0.0001

# Must constrain max_tree_depth param value to 4, given linear scaling of runtime after this depth. 
MAX_TREE_DEPTH = 4

# COMMAND ----------

# Downsize train_model to ideal sample size specified above. 
train_model_small = train_model.sample(SAMPLE_SIZE).cache()
validation_data_small = train_model_small.where("foldNumber % 2 != 0")

# COMMAND ----------

# MAGIC %md 1.) Define custom functions that train and test models and append to a comparison table.

# COMMAND ----------

# MAGIC %md Modified code for pyspark.ml.tuning to get a time-series valid, cross-validation class implementation. 

# COMMAND ----------

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import itertools
from multiprocessing.pool import ThreadPool

import numpy as np

from pyspark import keyword_only, since, SparkContext, inheritable_thread_target
from pyspark.ml import Estimator, Transformer, Model
from pyspark.ml.common import inherit_doc, _py2java, _java2py
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.param.shared import HasCollectSubModels, HasParallelism, HasSeed
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MetaAlgorithmReadWrite,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
    JavaMLReader,
    JavaMLWriter,
)
from pyspark.ml.wrapper import JavaParams, JavaEstimator, JavaWrapper
from pyspark.sql.functions import col, lit, rand, UserDefinedFunction
from pyspark.sql.types import BooleanType

__all__ = [
    "ParamGridBuilder",
    "CrossValidator",
    "CrossValidatorModel",
    "TrainValidationSplit",
    "TrainValidationSplitModel",
]


def _parallelFitTasks(est, train, eva, validation, epm, collectSubModels):
    """
    Creates a list of callables which can be called from different threads to fit and evaluate
    an estimator in parallel. Each callable returns an `(index, metric)` pair.
    Parameters
    ----------
    est : :py:class:`pyspark.ml.baseEstimator`
        he estimator to be fit.
    train : :py:class:`pyspark.sql.DataFrame`
        DataFrame, training data set, used for fitting.
    eva : :py:class:`pyspark.ml.evaluation.Evaluator`
        used to compute `metric`
    validation : :py:class:`pyspark.sql.DataFrame`
        DataFrame, validation data set, used for evaluation.
    epm : :py:class:`collections.abc.Sequence`
        Sequence of ParamMap, params maps to be used during fitting & evaluation.
    collectSubModel : bool
        Whether to collect sub model.
    Returns
    -------
    tuple
        (int, float, subModel), an index into `epm` and the associated metric value.
    """
    modelIter = est.fitMultiple(train, epm)

    def singleTask():
        index, model = next(modelIter)
        # TODO: duplicate evaluator to take extra params from input
        #  Note: Supporting tuning params in evaluator need update method
        #  `MetaAlgorithmReadWrite.getAllNestedStages`, make it return
        #  all nested stages and evaluators
        metric = eva.evaluate(model.transform(validation, epm[index]))
        return index, metric, model if collectSubModels else None

    return [singleTask] * len(epm)


class ParamGridBuilder(object):
    r"""
    Builder for a param grid used in grid search-based model selection.
    .. versionadded:: 1.4.0
    Examples
    --------
    >>> from pyspark.ml.classification import LogisticRegression
    >>> lr = LogisticRegression()
    >>> output = ParamGridBuilder() \
    ...     .baseOn({lr.labelCol: 'l'}) \
    ...     .baseOn([lr.predictionCol, 'p']) \
    ...     .addGrid(lr.regParam, [1.0, 2.0]) \
    ...     .addGrid(lr.maxIter, [1, 5]) \
    ...     .build()
    >>> expected = [
    ...     {lr.regParam: 1.0, lr.maxIter: 1, lr.labelCol: 'l', lr.predictionCol: 'p'},
    ...     {lr.regParam: 2.0, lr.maxIter: 1, lr.labelCol: 'l', lr.predictionCol: 'p'},
    ...     {lr.regParam: 1.0, lr.maxIter: 5, lr.labelCol: 'l', lr.predictionCol: 'p'},
    ...     {lr.regParam: 2.0, lr.maxIter: 5, lr.labelCol: 'l', lr.predictionCol: 'p'}]
    >>> len(output) == len(expected)
    True
    >>> all([m in expected for m in output])
    True
    """

    def __init__(self):
        self._param_grid = {}

    @since("1.4.0")
    def addGrid(self, param, values):
        """
        Sets the given parameters in this grid to fixed values.
        param must be an instance of Param associated with an instance of Params
        (such as Estimator or Transformer).
        """
        if isinstance(param, Param):
            self._param_grid[param] = values
        else:
            raise TypeError("param must be an instance of Param")

        return self

    @since("1.4.0")
    def baseOn(self, *args):
        """
        Sets the given parameters in this grid to fixed values.
        Accepts either a parameter dictionary or a list of (parameter, value) pairs.
        """
        if isinstance(args[0], dict):
            self.baseOn(*args[0].items())
        else:
            for (param, value) in args:
                self.addGrid(param, [value])

        return self

    @since("1.4.0")
    def build(self):
        """
        Builds and returns all combinations of parameters specified
        by the param grid.
        """
        keys = self._param_grid.keys()
        grid_values = self._param_grid.values()

        def to_key_value_pairs(keys, values):
            return [(key, key.typeConverter(value)) for key, value in zip(keys, values)]

        return [dict(to_key_value_pairs(keys, prod)) for prod in itertools.product(*grid_values)]


class _ValidatorParams(HasSeed):
    """
    Common params for TrainValidationSplit and CrossValidator.
    """

    estimator = Param(Params._dummy(), "estimator", "estimator to be cross-validated")
    estimatorParamMaps = Param(Params._dummy(), "estimatorParamMaps", "estimator param maps")
    evaluator = Param(
        Params._dummy(),
        "evaluator",
        "evaluator used to select hyper-parameters that maximize the validator metric",
    )

    @since("2.0.0")
    def getEstimator(self):
        """
        Gets the value of estimator or its default value.
        """
        return self.getOrDefault(self.estimator)

    @since("2.0.0")
    def getEstimatorParamMaps(self):
        """
        Gets the value of estimatorParamMaps or its default value.
        """
        return self.getOrDefault(self.estimatorParamMaps)

    @since("2.0.0")
    def getEvaluator(self):
        """
        Gets the value of evaluator or its default value.
        """
        return self.getOrDefault(self.evaluator)

    @classmethod
    def _from_java_impl(cls, java_stage):
        """
        Return Python estimator, estimatorParamMaps, and evaluator from a Java ValidatorParams.
        """

        # Load information from java_stage to the instance.
        estimator = JavaParams._from_java(java_stage.getEstimator())
        evaluator = JavaParams._from_java(java_stage.getEvaluator())
        if isinstance(estimator, JavaEstimator):
            epms = [
                estimator._transfer_param_map_from_java(epm)
                for epm in java_stage.getEstimatorParamMaps()
            ]
        elif MetaAlgorithmReadWrite.isMetaEstimator(estimator):
            # Meta estimator such as Pipeline, OneVsRest
            epms = _ValidatorSharedReadWrite.meta_estimator_transfer_param_maps_from_java(
                estimator, java_stage.getEstimatorParamMaps()
            )
        else:
            raise ValueError("Unsupported estimator used in tuning: " + str(estimator))

        return estimator, epms, evaluator

    def _to_java_impl(self):
        """
        Return Java estimator, estimatorParamMaps, and evaluator from this Python instance.
        """

        gateway = SparkContext._gateway
        cls = SparkContext._jvm.org.apache.spark.ml.param.ParamMap

        estimator = self.getEstimator()
        if isinstance(estimator, JavaEstimator):
            java_epms = gateway.new_array(cls, len(self.getEstimatorParamMaps()))
            for idx, epm in enumerate(self.getEstimatorParamMaps()):
                java_epms[idx] = self.getEstimator()._transfer_param_map_to_java(epm)
        elif MetaAlgorithmReadWrite.isMetaEstimator(estimator):
            # Meta estimator such as Pipeline, OneVsRest
            java_epms = _ValidatorSharedReadWrite.meta_estimator_transfer_param_maps_to_java(
                estimator, self.getEstimatorParamMaps()
            )
        else:
            raise ValueError("Unsupported estimator used in tuning: " + str(estimator))

        java_estimator = self.getEstimator()._to_java()
        java_evaluator = self.getEvaluator()._to_java()
        return java_estimator, java_epms, java_evaluator


class _ValidatorSharedReadWrite:
    @staticmethod
    def meta_estimator_transfer_param_maps_to_java(pyEstimator, pyParamMaps):
        pyStages = MetaAlgorithmReadWrite.getAllNestedStages(pyEstimator)
        stagePairs = list(map(lambda stage: (stage, stage._to_java()), pyStages))
        sc = SparkContext._active_spark_context

        paramMapCls = SparkContext._jvm.org.apache.spark.ml.param.ParamMap
        javaParamMaps = SparkContext._gateway.new_array(paramMapCls, len(pyParamMaps))

        for idx, pyParamMap in enumerate(pyParamMaps):
            javaParamMap = JavaWrapper._new_java_obj("org.apache.spark.ml.param.ParamMap")
            for pyParam, pyValue in pyParamMap.items():
                javaParam = None
                for pyStage, javaStage in stagePairs:
                    if pyStage._testOwnParam(pyParam.parent, pyParam.name):
                        javaParam = javaStage.getParam(pyParam.name)
                        break
                if javaParam is None:
                    raise ValueError("Resolve param in estimatorParamMaps failed: " + str(pyParam))
                if isinstance(pyValue, Params) and hasattr(pyValue, "_to_java"):
                    javaValue = pyValue._to_java()
                else:
                    javaValue = _py2java(sc, pyValue)
                pair = javaParam.w(javaValue)
                javaParamMap.put([pair])
            javaParamMaps[idx] = javaParamMap
        return javaParamMaps

    @staticmethod
    def meta_estimator_transfer_param_maps_from_java(pyEstimator, javaParamMaps):
        pyStages = MetaAlgorithmReadWrite.getAllNestedStages(pyEstimator)
        stagePairs = list(map(lambda stage: (stage, stage._to_java()), pyStages))
        sc = SparkContext._active_spark_context
        pyParamMaps = []
        for javaParamMap in javaParamMaps:
            pyParamMap = dict()
            for javaPair in javaParamMap.toList():
                javaParam = javaPair.param()
                pyParam = None
                for pyStage, javaStage in stagePairs:
                    if pyStage._testOwnParam(javaParam.parent(), javaParam.name()):
                        pyParam = pyStage.getParam(javaParam.name())
                if pyParam is None:
                    raise ValueError(
                        "Resolve param in estimatorParamMaps failed: "
                        + javaParam.parent()
                        + "."
                        + javaParam.name()
                    )
                javaValue = javaPair.value()
                if sc._jvm.Class.forName(
                    "org.apache.spark.ml.util.DefaultParamsWritable"
                ).isInstance(javaValue):
                    pyValue = JavaParams._from_java(javaValue)
                else:
                    pyValue = _java2py(sc, javaValue)
                pyParamMap[pyParam] = pyValue
            pyParamMaps.append(pyParamMap)
        return pyParamMaps

    @staticmethod
    def is_java_convertible(instance):
        allNestedStages = MetaAlgorithmReadWrite.getAllNestedStages(instance.getEstimator())
        evaluator_convertible = isinstance(instance.getEvaluator(), JavaParams)
        estimator_convertible = all(map(lambda stage: hasattr(stage, "_to_java"), allNestedStages))
        return estimator_convertible and evaluator_convertible

    @staticmethod
    def saveImpl(path, instance, sc, extraMetadata=None):
        numParamsNotJson = 0
        jsonEstimatorParamMaps = []
        for paramMap in instance.getEstimatorParamMaps():
            jsonParamMap = []
            for p, v in paramMap.items():
                jsonParam = {"parent": p.parent, "name": p.name}
                if (
                    (isinstance(v, Estimator) and not MetaAlgorithmReadWrite.isMetaEstimator(v))
                    or isinstance(v, Transformer)
                    or isinstance(v, Evaluator)
                ):
                    relative_path = f"epm_{p.name}{numParamsNotJson}"
                    param_path = os.path.join(path, relative_path)
                    numParamsNotJson += 1
                    v.save(param_path)
                    jsonParam["value"] = relative_path
                    jsonParam["isJson"] = False
                elif isinstance(v, MLWritable):
                    raise RuntimeError(
                        "ValidatorSharedReadWrite.saveImpl does not handle parameters of type: "
                        "MLWritable that are not Estimaor/Evaluator/Transformer, and if parameter "
                        "is estimator, it cannot be meta estimator such as Validator or OneVsRest"
                    )
                else:
                    jsonParam["value"] = v
                    jsonParam["isJson"] = True
                jsonParamMap.append(jsonParam)
            jsonEstimatorParamMaps.append(jsonParamMap)

        skipParams = ["estimator", "evaluator", "estimatorParamMaps"]
        jsonParams = DefaultParamsWriter.extractJsonParams(instance, skipParams)
        jsonParams["estimatorParamMaps"] = jsonEstimatorParamMaps

        DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, jsonParams)
        evaluatorPath = os.path.join(path, "evaluator")
        instance.getEvaluator().save(evaluatorPath)
        estimatorPath = os.path.join(path, "estimator")
        instance.getEstimator().save(estimatorPath)

    @staticmethod
    def load(path, sc, metadata):
        evaluatorPath = os.path.join(path, "evaluator")
        evaluator = DefaultParamsReader.loadParamsInstance(evaluatorPath, sc)
        estimatorPath = os.path.join(path, "estimator")
        estimator = DefaultParamsReader.loadParamsInstance(estimatorPath, sc)

        uidToParams = MetaAlgorithmReadWrite.getUidMap(estimator)
        uidToParams[evaluator.uid] = evaluator

        jsonEstimatorParamMaps = metadata["paramMap"]["estimatorParamMaps"]

        estimatorParamMaps = []
        for jsonParamMap in jsonEstimatorParamMaps:
            paramMap = {}
            for jsonParam in jsonParamMap:
                est = uidToParams[jsonParam["parent"]]
                param = getattr(est, jsonParam["name"])
                if "isJson" not in jsonParam or ("isJson" in jsonParam and jsonParam["isJson"]):
                    value = jsonParam["value"]
                else:
                    relativePath = jsonParam["value"]
                    valueSavedPath = os.path.join(path, relativePath)
                    value = DefaultParamsReader.loadParamsInstance(valueSavedPath, sc)
                paramMap[param] = value
            estimatorParamMaps.append(paramMap)

        return metadata, estimator, evaluator, estimatorParamMaps

    @staticmethod
    def validateParams(instance):
        estiamtor = instance.getEstimator()
        evaluator = instance.getEvaluator()
        uidMap = MetaAlgorithmReadWrite.getUidMap(estiamtor)

        for elem in [evaluator] + list(uidMap.values()):
            if not isinstance(elem, MLWritable):
                raise ValueError(
                    f"Validator write will fail because it contains {elem.uid} "
                    f"which is not writable."
                )

        estimatorParamMaps = instance.getEstimatorParamMaps()
        paramErr = (
            "Validator save requires all Params in estimatorParamMaps to apply to "
            "its Estimator, An extraneous Param was found: "
        )
        for paramMap in estimatorParamMaps:
            for param in paramMap:
                if param.parent not in uidMap:
                    raise ValueError(paramErr + repr(param))

    @staticmethod
    def getValidatorModelWriterPersistSubModelsParam(writer):
        if "persistsubmodels" in writer.optionMap:
            persistSubModelsParam = writer.optionMap["persistsubmodels"].lower()
            if persistSubModelsParam == "true":
                return True
            elif persistSubModelsParam == "false":
                return False
            else:
                raise ValueError(
                    f"persistSubModels option value {persistSubModelsParam} is invalid, "
                    f"the possible values are True, 'True' or False, 'False'"
                )
        else:
            return writer.instance.subModels is not None


_save_with_persist_submodels_no_submodels_found_err = (
    "When persisting tuning models, you can only set persistSubModels to true if the tuning "
    "was done with collectSubModels set to true. To save the sub-models, try rerunning fitting "
    "with collectSubModels set to true."
)


@inherit_doc
class CrossValidatorReader(MLReader):
    def __init__(self, cls):
        super(CrossValidatorReader, self).__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        if not DefaultParamsReader.isPythonParamsInstance(metadata):
            return JavaMLReader(self.cls).load(path)
        else:
            metadata, estimator, evaluator, estimatorParamMaps = _ValidatorSharedReadWrite.load(
                path, self.sc, metadata
            )
            cv = CrossValidator(
                estimator=estimator, estimatorParamMaps=estimatorParamMaps, evaluator=evaluator
            )
            cv = cv._resetUid(metadata["uid"])
            DefaultParamsReader.getAndSetParams(cv, metadata, skipParams=["estimatorParamMaps"])
            return cv


@inherit_doc
class CrossValidatorWriter(MLWriter):
    def __init__(self, instance):
        super(CrossValidatorWriter, self).__init__()
        self.instance = instance

    def saveImpl(self, path):
        _ValidatorSharedReadWrite.validateParams(self.instance)
        _ValidatorSharedReadWrite.saveImpl(path, self.instance, self.sc)


@inherit_doc
class CrossValidatorModelReader(MLReader):
    def __init__(self, cls):
        super(CrossValidatorModelReader, self).__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        if not DefaultParamsReader.isPythonParamsInstance(metadata):
            return JavaMLReader(self.cls).load(path)
        else:
            metadata, estimator, evaluator, estimatorParamMaps = _ValidatorSharedReadWrite.load(
                path, self.sc, metadata
            )
            numFolds = metadata["paramMap"]["numFolds"]
            bestModelPath = os.path.join(path, "bestModel")
            bestModel = DefaultParamsReader.loadParamsInstance(bestModelPath, self.sc)
            avgMetrics = metadata["avgMetrics"]
            if "stdMetrics" in metadata:
                stdMetrics = metadata["stdMetrics"]
            else:
                stdMetrics = None
            persistSubModels = ("persistSubModels" in metadata) and metadata["persistSubModels"]

            if persistSubModels:
                subModels = [[None] * len(estimatorParamMaps)] * numFolds
                for splitIndex in range(numFolds):
                    for paramIndex in range(len(estimatorParamMaps)):
                        modelPath = os.path.join(
                            path, "subModels", f"fold{splitIndex}", f"{paramIndex}"
                        )
                        subModels[splitIndex][paramIndex] = DefaultParamsReader.loadParamsInstance(
                            modelPath, self.sc
                        )
            else:
                subModels = None

            cvModel = CrossValidatorModel(
                bestModel, avgMetrics=avgMetrics, subModels=subModels, stdMetrics=stdMetrics
            )
            cvModel = cvModel._resetUid(metadata["uid"])
            cvModel.set(cvModel.estimator, estimator)
            cvModel.set(cvModel.estimatorParamMaps, estimatorParamMaps)
            cvModel.set(cvModel.evaluator, evaluator)
            DefaultParamsReader.getAndSetParams(
                cvModel, metadata, skipParams=["estimatorParamMaps"]
            )
            return cvModel


@inherit_doc
class CrossValidatorModelWriter(MLWriter):
    def __init__(self, instance):
        super(CrossValidatorModelWriter, self).__init__()
        self.instance = instance

    def saveImpl(self, path):
        _ValidatorSharedReadWrite.validateParams(self.instance)
        instance = self.instance
        persistSubModels = _ValidatorSharedReadWrite.getValidatorModelWriterPersistSubModelsParam(
            self
        )
        extraMetadata = {"avgMetrics": instance.avgMetrics, "persistSubModels": persistSubModels}
        if instance.stdMetrics:
            extraMetadata["stdMetrics"] = instance.stdMetrics

        _ValidatorSharedReadWrite.saveImpl(path, instance, self.sc, extraMetadata=extraMetadata)
        bestModelPath = os.path.join(path, "bestModel")
        instance.bestModel.save(bestModelPath)
        if persistSubModels:
            if instance.subModels is None:
                raise ValueError(_save_with_persist_submodels_no_submodels_found_err)
            subModelsPath = os.path.join(path, "subModels")
            for splitIndex in range(instance.getNumFolds()):
                splitPath = os.path.join(subModelsPath, f"fold{splitIndex}")
                for paramIndex in range(len(instance.getEstimatorParamMaps())):
                    modelPath = os.path.join(splitPath, f"{paramIndex}")
                    instance.subModels[splitIndex][paramIndex].save(modelPath)


class _CrossValidatorParams(_ValidatorParams):
    """
    Params for :py:class:`CrossValidator` and :py:class:`CrossValidatorModel`.
    .. versionadded:: 3.0.0
    """

    numFolds = Param(
        Params._dummy(),
        "numFolds",
        "number of folds for cross validation",
        typeConverter=TypeConverters.toInt,
    )

    foldCol = Param(
        Params._dummy(),
        "foldCol",
        "Param for the column name of user "
        + "specified fold number. Once this is specified, :py:class:`CrossValidator` "
        + "won't do random k-fold split. Note that this column should be integer type "
        + "with range [0, numFolds) and Spark will throw exception on out-of-range "
        + "fold numbers.",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self, *args):
        super(_CrossValidatorParams, self).__init__(*args)
        self._setDefault(numFolds=2, foldCol="")

    @since("1.4.0")
    def getNumFolds(self):
        """
        Gets the value of numFolds or its default value.
        """
        return self.getOrDefault(self.numFolds)

    @since("3.1.0")
    def getFoldCol(self):
        """
        Gets the value of foldCol or its default value.
        """
        return self.getOrDefault(self.foldCol)


class CrossValidator(
    Estimator, _CrossValidatorParams, HasParallelism, HasCollectSubModels, MLReadable, MLWritable
):
    """
    K-fold cross validation performs model selection by splitting the dataset into a set of
    non-overlapping randomly partitioned folds which are used as separate training and test datasets
    e.g., with k=3 folds, K-fold cross validation will generate 3 (training, test) dataset pairs,
    each of which uses 2/3 of the data for training and 1/3 for testing. Each fold is used as the
    test set exactly once.
    .. versionadded:: 1.4.0
    Examples
    --------
    >>> from pyspark.ml.classification import LogisticRegression
    >>> from pyspark.ml.evaluation import BinaryClassificationEvaluator
    >>> from pyspark.ml.linalg import Vectors
    >>> from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
    >>> import tempfile
    >>> dataset = spark.createDataFrame(
    ...     [(Vectors.dense([0.0]), 0.0),
    ...      (Vectors.dense([0.4]), 1.0),
    ...      (Vectors.dense([0.5]), 0.0),
    ...      (Vectors.dense([0.6]), 1.0),
    ...      (Vectors.dense([1.0]), 1.0)] * 10,
    ...     ["features", "label"])
    >>> lr = LogisticRegression()
    >>> grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    >>> evaluator = BinaryClassificationEvaluator()
    >>> cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator,
    ...     parallelism=2)
    >>> cvModel = cv.fit(dataset)
    >>> cvModel.getNumFolds()
    3
    >>> cvModel.avgMetrics[0]
    0.5
    >>> path = tempfile.mkdtemp()
    >>> model_path = path + "/model"
    >>> cvModel.write().save(model_path)
    >>> cvModelRead = CrossValidatorModel.read().load(model_path)
    >>> cvModelRead.avgMetrics
    [0.5, ...
    >>> evaluator.evaluate(cvModel.transform(dataset))
    0.8333...
    >>> evaluator.evaluate(cvModelRead.transform(dataset))
    0.8333...
    """

    @keyword_only
    def __init__(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        numFolds=2,
        seed=None,
        parallelism=1,
        collectSubModels=False,
        foldCol="",
    ):
        """
        __init__(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,\
                 seed=None, parallelism=1, collectSubModels=False, foldCol="")
        """
        super(CrossValidator, self).__init__()
        self._setDefault(parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    @since("1.4.0")
    def setParams(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        numFolds=2,
        seed=None,
        parallelism=1,
        collectSubModels=False,
        foldCol="",
    ):
        """
        setParams(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,\
                  seed=None, parallelism=1, collectSubModels=False, foldCol=""):
        Sets params for cross validator.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @since("2.0.0")
    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`estimator`.
        """
        return self._set(estimator=value)

    @since("2.0.0")
    def setEstimatorParamMaps(self, value):
        """
        Sets the value of :py:attr:`estimatorParamMaps`.
        """
        return self._set(estimatorParamMaps=value)

    @since("2.0.0")
    def setEvaluator(self, value):
        """
        Sets the value of :py:attr:`evaluator`.
        """
        return self._set(evaluator=value)

    @since("1.4.0")
    def setNumFolds(self, value):
        """
        Sets the value of :py:attr:`numFolds`.
        """
        return self._set(numFolds=value)

    @since("3.1.0")
    def setFoldCol(self, value):
        """
        Sets the value of :py:attr:`foldCol`.
        """
        return self._set(foldCol=value)

    def setSeed(self, value):
        """
        Sets the value of :py:attr:`seed`.
        """
        return self._set(seed=value)

    def setParallelism(self, value):
        """
        Sets the value of :py:attr:`parallelism`.
        """
        return self._set(parallelism=value)

    def setCollectSubModels(self, value):
        """
        Sets the value of :py:attr:`collectSubModels`.
        """
        return self._set(collectSubModels=value)

    @staticmethod
    def _gen_avg_and_std_metrics(metrics_all):
        avg_metrics = np.mean(metrics_all, axis=0)
        std_metrics = np.std(metrics_all, axis=0)
        return list(avg_metrics), list(std_metrics)

    def _fit(self, dataset):
        print("Running Custom CrossValidator Class")
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        metrics_all = [[0.0] * numModels for i in range(nFolds)]

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        datasets = self._kFold(dataset)
        for i in range(nFolds):
            validation = datasets[i][1].cache()
            train = datasets[i][0].cache()

            tasks = map(
                inheritable_thread_target,
                _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam),
            )
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics_all[i][j] = metric
                if collectSubModelsParam:
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        metrics, std_metrics = CrossValidator._gen_avg_and_std_metrics(metrics_all)

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics, subModels, std_metrics))

    def _kFold(self, dataset):
        nFolds = self.getOrDefault(self.numFolds)
        foldCol = self.getOrDefault(self.foldCol)

        datasets = []
        if not foldCol:
            # Do random k-fold split.
            seed = self.getOrDefault(self.seed)
            h = 1.0 / nFolds
            randCol = self.uid + "_rand"
            df = dataset.select("*", rand(seed).alias(randCol))
            for i in range(nFolds):
                validateLB = i * h
                validateUB = (i + 1) * h
                condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
                validation = df.filter(condition)
                train = df.filter(~condition)
                datasets.append((train, validation))
        else:
            # Use user-specified fold numbers.
            def checker(foldNum):
                if foldNum < 0 or foldNum >= (nFolds*2):
                    raise ValueError(
                        "Fold number must be in range [0, %s), but got %s." % (nFolds, foldNum)
                    )
                return True

            checker_udf = UserDefinedFunction(checker, BooleanType())
            for i in range(nFolds*2):
            #Custom logic to use i as training, and i+1 as validation for i / 2 folds (since each fold is a training/val pair). 
                if i % 2 == 0: 
                    training = dataset.filter(checker_udf(dataset[foldCol]) & (col(foldCol) == lit(i)))
                    validation = dataset.filter(
                        checker_udf(dataset[foldCol]) & (col(foldCol) == lit((i+1)))
                    )
                    if training.rdd.getNumPartitions() == 0 or len(training.take(1)) == 0:
                        raise ValueError("The training data at fold %s is empty." % i)
                    if validation.rdd.getNumPartitions() == 0 or len(validation.take(1)) == 0:
                        raise ValueError("The validation data at fold %s is empty." % i+1)
                    datasets.append((training, validation))

        return datasets

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies creates a deep copy of
        the embedded paramMap, and copies the embedded and extra parameters over.
        .. versionadded:: 1.4.0
        Parameters
        ----------
        extra : dict, optional
            Extra parameters to copy to the new instance
        Returns
        -------
        :py:class:`CrossValidator`
            Copy of this instance
        """
        if extra is None:
            extra = dict()
        newCV = Params.copy(self, extra)
        if self.isSet(self.estimator):
            newCV.setEstimator(self.getEstimator().copy(extra))
        # estimatorParamMaps remain the same
        if self.isSet(self.evaluator):
            newCV.setEvaluator(self.getEvaluator().copy(extra))
        return newCV

    @since("2.3.0")
    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        if _ValidatorSharedReadWrite.is_java_convertible(self):
            return JavaMLWriter(self)
        return CrossValidatorWriter(self)

    @classmethod
    @since("2.3.0")
    def read(cls):
        """Returns an MLReader instance for this class."""
        return CrossValidatorReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java CrossValidator, create and return a Python wrapper of it.
        Used for ML persistence.
        """

        estimator, epms, evaluator = super(CrossValidator, cls)._from_java_impl(java_stage)
        numFolds = java_stage.getNumFolds()
        seed = java_stage.getSeed()
        parallelism = java_stage.getParallelism()
        collectSubModels = java_stage.getCollectSubModels()
        foldCol = java_stage.getFoldCol()
        # Create a new instance of this stage.
        py_stage = cls(
            estimator=estimator,
            estimatorParamMaps=epms,
            evaluator=evaluator,
            numFolds=numFolds,
            seed=seed,
            parallelism=parallelism,
            collectSubModels=collectSubModels,
            foldCol=foldCol,
        )
        py_stage._resetUid(java_stage.uid())
        return py_stage

    def _to_java(self):
        """
        Transfer this instance to a Java CrossValidator. Used for ML persistence.
        Returns
        -------
        py4j.java_gateway.JavaObject
            Java object equivalent to this instance.
        """

        estimator, epms, evaluator = super(CrossValidator, self)._to_java_impl()

        _java_obj = JavaParams._new_java_obj("org.apache.spark.ml.tuning.CrossValidator", self.uid)
        _java_obj.setEstimatorParamMaps(epms)
        _java_obj.setEvaluator(evaluator)
        _java_obj.setEstimator(estimator)
        _java_obj.setSeed(self.getSeed())
        _java_obj.setNumFolds(self.getNumFolds())
        _java_obj.setParallelism(self.getParallelism())
        _java_obj.setCollectSubModels(self.getCollectSubModels())
        _java_obj.setFoldCol(self.getFoldCol())

        return _java_obj


class CrossValidatorModel(Model, _CrossValidatorParams, MLReadable, MLWritable):
    """
    CrossValidatorModel contains the model with the highest average cross-validation
    metric across folds and uses this model to transform input data. CrossValidatorModel
    also tracks the metrics for each param map evaluated.
    .. versionadded:: 1.4.0
    Notes
    -----
    Since version 3.3.0, CrossValidatorModel contains a new attribute "stdMetrics",
    which represent standard deviation of metrics for each paramMap in
    CrossValidator.estimatorParamMaps.
    """

    def __init__(self, bestModel, avgMetrics=None, subModels=None, stdMetrics=None):
        super(CrossValidatorModel, self).__init__()
        #: best model from cross validation
        self.bestModel = bestModel
        #: Average cross-validation metrics for each paramMap in
        #: CrossValidator.estimatorParamMaps, in the corresponding order.
        self.avgMetrics = avgMetrics or []
        #: sub model list from cross validation
        self.subModels = subModels
        #: standard deviation of metrics for each paramMap in
        #: CrossValidator.estimatorParamMaps, in the corresponding order.
        self.stdMetrics = stdMetrics or []

    def _transform(self, dataset):
        return self.bestModel.transform(dataset)

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying bestModel,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.
        It does not copy the extra Params into the subModels.
        .. versionadded:: 1.4.0
        Parameters
        ----------
        extra : dict, optional
            Extra parameters to copy to the new instance
        Returns
        -------
        :py:class:`CrossValidatorModel`
            Copy of this instance
        """
        if extra is None:
            extra = dict()
        bestModel = self.bestModel.copy(extra)
        avgMetrics = list(self.avgMetrics)
        subModels = [
            [sub_model.copy() for sub_model in fold_sub_models]
            for fold_sub_models in self.subModels
        ]
        stdMetrics = list(self.stdMetrics)
        return self._copyValues(
            CrossValidatorModel(bestModel, avgMetrics, subModels, stdMetrics), extra=extra
        )

    @since("2.3.0")
    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        if _ValidatorSharedReadWrite.is_java_convertible(self):
            return JavaMLWriter(self)
        return CrossValidatorModelWriter(self)

    @classmethod
    @since("2.3.0")
    def read(cls):
        """Returns an MLReader instance for this class."""
        return CrossValidatorModelReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java CrossValidatorModel, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        sc = SparkContext._active_spark_context
        bestModel = JavaParams._from_java(java_stage.bestModel())
        avgMetrics = _java2py(sc, java_stage.avgMetrics())
        estimator, epms, evaluator = super(CrossValidatorModel, cls)._from_java_impl(java_stage)

        py_stage = cls(bestModel=bestModel, avgMetrics=avgMetrics)
        params = {
            "evaluator": evaluator,
            "estimator": estimator,
            "estimatorParamMaps": epms,
            "numFolds": java_stage.getNumFolds(),
            "foldCol": java_stage.getFoldCol(),
            "seed": java_stage.getSeed(),
        }
        for param_name, param_val in params.items():
            py_stage = py_stage._set(**{param_name: param_val})

        if java_stage.hasSubModels():
            py_stage.subModels = [
                [JavaParams._from_java(sub_model) for sub_model in fold_sub_models]
                for fold_sub_models in java_stage.subModels()
            ]

        py_stage._resetUid(java_stage.uid())
        return py_stage

    def _to_java(self):
        """
        Transfer this instance to a Java CrossValidatorModel. Used for ML persistence.
        Returns
        -------
        py4j.java_gateway.JavaObject
            Java object equivalent to this instance.
        """

        sc = SparkContext._active_spark_context
        _java_obj = JavaParams._new_java_obj(
            "org.apache.spark.ml.tuning.CrossValidatorModel",
            self.uid,
            self.bestModel._to_java(),
            _py2java(sc, self.avgMetrics),
        )
        estimator, epms, evaluator = super(CrossValidatorModel, self)._to_java_impl()

        params = {
            "evaluator": evaluator,
            "estimator": estimator,
            "estimatorParamMaps": epms,
            "numFolds": self.getNumFolds(),
            "foldCol": self.getFoldCol(),
            "seed": self.getSeed(),
        }
        for param_name, param_val in params.items():
            java_param = _java_obj.getParam(param_name)
            pair = java_param.w(param_val)
            _java_obj.set(pair)

        if self.subModels is not None:
            java_sub_models = [
                [sub_model._to_java() for sub_model in fold_sub_models]
                for fold_sub_models in self.subModels
            ]
            _java_obj.setSubModels(java_sub_models)
        return _java_obj


@inherit_doc
class TrainValidationSplitReader(MLReader):
    def __init__(self, cls):
        super(TrainValidationSplitReader, self).__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        if not DefaultParamsReader.isPythonParamsInstance(metadata):
            return JavaMLReader(self.cls).load(path)
        else:
            metadata, estimator, evaluator, estimatorParamMaps = _ValidatorSharedReadWrite.load(
                path, self.sc, metadata
            )
            tvs = TrainValidationSplit(
                estimator=estimator, estimatorParamMaps=estimatorParamMaps, evaluator=evaluator
            )
            tvs = tvs._resetUid(metadata["uid"])
            DefaultParamsReader.getAndSetParams(tvs, metadata, skipParams=["estimatorParamMaps"])
            return tvs


@inherit_doc
class TrainValidationSplitWriter(MLWriter):
    def __init__(self, instance):
        super(TrainValidationSplitWriter, self).__init__()
        self.instance = instance

    def saveImpl(self, path):
        _ValidatorSharedReadWrite.validateParams(self.instance)
        _ValidatorSharedReadWrite.saveImpl(path, self.instance, self.sc)


@inherit_doc
class TrainValidationSplitModelReader(MLReader):
    def __init__(self, cls):
        super(TrainValidationSplitModelReader, self).__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        if not DefaultParamsReader.isPythonParamsInstance(metadata):
            return JavaMLReader(self.cls).load(path)
        else:
            metadata, estimator, evaluator, estimatorParamMaps = _ValidatorSharedReadWrite.load(
                path, self.sc, metadata
            )
            bestModelPath = os.path.join(path, "bestModel")
            bestModel = DefaultParamsReader.loadParamsInstance(bestModelPath, self.sc)
            validationMetrics = metadata["validationMetrics"]
            persistSubModels = ("persistSubModels" in metadata) and metadata["persistSubModels"]

            if persistSubModels:
                subModels = [None] * len(estimatorParamMaps)
                for paramIndex in range(len(estimatorParamMaps)):
                    modelPath = os.path.join(path, "subModels", f"{paramIndex}")
                    subModels[paramIndex] = DefaultParamsReader.loadParamsInstance(
                        modelPath, self.sc
                    )
            else:
                subModels = None

            tvsModel = TrainValidationSplitModel(
                bestModel, validationMetrics=validationMetrics, subModels=subModels
            )
            tvsModel = tvsModel._resetUid(metadata["uid"])
            tvsModel.set(tvsModel.estimator, estimator)
            tvsModel.set(tvsModel.estimatorParamMaps, estimatorParamMaps)
            tvsModel.set(tvsModel.evaluator, evaluator)
            DefaultParamsReader.getAndSetParams(
                tvsModel, metadata, skipParams=["estimatorParamMaps"]
            )
            return tvsModel


@inherit_doc
class TrainValidationSplitModelWriter(MLWriter):
    def __init__(self, instance):
        super(TrainValidationSplitModelWriter, self).__init__()
        self.instance = instance

    def saveImpl(self, path):
        _ValidatorSharedReadWrite.validateParams(self.instance)
        instance = self.instance
        persistSubModels = _ValidatorSharedReadWrite.getValidatorModelWriterPersistSubModelsParam(
            self
        )

        extraMetadata = {
            "validationMetrics": instance.validationMetrics,
            "persistSubModels": persistSubModels,
        }
        _ValidatorSharedReadWrite.saveImpl(path, instance, self.sc, extraMetadata=extraMetadata)
        bestModelPath = os.path.join(path, "bestModel")
        instance.bestModel.save(bestModelPath)
        if persistSubModels:
            if instance.subModels is None:
                raise ValueError(_save_with_persist_submodels_no_submodels_found_err)
            subModelsPath = os.path.join(path, "subModels")
            for paramIndex in range(len(instance.getEstimatorParamMaps())):
                modelPath = os.path.join(subModelsPath, f"{paramIndex}")
                instance.subModels[paramIndex].save(modelPath)


class _TrainValidationSplitParams(_ValidatorParams):
    """
    Params for :py:class:`TrainValidationSplit` and :py:class:`TrainValidationSplitModel`.
    .. versionadded:: 3.0.0
    """

    trainRatio = Param(
        Params._dummy(),
        "trainRatio",
        "Param for ratio between train and\
     validation data. Must be between 0 and 1.",
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self, *args):
        super(_TrainValidationSplitParams, self).__init__(*args)
        self._setDefault(trainRatio=0.75)

    @since("2.0.0")
    def getTrainRatio(self):
        """
        Gets the value of trainRatio or its default value.
        """
        return self.getOrDefault(self.trainRatio)


class TrainValidationSplit(
    Estimator,
    _TrainValidationSplitParams,
    HasParallelism,
    HasCollectSubModels,
    MLReadable,
    MLWritable,
):
    """
    Validation for hyper-parameter tuning. Randomly splits the input dataset into train and
    validation sets, and uses evaluation metric on the validation set to select the best model.
    Similar to :class:`CrossValidator`, but only splits the set once.
    .. versionadded:: 2.0.0
    Examples
    --------
    >>> from pyspark.ml.classification import LogisticRegression
    >>> from pyspark.ml.evaluation import BinaryClassificationEvaluator
    >>> from pyspark.ml.linalg import Vectors
    >>> from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
    >>> from pyspark.ml.tuning import TrainValidationSplitModel
    >>> import tempfile
    >>> dataset = spark.createDataFrame(
    ...     [(Vectors.dense([0.0]), 0.0),
    ...      (Vectors.dense([0.4]), 1.0),
    ...      (Vectors.dense([0.5]), 0.0),
    ...      (Vectors.dense([0.6]), 1.0),
    ...      (Vectors.dense([1.0]), 1.0)] * 10,
    ...     ["features", "label"]).repartition(1)
    >>> lr = LogisticRegression()
    >>> grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    >>> evaluator = BinaryClassificationEvaluator()
    >>> tvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator,
    ...     parallelism=1, seed=42)
    >>> tvsModel = tvs.fit(dataset)
    >>> tvsModel.getTrainRatio()
    0.75
    >>> tvsModel.validationMetrics
    [0.5, ...
    >>> path = tempfile.mkdtemp()
    >>> model_path = path + "/model"
    >>> tvsModel.write().save(model_path)
    >>> tvsModelRead = TrainValidationSplitModel.read().load(model_path)
    >>> tvsModelRead.validationMetrics
    [0.5, ...
    >>> evaluator.evaluate(tvsModel.transform(dataset))
    0.833...
    >>> evaluator.evaluate(tvsModelRead.transform(dataset))
    0.833...
    """

    @keyword_only
    def __init__(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        trainRatio=0.75,
        parallelism=1,
        collectSubModels=False,
        seed=None,
    ):
        """
        __init__(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, \
                 trainRatio=0.75, parallelism=1, collectSubModels=False, seed=None)
        """
        super(TrainValidationSplit, self).__init__()
        self._setDefault(parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @since("2.0.0")
    @keyword_only
    def setParams(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        trainRatio=0.75,
        parallelism=1,
        collectSubModels=False,
        seed=None,
    ):
        """
        setParams(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, \
                  trainRatio=0.75, parallelism=1, collectSubModels=False, seed=None):
        Sets params for the train validation split.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @since("2.0.0")
    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`estimator`.
        """
        return self._set(estimator=value)

    @since("2.0.0")
    def setEstimatorParamMaps(self, value):
        """
        Sets the value of :py:attr:`estimatorParamMaps`.
        """
        return self._set(estimatorParamMaps=value)

    @since("2.0.0")
    def setEvaluator(self, value):
        """
        Sets the value of :py:attr:`evaluator`.
        """
        return self._set(evaluator=value)

    @since("2.0.0")
    def setTrainRatio(self, value):
        """
        Sets the value of :py:attr:`trainRatio`.
        """
        return self._set(trainRatio=value)

    def setSeed(self, value):
        """
        Sets the value of :py:attr:`seed`.
        """
        return self._set(seed=value)

    def setParallelism(self, value):
        """
        Sets the value of :py:attr:`parallelism`.
        """
        return self._set(parallelism=value)

    def setCollectSubModels(self, value):
        """
        Sets the value of :py:attr:`collectSubModels`.
        """
        return self._set(collectSubModels=value)

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        tRatio = self.getOrDefault(self.trainRatio)
        seed = self.getOrDefault(self.seed)
        randCol = self.uid + "_rand"
        df = dataset.select("*", rand(seed).alias(randCol))
        condition = df[randCol] >= tRatio
        validation = df.filter(condition).cache()
        train = df.filter(~condition).cache()

        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [None for i in range(numModels)]

        tasks = map(
            inheritable_thread_target,
            _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam),
        )
        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        metrics = [None] * numModels
        for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
            metrics[j] = metric
            if collectSubModelsParam:
                subModels[j] = subModel

        train.unpersist()
        validation.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(TrainValidationSplitModel(bestModel, metrics, subModels))

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies creates a deep copy of
        the embedded paramMap, and copies the embedded and extra parameters over.
        .. versionadded:: 2.0.0
        Parameters
        ----------
        extra : dict, optional
            Extra parameters to copy to the new instance
        Returns
        -------
        :py:class:`TrainValidationSplit`
            Copy of this instance
        """
        if extra is None:
            extra = dict()
        newTVS = Params.copy(self, extra)
        if self.isSet(self.estimator):
            newTVS.setEstimator(self.getEstimator().copy(extra))
        # estimatorParamMaps remain the same
        if self.isSet(self.evaluator):
            newTVS.setEvaluator(self.getEvaluator().copy(extra))
        return newTVS

    @since("2.3.0")
    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        if _ValidatorSharedReadWrite.is_java_convertible(self):
            return JavaMLWriter(self)
        return TrainValidationSplitWriter(self)

    @classmethod
    @since("2.3.0")
    def read(cls):
        """Returns an MLReader instance for this class."""
        return TrainValidationSplitReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java TrainValidationSplit, create and return a Python wrapper of it.
        Used for ML persistence.
        """

        estimator, epms, evaluator = super(TrainValidationSplit, cls)._from_java_impl(java_stage)
        trainRatio = java_stage.getTrainRatio()
        seed = java_stage.getSeed()
        parallelism = java_stage.getParallelism()
        collectSubModels = java_stage.getCollectSubModels()
        # Create a new instance of this stage.
        py_stage = cls(
            estimator=estimator,
            estimatorParamMaps=epms,
            evaluator=evaluator,
            trainRatio=trainRatio,
            seed=seed,
            parallelism=parallelism,
            collectSubModels=collectSubModels,
        )
        py_stage._resetUid(java_stage.uid())
        return py_stage

    def _to_java(self):
        """
        Transfer this instance to a Java TrainValidationSplit. Used for ML persistence.
        Returns
        -------
        py4j.java_gateway.JavaObject
            Java object equivalent to this instance.
        """

        estimator, epms, evaluator = super(TrainValidationSplit, self)._to_java_impl()

        _java_obj = JavaParams._new_java_obj(
            "org.apache.spark.ml.tuning.TrainValidationSplit", self.uid
        )
        _java_obj.setEstimatorParamMaps(epms)
        _java_obj.setEvaluator(evaluator)
        _java_obj.setEstimator(estimator)
        _java_obj.setTrainRatio(self.getTrainRatio())
        _java_obj.setSeed(self.getSeed())
        _java_obj.setParallelism(self.getParallelism())
        _java_obj.setCollectSubModels(self.getCollectSubModels())
        return _java_obj


class TrainValidationSplitModel(Model, _TrainValidationSplitParams, MLReadable, MLWritable):
    """
    Model from train validation split.
    .. versionadded:: 2.0.0
    """

    def __init__(self, bestModel, validationMetrics=None, subModels=None):
        super(TrainValidationSplitModel, self).__init__()
        #: best model from train validation split
        self.bestModel = bestModel
        #: evaluated validation metrics
        self.validationMetrics = validationMetrics or []
        #: sub models from train validation split
        self.subModels = subModels

    def _transform(self, dataset):
        return self.bestModel.transform(dataset)

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying bestModel,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.
        And, this creates a shallow copy of the validationMetrics.
        It does not copy the extra Params into the subModels.
        .. versionadded:: 2.0.0
        Parameters
        ----------
        extra : dict, optional
            Extra parameters to copy to the new instance
        Returns
        -------
        :py:class:`TrainValidationSplitModel`
            Copy of this instance
        """
        if extra is None:
            extra = dict()
        bestModel = self.bestModel.copy(extra)
        validationMetrics = list(self.validationMetrics)
        subModels = [model.copy() for model in self.subModels]
        return self._copyValues(
            TrainValidationSplitModel(bestModel, validationMetrics, subModels), extra=extra
        )

    @since("2.3.0")
    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        if _ValidatorSharedReadWrite.is_java_convertible(self):
            return JavaMLWriter(self)
        return TrainValidationSplitModelWriter(self)

    @classmethod
    @since("2.3.0")
    def read(cls):
        """Returns an MLReader instance for this class."""
        return TrainValidationSplitModelReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java TrainValidationSplitModel, create and return a Python wrapper of it.
        Used for ML persistence.
        """

        # Load information from java_stage to the instance.
        sc = SparkContext._active_spark_context
        bestModel = JavaParams._from_java(java_stage.bestModel())
        validationMetrics = _java2py(sc, java_stage.validationMetrics())
        estimator, epms, evaluator = super(TrainValidationSplitModel, cls)._from_java_impl(
            java_stage
        )
        # Create a new instance of this stage.
        py_stage = cls(bestModel=bestModel, validationMetrics=validationMetrics)
        params = {
            "evaluator": evaluator,
            "estimator": estimator,
            "estimatorParamMaps": epms,
            "trainRatio": java_stage.getTrainRatio(),
            "seed": java_stage.getSeed(),
        }
        for param_name, param_val in params.items():
            py_stage = py_stage._set(**{param_name: param_val})

        if java_stage.hasSubModels():
            py_stage.subModels = [
                JavaParams._from_java(sub_model) for sub_model in java_stage.subModels()
            ]

        py_stage._resetUid(java_stage.uid())
        return py_stage

    def _to_java(self):
        """
        Transfer this instance to a Java TrainValidationSplitModel. Used for ML persistence.
        Returns
        -------
        py4j.java_gateway.JavaObject
            Java object equivalent to this instance.
        """

        sc = SparkContext._active_spark_context
        _java_obj = JavaParams._new_java_obj(
            "org.apache.spark.ml.tuning.TrainValidationSplitModel",
            self.uid,
            self.bestModel._to_java(),
            _py2java(sc, self.validationMetrics),
        )
        estimator, epms, evaluator = super(TrainValidationSplitModel, self)._to_java_impl()

        params = {
            "evaluator": evaluator,
            "estimator": estimator,
            "estimatorParamMaps": epms,
            "trainRatio": self.getTrainRatio(),
            "seed": self.getSeed(),
        }
        for param_name, param_val in params.items():
            java_param = _java_obj.getParam(param_name)
            pair = java_param.w(param_val)
            _java_obj.set(pair)

        if self.subModels is not None:
            java_sub_models = [sub_model._to_java() for sub_model in self.subModels]
            _java_obj.setSubModels(java_sub_models)

        return _java_obj

# COMMAND ----------

def compareBaselines(model = None, model_name = None, features = None, paramGrid = None, train_data = train_model_small, validation_data = validation_data_small, test_data = False):  
  '''Baseline model comparison: Similiar to the custom tuning function, this function will take a model, a feature list, training data, validation data. It will train and test the model, appending the modelName, modelObject, featuresList, precision score (i.e. the model precision score) to a list of lists to use for comparison. If model is None, predict never delayed as our 'null hypothesis' comparison.'''
  #If no model is passed, predict the majority class for our validation data (the odd numbered fold numbers in our foldCol). 
  if model is None: 
    #Append 0.0 literal to evaluation data as "prediction". 
    predictions = validation_data.withColumn('prediction_majority', f.lit(0.0))
    f_beta = MulticlassClassificationEvaluator(labelCol='dep_del15', predictionCol='prediction_majority', metricName='f1', beta = 0.5, metricLabel = 1)
    f_beta_score = f_beta.evaluate(predictions)
    #TODO: Calculate actual std. dev in fbeta across folds. 
    stdDev = 0.0
    bestParams = None
    #Note we pass the paramGrid object with the baseline model so that we can easily extract the paramGrid to use for best model. 
    return [model_name, model, features, f_beta_score, stdDev, paramGrid, bestParams]
  else:
    pipeline = Pipeline(stages=[model])
    f_beta = MulticlassClassificationEvaluator(labelCol='dep_del15', predictionCol='prediction', metricName='f1', beta = 0.5, metricLabel = 1)
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=f_beta, numFolds = 10, parallelism=35, foldCol = 'foldNumber', collectSubModels = False)
    cvModel = cv.fit(train_data)
    bestModel = cvModel.bestModel
    # Get average of performance metric (F-beta) for best model 
    f_beta_score = cvModel.avgMetrics[0]
    #Get standard deviation of performance metric (F-beta) for best model
    stdDev = cvModel.stdMetrics[0]
    #Get the best params
    bestParams = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]
    return [model_name, cvModel, features, f_beta_score, stdDev, paramGrid, bestParams]

# COMMAND ----------

def statisticalTestModels(df = None): 
  '''Takes a dateframe, sorted by increasing f_beta scores. Conducts two sample welch t-test for unequal variances
  between two rows to determine if the f_beta score is significantly higher than previous row, or due to chance.'''
  prev_fbeta = None
  prev_std = None 
  p_value_to_prev = []

  for index, row in df.iterrows(): 
    if index > 0: 
      #Update current row values
      current_fbeta = row['f_beta_score']
      current_std = row['f_beta_std_dev']
      #Two sample welch t-test for unequal variances
      p_value = stats.ttest_ind_from_stats(mean1 = current_fbeta, std1 = current_std, nobs1 = 10, mean2 = prev_fbeta, std2 = prev_std, nobs2 = 10, equal_var = False, alternative = 'greater').pvalue
      p_value_to_prev.append(p_value)
    else: 
      # Append null if on first row
      p_value_to_prev.append(None)

    #Update the previous row values
    prev_fbeta = row['f_beta_score']
    prev_std = row['f_beta_std_dev']


  df['p_value_to_prev'] = p_value_to_prev
  return df

# COMMAND ----------

# MAGIC %md 2.) Define baseline models and their hyperparam grids for non-random grid search. Consistent hyperparams across models ensures valid experimental framework. 

# COMMAND ----------

# Manually define non-random search space. This ensures data and params are controlled for in our experimental comparisons, yielding valid results. 
MIN_INFO_GAIN_SEARCH_LIST = [0.0, 0.2]
MAX_DEPTH_SEARCH_LIST = [2, 4]

### 1.) Decision Tree Classifier ###
dt_model = DecisionTreeClassifier(featuresCol = 'features', labelCol='dep_del15')

# DT Param Grid
dt_paramGrid = ParamGridBuilder() \
    .addGrid(dt_model.minInfoGain, MIN_INFO_GAIN_SEARCH_LIST) \
    .addGrid(dt_model.maxDepth, MAX_DEPTH_SEARCH_LIST) \
    .build()


### 2.) Gradient Boosted Tree Classifier ###
gbt_model = GBTClassifier(featuresCol = 'features', labelCol='dep_del15', maxIter = 20)

# GBT Param Grid
# gbt_paramGrid = ParamGridBuilder() \
#     .addGrid(gbt_model.stepSize, random.sample(list(np.linspace(0.1,1,11)), NUM_PARAM_VALUES)) \
#     .build()

gbt_paramGrid = ParamGridBuilder() \
    .addGrid(gbt_model.minInfoGain, MIN_INFO_GAIN_SEARCH_LIST) \
    .addGrid(gbt_model.maxDepth, MAX_DEPTH_SEARCH_LIST) \
    .build()


### 3.) Random Forest Classifier ###
rf_model = RandomForestClassifier(featuresCol = 'features', labelCol='dep_del15')

# RF Param Grid
rf_paramGrid = ParamGridBuilder() \
    .addGrid(rf_model.minInfoGain, MIN_INFO_GAIN_SEARCH_LIST) \
    .addGrid(rf_model.maxDepth, MAX_DEPTH_SEARCH_LIST) \
    .build()

             
modelList = [(None, "Majority Class", None), 
             (dt_model, "Decision Tree", dt_paramGrid),
             (gbt_model, "Gradient Boosted Tree", gbt_paramGrid),
             (rf_model, "Random Forest", rf_paramGrid)]

# COMMAND ----------

# MAGIC %md 3.) Loop over baseline models and build comparison of performance / features. 

# COMMAND ----------

#Create an empty list of lists that we will append models & performance metrics to.
# Data order will be: model_name[str], model[obj], features[list], f_beta_score[float], f_beta_std_dev[float], paramGrid [obj] 
modelComparisons = []

#Build comparison table. 
for model, model_name, paramGrid in modelList: 
  modelComparisons.append(compareBaselines(model = model, model_name = model_name, paramGrid = paramGrid))

#model_name[str], model[obj], features[list], precision[float]
modelComparisonsDF = pd.DataFrame(modelComparisons, columns = ['model_name', 'model_obj','feature_names','f_beta_score', 'f_beta_std_dev', 'paramGrid_obj', 'bestParams']).sort_values(by = 'f_beta_score').reset_index(drop=True)

# Show results
modelComparisonsDF['bestParams']

# COMMAND ----------

# MAGIC %md 4.) Statistical testing of differences in model performance. 

# COMMAND ----------

modelComparisonsDF = statisticalTestModels(modelComparisonsDF)
modelComparisonsDF

# COMMAND ----------

# MAGIC %md 5.) Plot the F-beta score of each of our models / model-runs in both a Dataframe and a visual to use in our presentation. Use this step to confirm everything in train/val is working as intended and we are comfortable with our final model and hyperparams. Look for signs of overfitting, etc. 

# COMMAND ----------

#plotCols = modelComparisonsDF[['model_name','precision_score']]
x = modelComparisonsDF['model_name']
y = modelComparisonsDF['f_beta_score']

plt.bar(x, y)
for x,y in zip(x,y): 
  label = "{:.3f}".format(y)
  plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,15), # distance from text to points (x,y)
                 ha='center')
plt.xticks(rotation = 90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Algorithm Implementation
# MAGIC Create your own toy example that matches the dataset provided and use this toy example to explain the math behind the algorithm that you will perform. Apply your algorithm to the training dataset and evaluate your results on the test set. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Tree Explanation and Toy Example:  
# MAGIC [Source](https://towardsdatascience.com/a-dive-into-decision-trees-a128923c9298)
# MAGIC 
# MAGIC Decision Trees operate by selecting the optimal features to split the data. The data is split either until a pre-definened parameter is met, like maximum depth or minimum information gain, or until the data is perfectly split. The nodes of a tree represent a feature the data is split on, each node will have a branch that either leads to another node where another split takes place, or to a leaf which represents the decision the tree arrives at. Decision Trees can be used in both classification and regression, in this project we are working on classifying if a plane is delayed greater than 15 minutes or not, so we will focus on classification.
# MAGIC 
# MAGIC When a decision tree splits the data it must determine which feature to split and how to split it optimally. For a split to be optimal, it must maximize the possible information gain. In spark the default implemenation calulates the gini impurity to measure how optimal a split is. The lower the gini impurity, the higher the information gain and the better the split. A pure split with all 0's in one leaf and all 1's in another would have a gini impurity of 0. A perfectly impure split, which results in an equal number of each class in each branch would result in the maximum gini impurity of 0.5 which represents no information gain. 
# MAGIC 
# MAGIC The equation for gini impurity is:  
# MAGIC \\(gini\ impurity = 1 - P(class\ 1)^2 - P(class\ 2)^2 \\)
# MAGIC 
# MAGIC The above equation is used to determine how to split a given feature optimally. In the case of a categorical feature, the gini impurity is calculated for each branch on each possible split of the feature, we then take the weighted average gini impurity across both branches to determine the overall gini impurity for each split. Below is the equation for the weighted average of gini impurity.
# MAGIC 
# MAGIC \\(weighted\ average = \frac{branch\ 1\ count}{total\ count}(branch\ 1\ gini\ impurity) + \frac{branch\ 2\ count}{total\ count}(branch\ 2\ gini\ impurity) \\)

# COMMAND ----------

# create a small example data frame
toy_df = pd.DataFrame({'class': [0, 0, 1, 1, 1, 1, 0, 0], 'categorical_feature': ['a', 'a', 'b', 'c', 'b', 'c', 'b', 'a'], 'continuous_feature': [26, 17, 12, 8, 1, 14, 2, 10]})
print('original dataframe')
print(toy_df)

# sort by categorical feature
toy_df.sort_values(by='categorical_feature', axis=0, inplace=True)
print('')
print('sorted dataframe')
print(toy_df)

# define gini funciton
def gini(c1, c2):
  """Returns Gini impurity given class counts"""
  gini = 1-(c1/(c1+c2))**2 - (c2/(c1+c2))**2
  return gini
# define weighted average gini function
def cumulative_gini(c1a, c2a, c1b, c2b):
  total = c1a + c2a + c1b + c2b
  ginia = gini(c1a,c2a) * ((c1a+c2a)/total)
  ginib = gini(c1b,c2b) * ((c1b+c2b)/total)
  return ginia + ginib

print('Split on "a"')
print('Branch 1:')
# split data frame on categorical feature, split feature on 'a'
# get class 0 count for branch 1
split_condition = toy_df.apply(lambda x: True if x['categorical_feature'] == 'a' and x['class'] == 0 else False, axis=1)
b1_class_0_count = len(split_condition[split_condition == True])

print(f'Branch 1, Class 0 count: {b1_class_0_count}')
# get class 1 count for branch 1
split_condition = toy_df.apply(lambda x: True if x['categorical_feature'] == 'a' and x['class'] == 1 else False, axis=1)
b1_class_1_count = len(split_condition[split_condition == True])
print(f'Branch 1, Class 1 count: {b1_class_1_count}')
print(f'Branch 1 Gini Impurity: {gini(b1_class_0_count, b1_class_1_count)}')
print('---')
print('Branch 2:')
# split data frame on categorical feature, split feature on 'a'
# get class 0 count for branch 2
split_condition = toy_df.apply(lambda x: True if x['categorical_feature'] != 'a' and x['class'] == 0 else False, axis=1)
b2_class_0_count = len(split_condition[split_condition == True])
print(f'Branch 2, Class 0 count: {b2_class_0_count}')
# get class 1 count for branch 1
split_condition = toy_df.apply(lambda x: True if x['categorical_feature'] != 'a' and x['class'] == 1 else False, axis=1)
b2_class_1_count = len(split_condition[split_condition == True])
print(f'Branch 2, Class 1 count: {b2_class_1_count}')
print(f'Branch 2 Gini Impurity: {gini(b2_class_0_count, b2_class_1_count)}')
print(f'Weighted Gini Impurity for spliting categorical_feature on "a": {cumulative_gini(b1_class_0_count, b1_class_1_count, b2_class_0_count, b2_class_1_count)}')
print('\n')

print('Split on "b"')
print('Branch 1:')
# split data frame on categorical feature, split feature on 'a'
# get class 0 count for branch 1
split_condition = toy_df.apply(lambda x: True if x['categorical_feature'] != 'c' and x['class'] == 0 else False, axis=1)
b1_class_0_count = len(split_condition[split_condition == True])
print(f'Branch 1, Class 0 count: {b1_class_0_count}')
# get class 1 count for branch 1
split_condition = toy_df.apply(lambda x: True if x['categorical_feature'] != 'c' and x['class'] == 1 else False, axis=1)
b1_class_1_count = len(split_condition[split_condition == True])
print(f'Branch 1, Class 1 count: {b1_class_1_count}')
print(f'Branch 1 Gini Impurity: {gini(b1_class_0_count, b1_class_1_count)}')
print('---')
print('Branch 2:')
# split data frame on categorical feature, split feature on 'a'
# get class 0 count for branch 2
split_condition = toy_df.apply(lambda x: True if x['categorical_feature'] == 'c' and x['class'] == 0 else False, axis=1)
b2_class_0_count = len(split_condition[split_condition == True])
print(f'Branch 2, Class 0 count: {b2_class_0_count}')
# get class 1 count for branch 1
split_condition = toy_df.apply(lambda x: True if x['categorical_feature'] == 'c' and x['class'] == 1 else False, axis=1)
b2_class_1_count = len(split_condition[split_condition == True])
print(f'Branch 2, Class 1 count: {b2_class_1_count}')
print(f'Branch 2 Gini Impurity: {gini(b2_class_0_count, b2_class_1_count)}')
print(f'Weighted Gini Impurity for spliting categorical_feature on "b": {cumulative_gini(b1_class_0_count, b1_class_1_count, b2_class_0_count, b2_class_1_count)}')

# COMMAND ----------

# MAGIC %md
# MAGIC When we split on a continuous feature, the dataframe is sorted by the categorical feature, and the average of every two values is taken. These average values are then consitered as potential split points and the gini impurity is calculated at each potential split point. Below is an example of this.

# COMMAND ----------

# create a small example data frame
toy_df = pd.DataFrame({'class': [0, 0, 1, 1, 1, 1, 0, 0], 'categorical_feature': ['a', 'a', 'b', 'c', 'b', 'c', 'b', 'a'], 'continuous_feature': [26, 17, 12, 8, 1, 14, 2, 10]})
print('original dataframe')
print(toy_df)

# sort by continuosus feature
toy_df.sort_values(by='continuous_feature', ascending=False, axis=0, inplace=True)
print('')
print('continuous feature sorted dataframe')
print(toy_df)

# calculate averages
means = []
for i in range(len(toy_df.continuous_feature)-1):
  avg_val =  (list(toy_df.continuous_feature)[i] + list(toy_df.continuous_feature)[i+1]) / 2
  means.append(avg_val)
means.append(0)
toy_df['means'] = means
print('dataframe with means of each pair (shifted up by 1)')
print(toy_df)

for i in list(toy_df.means[:-1]):
  print(f'Split on {i}')
  print('Branch 1:')
  # split data frame on continuous feature
  # get class 0 count for branch 1
  split_condition = toy_df.apply(lambda x: True if x['continuous_feature'] >= i and x['class'] == 0 else False, axis=1)
  b1_class_0_count = len(split_condition[split_condition == True])
  print(f'Branch 1, Class 0 count: {b1_class_0_count}')
  # get class 1 count for branch 1
  split_condition = toy_df.apply(lambda x: True if x['continuous_feature'] >= i and x['class'] == 1 else False, axis=1)
  b1_class_1_count = len(split_condition[split_condition == True])
  print(f'Branch 1, Class 1 count: {b1_class_1_count}')
  print(f'Branch 1 Gini Impurity: {gini(b1_class_0_count, b1_class_1_count)}')
  print('Branch 2:')
  # split data frame on categorical feature, split feature on 'a'
  # get class 0 count for branch 2
  split_condition = toy_df.apply(lambda x: True if x['continuous_feature'] < i and x['class'] == 0 else False, axis=1)
  b2_class_0_count = len(split_condition[split_condition == True])
  print(f'Branch 2, Class 0 count: {b2_class_0_count}')
  # get class 1 count for branch 1
  split_condition = toy_df.apply(lambda x: True if x['continuous_feature'] < i and x['class'] == 1 else False, axis=1)
  b2_class_1_count = len(split_condition[split_condition == True])
  print(f'Branch 2, Class 1 count: {b2_class_1_count}')
  print(f'Branch 2 Gini Impurity: {gini(b2_class_0_count, b2_class_1_count)}')
  print(f'Weighted Gini Impurity for spliting categorical_feature on {i}: {cumulative_gini(b1_class_0_count, b1_class_1_count, b2_class_0_count, b2_class_1_count)}')
  print('---')

# COMMAND ----------

# MAGIC %md
# MAGIC After running using the equations detailed above to find the weighted gini impurity at each potential split point of both the categorical and continuous variable, we can see that the optimal split point is the categorical feature, split on 'a' because it results in the lowest gini impurity of 0.199. As a result this feature and split point will be the root (the first split point) of the decision tree. This process will be repeated at each node until 1) there is a pure split, 2) we reach a maximum depth specified as a hyperparameter, or 3) we do not meat a minimum information gain threshold specified as a hyperparameter. Once we meet one of these 3 conditions, the final node on the tree branches becomes a 'leaf' which is what the decision tree will use to make its final decisions.
# MAGIC   
# MAGIC   In the context of classification, a decision tree will return the most probable answer on the leaf it ends up on. In our example above, if we had a decision tree that has only one split on the categorical feature at point 'a' then our branches would have the following counts:
# MAGIC   
# MAGIC   Leaf 1  
# MAGIC   class 0 count: 3  
# MAGIC   class 1 count: 0
# MAGIC   
# MAGIC   Leaf 2  
# MAGIC   class 0 count: 1  
# MAGIC   class 1 count: 4
# MAGIC   
# MAGIC   If we are on leaf 1 of this tree, we will predict class 0 because it has a probability of \\(\frac{3}{3} = 1.0\\) vs the probability of class 1 of \\(\frac{0}{3} = 0.0\\)  
# MAGIC   
# MAGIC   
# MAGIC   If we are on leaf 1 of this tree, we will predict class 1 because it has a probability of \\(\frac{4}{5} = .80\\) vs the probability of class 0 of \\(\frac{1}{5} = .20\\)  

# COMMAND ----------

# MAGIC %md # Final Model Selection and Tuning. 
# MAGIC 
# MAGIC Do a random depth-search across our best experimental model, on the full training/validation data to ensure optimal configuration for final train/test. 

# COMMAND ----------

# Logic to select the most performant model along both F-beta and runtime. 
prev_fbeta = None
alpha_sig_value = 0.05
prev_runtime = None
best_model_index = None

for index, row in modelComparisonsDF.iterrows(): 
  if index > 0: 
    #Update current row values
    current_fbeta = row['f_beta_score']
    #Current p_value
    current_p_value = row['p_value_to_prev']
    #TODO: Substitute runtime column
    current_runtime = None
    #If performance is better, with constant or better runtime, select model regardless of significance. 
    #if current_fbeta > prev_fbeta and current_runtime <= prev_runtime: 
    #  best_model_index = index
    #If performance is significantly better, select model regardless of runtime. 
    if current_fbeta > prev_fbeta and current_p_value < alpha_sig_value: 
      best_model_index = index
  else: 
    # Append null if on first row
    best_model_index = index

  #Update the previous row values
  prev_fbeta = row['f_beta_score']
  prev_runtime = None


best_model = modelComparisonsDF["model_obj"][best_model_index]
best_model_name = modelComparisonsDF["model_name"][best_model_index]

best_model_name

# COMMAND ----------

# MAGIC %md Define the depth-search paramGrids for each of 3 models so the final tuning is automated from experimentation. 

# COMMAND ----------

NUM_PARAM_VALUES = 3

# COMMAND ----------

# MAGIC %md Redefine models, but this time with a random search space, over more values. 

# COMMAND ----------

### 1.) Decision Tree Classifier ###
dt_model = DecisionTreeClassifier(featuresCol = 'features', labelCol='dep_del15')

# DT Param Grid
final_paramGrid = ParamGridBuilder() \
    .addGrid(dt_model.minInfoGain, random.sample(list(np.linspace(0,4,5)), NUM_PARAM_VALUES)) \
    .addGrid(dt_model.maxDepth, random.sample(list(np.linspace(0, MAX_TREE_DEPTH, MAX_TREE_DEPTH + 1)), NUM_PARAM_VALUES)) \
    .build()


### 2.) Gradient Boosted Tree Classifier ###
gbt_model = GBTClassifier(featuresCol = 'features', labelCol='dep_del15', maxIter = 20)

# GBT Param Grid
gbt_paramGrid = ParamGridBuilder() \
    .addGrid(gbt_model.minInfoGain, random.sample(list(np.linspace(0,4,5)), NUM_PARAM_VALUES)) \
    .addGrid(gbt_model.maxDepth, random.sample(list(np.linspace(0, MAX_TREE_DEPTH, MAX_TREE_DEPTH + 1)), NUM_PARAM_VALUES)) \
    .build()


### 3.) Random Forest Classifier ###
rf_model = RandomForestClassifier(featuresCol = 'features', labelCol='dep_del15')

# RF Param Grid
rf_paramGrid = ParamGridBuilder() \
    .addGrid(gbt_model.minInfoGain, random.sample(list(np.linspace(0,6,7)), NUM_PARAM_VALUES)) \
    .addGrid(gbt_model.maxDepth, random.sample(list(np.linspace(0, MAX_TREE_DEPTH, MAX_TREE_DEPTH + 1)), NUM_PARAM_VALUES)) \
    .build()


# COMMAND ----------

best_model_name

# COMMAND ----------

# Build our final model to do random depth-search. 
if best_model_name == 'Random Forest': 
  best_model = rf_model
  best_paramGrid = rf_paramGrid
elif best_model_name == 'Decision Tree': 
  best_model = dt_model
  best_paramGrid = dt_paramGrid
elif best_model_name == 'Gradient Boosted Tree': 
  best_model = gbt_model
  best_paramGrid = gbt_paramGrid

bestModelList = [(best_model, best_model_name + ' Full Model', best_paramGrid)]

# COMMAND ----------



# COMMAND ----------

#Build comparison table. 
for model, model_name, paramGrid in bestModelList: 
  model_name, bestModel, features, f_beta_score, stdDev, paramGrid, bestParams = compareBaselines(model = model, model_name = model_name, paramGrid = paramGrid, train_data = train_model_small)
  a_series = pd.Series([model_name, bestModel, features, f_beta_score, stdDev, paramGrid, bestParams, None], index = modelComparisonsDF.columns)
  modelComparisonsDF = modelComparisonsDF.append(a_series, ignore_index=True)

modelComparisonsDF

# COMMAND ----------

# MAGIC %md Take the best model, now fully tuned on the full dataset. Fit on the full trainset and test on the full testset. 

# COMMAND ----------

final_params = bestParams
test_predictions = bestModel.transform(test_model)
f_beta = MulticlassClassificationEvaluator(labelCol='dep_del15', predictionCol='prediction', metricName='f1', beta = 0.5, metricLabel = 1)
final_f_beta_score = f_beta.evaluate(test_predictions)

a_series = pd.Series([model_name + ' Test Results', bestModel, features, final_f_beta_score, None, None, bestParams, None], index = modelComparisonsDF.columns)
modelComparisonsDF = modelComparisonsDF.append(a_series, ignore_index=True)
modelComparisonsDF

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


