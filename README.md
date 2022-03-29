# Time-Series-Forecasting-with-Tensorflow-and-InfluxDB

To install influxdb-client library in python you can use python package manager(PIP).

```
## install using terminal
$ pip install tensorflow

## install using jupyter-notebook
! pip install tensorflow
```

# Making a Connection

Parameters that must be known to make DB connection are:
 - **Token:** Access token that you have generated using the console, you can log in to the influxDb dashboard and copy the same.
 - **Bucket:** This requires the name of the bucket on which you would be working. You can choose the initial bucket or create a new one using the dashboard.
 - **Organisation:** Organisation that you have named during the initial setup of the influx DB.
Since this connection would be made locally the connection script would look like this:

```
## import dependencies
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

## You can generate a Token from the "Tokens Tab" in the UI
token = "your-token"
org = "my-org"
bucket = "Testing"

## connect to influxdb 
client = InfluxDBClient(url="http://localhost:8086", token=token, org = org)
```

# Inserting Data 

To insert the data in inlfuxdb using python you need to create an object of the **write API**. 
```
## create object of write API
write_api = client.write_api(write_options=SYNCHRONOUS)
```
Here synchronous specifies that you would be storing multiple rows of data at a time. Now you just need to pass your dataframe to this write object.

```
## write data to influxdb
response = write_api.write(bucket, record = data, data_frame_measurement_name='sunspot',
                        data_frame_tag_columns=['sunspot'])
```

# Reading Data

To query the data from influxdb you need to create an object of **read API**. [Flux-Query](https://docs.influxdata.com/influxdb/cloud/query-data/get-started/) is the easiest way to query the data from the database. You just need to specify the period for which you need to select the data from the database. 

```
## query data
query_api = client.query_api()
tables = query_api.query('from(bucket:"Testing") |> range(start: -275y)')
```

To explore InfluxDB in detail you can refer the following [link](https://docs.influxdata.com/influxdb/v2.0/).
