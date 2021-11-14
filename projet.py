from json import load
from numpy.random.mtrand import f
import streamlit as st
# importing numpy and pandas for to work with sample data.
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import datetime
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px


import logging, random
import time
import streamlit.components.v1 as components
import base64

from functools import wraps

############ CONFIG ############
st.set_page_config(layout="wide")
logger = logging.getLogger(__name__)


# Misc logger setup so a debug log statement gets printed on stdout.
logger.setLevel("INFO")
handler = logging.FileHandler(filename="log.txt", mode="a")
log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.info('\nNew Execution at :', time.time(), "\n")

############ END CONFIG ############

############ FUNCTIONS ############
def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper

@timed
@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist=True)
def load_data_raw(url, delimiter):
    data=pd.read_csv(url, delimiter)

    data["Date/Time"] = pd.to_datetime(data["Date/Time"])

    data['day'] = data['Date/Time'].map(get_dom)

    data['weekday'] = data['Date/Time'].map(get_weekday)

    data['hour'] = data['Date/Time'].map(get_hour)

    data['minute'] = data['Date/Time'].map(get_minute)

    return data

@timed
@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist=True)
def load_data_ny(url, delimiter):

    data = pd.read_csv(url, delimiter)

    data['tpep_pickup_datetime'] = data['tpep_pickup_datetime'].map(pd.to_datetime)
    data['tpep_dropoff_datetime'] = data['tpep_dropoff_datetime'].map(pd.to_datetime)

    #Pickup datetime transformation and insertion in new column using get_... functions
    data['day_pickup'] = data['tpep_pickup_datetime'].map(get_dom)

    data['weekday_pickup'] = data['tpep_pickup_datetime'].map(get_weekday)

    data['hour_pickup'] = data['tpep_pickup_datetime'].map(get_hour)

    data['minute_pickup'] = data['tpep_pickup_datetime'].map(get_minute)


    #Dropoff datetime transformation and insertion in new column using get_... functions
    data['day_dropoff'] = data['tpep_dropoff_datetime'].map(get_dom)

    data['weekday_dropoff'] = data['tpep_dropoff_datetime'].map(get_weekday)

    data['hour_dropoff'] = data['tpep_dropoff_datetime'].map(get_hour)

    data['minute_dropoff'] = data['tpep_dropoff_datetime'].map(get_minute)

    data['trip_duration'] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime'])
    #Create a new column 'trip_duration' by substract dropoff datetime by pickup datetime and compute it in minutes
    data['trip_duration'] = pd.to_timedelta(data['trip_duration']).dt.total_seconds()/60

    #Create a column 'average_speed' by dividing trip distance by trip duration in hour
    data['average_speed'] = data["trip_distance"] / (data["trip_duration"] / 60) 

    return data

@st.cache(allow_output_mutation=True)
def dflalon(df):
    dflalon = df[['latitude','longitude']]
    return dflalon

def get_month(dt):
    return dt.month

def get_year(dt):
    return dt.year

def get_dom(dt):
    return dt.day

def get_weekday(dt):
    return dt.weekday()

def get_hour(dt):
    return dt.hour

def get_minute(dt):
    return dt.minute


def count_rows(rows):
    return len(rows)

@timed
def map(data, lat, lon, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["Lon", "Lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))

############### START PROJECTS FUNCTIONS #########
@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist=True)
def load_data(url, delimiter, p):
    df = pd.read_csv(url,delimiter,skiprows=lambda i: i>0 and random.random() > p)
    #df = pd.read_csv(url, delimiter)

    return df


@timed
@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist=True)
def load_all_data(percent):
  #df_full = load_data('full.csv', ',')

  df_2016 = load_data('full_2016.csv', ',', percent)
  #df_2017 = load_data('full_2017.csv', ',')
  #df_2018 = load_data('full_2018.csv', ',')
  #df_2019 = load_data('full_2019.csv', ',')
  #df_2020 = load_data('full_2020.csv', ',')

  #frames = [df_2016, df_2017, df_2018, df_2019, df_2020]
  #frames = [df_2016, df_2017]

  #df = pd.concat(frames)
  df = df_2016

  #Pre-process data
  df_2016['date_mutation'] = df_2016['date_mutation'].map(pd.to_datetime)
  df_2016['code_departement'] = df_2016['code_departement'].map(str)

  df['day'] = df['date_mutation'].map(get_dom)
  df['month'] = df['date_mutation'].map(get_month)
  df['year'] = df['date_mutation'].map(get_year)

  return df_2016


############ END FUNTIONS ############

############ START EXECUTION #########

df=load_data_raw('uber-raw-data-apr14.csv', ',')
dfNy = load_data_ny("ny-trips-data.csv",',')

df_RE = load_all_data(0.05)


file_ = open("chart1.png", "rb")
contents = file_.read()
data_url_chart1 = base64.b64encode(contents).decode("utf-8")
file_.close()

file_ = open("chart2.png", "rb")
contents = file_.read()
data_url_chart2 = base64.b64encode(contents).decode("utf-8")
file_.close()



############ END EXECUTION #########

############ START FORMATTING #########



option = 'Uber April 2014'
st.write(df_RE.head())
st.write(df_RE.tail())

if option == 'Uber April 2014':


    st.markdown("<h1 style='text-align: center;font-family=\'Helvetica\';'>PROJECT - Valeurs foncières Visualization 👁</h1>", unsafe_allow_html=True)

    st.markdown('This is a dashboard to visualize relevant datas and charts about the Real Estate values between 2016 and 2020.')
    st.markdown('***')

    #Expander Real Estate Dataset Values
    expander = st.expander("Real Estate Values - 2016")
    col1, col2 = expander.columns(2)
    col1.metric("Nombre de lignes", df_RE.shape[0])
    col2.metric("Nombre de colonnes", df_RE.shape[1])
    expander.write(df_RE.head())
    expander.write(df_RE.tail())





    #Date Input - 2 Columns
    st.markdown("<h2 style='text-align: center;font-family=\'Helvetica\';'><b>Visualization between two dates 🗓</b></h2><br/>", unsafe_allow_html=True)

    st.write("You can choose the starting and ending date to visualize frequency by Day of the Month")
    left_column2, right_column2 = st.columns(2)

    start_date = left_column2.date_input(
        "Starting Date",
        datetime.date(2016, 1, 1))

    end_date = right_column2.date_input(
        "Ending Date",
        datetime.date(2020, 12, 30))

    st.markdown("<br/>", unsafe_allow_html=True)

    #Filtered data for chart "Frequency by day of Month"
    in_interval = (df_RE['day'] >= start_date.day) & (df_RE['day'] <= end_date.day)
    filtered_data = df_RE[in_interval]
    by_id = filtered_data.groupby(['id_mutation'])
    filtered_data = filtered_data.groupby('day').apply(count_rows)

    #Chart by Day and Hours
    st.markdown("<h4 style='text-align: center;font-family=\'Helvetica\';'><b>Frequency by DoM - Valeurs foncières</b></h4>", unsafe_allow_html=True)
    st.bar_chart(filtered_data)
    st.markdown('***')


    #Filtered data for chart "Price per department"
    price_in_interval = (df_RE['day'] >= start_date.day) & (df_RE['day'] <= end_date.day)
    price_filtered_data = df_RE[price_in_interval]
    price_filtered_data = price_filtered_data.groupby(['id_mutation', 'valeur_fonciere', 'code_departement'], as_index=False).first()
    price_by_id = price_filtered_data.groupby('code_departement').sum()['valeur_fonciere']



    #Chart show price by code_departement
    st.markdown("<h4 style='text-align: center;margin-left: -10%;font-family=\'Helvetica\';'><b>Sum of transactions by department - Valeurs foncières - Euros 💶</b></h4>", unsafe_allow_html=True)
    st.bar_chart(price_by_id)
    st.markdown('***')



    #External plot (using plotly) - N°1
    #Filtered data for chart "Mean per departement"
    mean_price_by_department = price_filtered_data.groupby('code_departement').mean()['valeur_fonciere']

    #Chart of "Mean per department"
    fig = px.bar(mean_price_by_department)
    
    st.markdown("<h4 style='text-align: center;margin-left: -10%;font-family=\'Helvetica\';'><b>Mean of transaction by department - Euros 💶</b></h4>", unsafe_allow_html=True)

    fig.update_layout(width=1300,
                    xaxis_title="Department",
                    yaxis_title="Mean of transactions",)
    st.plotly_chart(fig)
    st.markdown('***')


    

    #External plot (using plotly) - N°2
    #Slider
    max_price_df = df_RE.groupby(['id_mutation', 'valeur_fonciere', 'code_departement'], as_index=False).first()
    max_price_df = int(max_price_df['valeur_fonciere'].max())
    price_slider = st.slider(
    "Select a range of price to visualize number of transaction in this range: ",
    0, 
    1000000,
    step=10000, 
    value=(250000, 750000))

    #Column spliter for data linked to slider range price
    left_col_price, right_col_price = st.columns(2)

    #Filtered data for chart "Number of transaction in range"
    slider_price_interval = (df_RE['valeur_fonciere'] >= price_slider[0]) & (df_RE['valeur_fonciere'] <= price_slider[1])
    transactions_slider_price_data = df_RE[slider_price_interval]
    transactions_slider_price__filtered_data = transactions_slider_price_data.groupby(['id_mutation', 'valeur_fonciere', 'code_departement'], as_index=False).first()
    transaction_data_in_slider = transactions_slider_price__filtered_data.groupby('code_departement').apply(count_rows)
    
    #Chart of "Number of transaction in price range"
    fig = px.bar(transaction_data_in_slider)
    
    left_col_price.markdown("<h4 style='text-align: center;margin-left: -10%;font-family=\'Helvetica\';'><b>Number of transactions by department in the selected price range - Euros 💶</b></h4>", unsafe_allow_html=True)

    fig.update_layout(width=700,
                    xaxis_title="Department",
                    yaxis_title="Number of transactions",)
    left_col_price.plotly_chart(fig)


    #External plot (using plotly) - N°3
    #Filtered data for chart "Repartition of real estate's types transactions"
    labels=['House',"Apartment","Dependency","Industrial and commercial premises"]
    values = transactions_slider_price_data['type_local'].value_counts()

    #Pie chart of Mutations' types repartition
    right_col_price.markdown("<h4 style='text-align: center;margin-left: -10%;font-family=\'Helvetica\';'><b>Repartition of real estate's types transactions in the selected price range</b></h4>", unsafe_allow_html=True)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(width=700)
    right_col_price.plotly_chart(fig)

    st.markdown("***")



    #External plot (using plotly) - N°4 (last)
    #Map of all transactions in metropolitain france
    st.markdown("<h4 style='text-align: center;margin-left: -10%;font-family=\'Helvetica\';'><b>HeatMap of transactions across France 🇫🇷</b></h4>", unsafe_allow_html=True)

    fig = px.density_mapbox(dflalon(df_RE), lat='latitude', lon='longitude',center=dict(lat=47.00, lon=2.19), zoom=5, radius=1)
    fig.update_layout(width=1200, height=800, mapbox_style="open-street-map")
    st.plotly_chart(fig)





    xxxx = '''
    st.markdown("<h2 style='text-align: center;font-family=\'Helvetica\';'><b>Visualization by hour(s) and by date(s) 🕦</b></h2><br/>", unsafe_allow_html=True)

    
    #Breakdown of rides per minute between 0:00 and 1:00
    left_column1, right_column1 = st.columns(2)

    breakdown = right_column1.slider(
    "Select time range (of one hour, or more..) to visualize breakdown of rides :",
    0, 
    23, 
    value=(8, 22))

    breakdown_hour_start = breakdown[0]
    breakdown_hour_end = breakdown[1]

    selected_day = 1
    breakdown_day = left_column1.date_input(
        "Breakdown day :",
        datetime.date(2014, 1, 4), 
        datetime.date(2014, 1, 4), 
        datetime.date(2014, 1, 30))


    filter_day_hour = (df['day'] == breakdown_day.day) & (df['hour'] >= breakdown_hour_start) & (df['hour'] <= breakdown_hour_end)
    filtered_day_hour = df[filter_day_hour]
    filtered_day_hour = filtered_day_hour.groupby('minute').apply(count_rows)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;font-family=\'Helvetica\';'><b>" + str("Breakdown of rides on the " + str(breakdown_day) + "/01/2014 per minute between " + str(breakdown_hour_start) + ":00 and " + str(breakdown_hour_end) + ":00.") + "</b></h4>", unsafe_allow_html=True)

    st.bar_chart(filtered_day_hour)

    data = df[df["Date/Time"].dt.hour == breakdown_hour_start][["Lon", "Lat"]]

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;font-family=\'Helvetica\';'><b>" + str("All New-York Pick-up points on" + str(breakdown_day) +  "/01/2014 between " + str(breakdown_hour_start) + ":00 and " + str(breakdown_hour_end) + ":00.") + "</b></h4>", unsafe_allow_html=True)

    midpoint = (np.average(data["Lat"]), np.average(data["Lon"]))
    map(data, midpoint[0], midpoint[1], 11)

else: 
    st.markdown("<h1 style='text-align: center;font-family=\'Helvetica\';'>PART 3 - New-York Uber trips Visualization on 15/01/2015 🗽</h1>", unsafe_allow_html=True)

    st.markdown('This is a dashboard to visualize relevant datas and charts about Ubers\'s trips on 15/01/2015 in New-York')
    st.markdown('***')

    #Expander Uber 2014 Dataset
    expander = st.expander("Uber Trips 15/01/2015 Dataset")
    col1, col2 = expander.columns(2)
    col1.metric("Nombre de lignes", dfNy.shape[0])
    col2.metric("Nombre de colonnes", dfNy.shape[1])
    expander.write(dfNy.head())

    

    pickup_hour = st.slider(
    "Select time range (of one hour, or more..) to visualize Pickup frequency :",
    0, 
    23, 
    value=(0, 23))

    #Filtered data for chart "Frequency of pickup bt Hour"
    ny_in_interval = (dfNy['hour_pickup'] >= pickup_hour[0]) & (dfNy['hour_pickup'] <= pickup_hour[1])
    filtered_data_ny = dfNy[ny_in_interval]
    filtered_data_hour_ny = filtered_data_ny.groupby('hour_pickup').apply(count_rows)

     #Chart Pickup by Hour
    st.markdown("<h4 style='text-align: center;font-family=\'Helvetica\';'><b>Frequency of pickup by hour - Uber - 15/01/2015</b></h4>", unsafe_allow_html=True)
    st.bar_chart(filtered_data_hour_ny)

    #Chart Pickup by Minutes
    filtered_data_minute_ny = filtered_data_ny.groupby('minute_pickup').apply(count_rows)

    st.markdown("<h4 style='text-align: center;font-family=\'Helvetica\';'><b>Frequency of pickup by Minutes during the day - Uber - 15/01/2015</b></h4>", unsafe_allow_html=True)
    st.bar_chart(filtered_data_minute_ny)

    dfNy_plot = filtered_data_ny[["hour_pickup", "average_speed"]].groupby('hour_pickup').mean()
    dfNy_plot = dfNy_plot.fillna(dfNy_plot["average_speed"].mean())
    st.markdown("<h4 style='text-align: center;font-family=\'Helvetica\';'><b>Average trips' speed by hours during the day - Uber - 15/01/2015</b></h4>", unsafe_allow_html=True)
    st.line_chart(dfNy_plot)

st.markdown("<br/><br/><br/>", unsafe_allow_html=True)
'''



    


############ END FORMATTING #########