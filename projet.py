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


@st.cache(allow_output_mutation=True)
def get_df_lat_lon(df):
    get_df_lat_lon = df[['latitude','longitude']]
    return get_df_lat_lon

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
def load_data_by_year(year):

    df = pd.read_csv('./Ressources/' + str(year) + '_final.csv', ',')

    #Pre-process data
    df['date_mutation'] = df['date_mutation'].map(pd.to_datetime)
    df['code_departement'] = df['code_departement'].map(str)
    df['transaction_type'] = df['type_local']

    df['day'] = df['date_mutation'].map(get_dom)
    df['month'] = df['date_mutation'].map(get_month)
    df['year'] = df['date_mutation'].map(get_year)

    return df

@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist=True)
def load_all_data(df_2017, df_2018, df_2019, df_2020):
        
    frames = [df_2017, df_2018, df_2019, df_2020]
    df = pd.concat(frames)

    return df

xx = """ @timed
@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist=True)
def load_all_data(percent):
  #df_full = load_data('full.csv', ',')

  df_2016 = load_data('full_2016.csv', ',', percent)
  df_2017 = load_data('full_2017.csv', ',', percent)
  df_2018 = load_data('full_2018.csv', ',', percent)
  df_2019 = load_data('full_2019.csv', ',', percent)
  df_2020 = load_data('full_2020.csv', ',', percent)

  frames = [df_2016, df_2017, df_2018, df_2019, df_2020]
  #frames = [df_2016, df_2017]

  df = pd.concat(frames)
  #df = df_2016

  #Pre-process data
  df['date_mutation'] = df['date_mutation'].map(pd.to_datetime)
  df['code_departement'] = df['code_departement'].map(str)

  df['day'] = df['date_mutation'].map(get_dom)
  df['month'] = df['date_mutation'].map(get_month)
  df['year'] = df['date_mutation'].map(get_year)

  return df """


############ END FUNTIONS ############

############ START EXECUTION #########






############ END EXECUTION #########

############ START FORMATTING #########





def print_main(df_RE):


    st.markdown("<h1 style='text-align: center;font-family=\'Helvetica\';'>PROJECT - Valeurs foncières Visualization 👁</h1>", unsafe_allow_html=True)

    st.markdown('This is a dashboard to visualize relevant datas and charts about the Real Estate values between 2016 and 2020.')
    st.markdown('***')

    #Expander Real Estate Dataset Values
    expander = st.expander("Real Estate Values - 2016")
    col1, col2 = expander.columns(2)
    col1.metric("Nombre de lignes", df_RE.shape[0])
    col2.metric("Nombre de colonnes", df_RE.shape[1])





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
    values = df_RE['transaction_type'].value_counts()

    #Pie chart of Mutations' types repartition
    right_col_price.markdown("<h4 style='text-align: center;margin-left: -10%;font-family=\'Helvetica\';'><b>Repartition of real estate's types transactions in the selected price range</b></h4>", unsafe_allow_html=True)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(width=700)
    right_col_price.plotly_chart(fig)

    st.markdown("***")



    #External plot (using plotly) - N°4 (last)
    #Map of all transactions in metropolitain france
    st.markdown("<h4 style='text-align: center;margin-left: -10%;font-family=\'Helvetica\';'><b>HeatMap of transactions across France 🇫🇷</b></h4>", unsafe_allow_html=True)

    map_left, map_right = st.columns(2)

    type = map_left.radio(
        "Select the transactions\' types you want to see on the map:",
        ('Houses', 'Apartments', 'Dependency', 'Industrial or commercial premises'))
    
    type_radio_condition = (df_RE['type_local'] == "Global")

    if type == 'Houses':
        type_radio_condition = (df_RE['type_local'] == "Maison")
        type_radio_data = df_RE[type_radio_condition]

    elif type == "Apartments":
        type_radio_condition = (df_RE['type_local'] == "Appartement")
        type_radio_data = df_RE[type_radio_condition]

    elif type == "Dependency":
        type_radio_condition = (df_RE['type_local'] == "Dépendance")
        type_radio_data = df_RE[type_radio_condition]

    elif type == "Industrial or commercial premises":
        type_radio_condition = (df_RE['type_local'] == "Local industriel. commercial ou assimilé")
        type_radio_data = df_RE[type_radio_condition]

        

    #Choice of region to display 
    map_choice = map_right.selectbox('Which map do you want to vizualise?', ('Metropolitan France', 'Martinique', 'Reunion Island' ))
    if map_choice == ('Metropolitan France') :
      
        #Affichage de la carte
        fig = px.density_mapbox(get_df_lat_lon(type_radio_data), lat='latitude', lon='longitude',center=dict(lat=47.00, lon=2.19), zoom=5, radius=1)
        fig.update_layout(width=1200, height=800, mapbox_style="open-street-map")
        st.plotly_chart(fig)

    if map_choice == 'Martinique':

        #Affichage de la carte
        fig2 = px.density_mapbox(get_df_lat_lon(type_radio_data), lat='latitude', lon='longitude', radius=1,center=dict(lat=16, lon=-61), zoom=5,mapbox_style="stamen-terrain")
        fig2.update_layout(width=1200, height=800, mapbox_style="open-street-map")
        st.plotly_chart(fig2)    

    if map_choice == ('Reunion Island'):

        #Affichage de la carte
        fig3 = px.density_mapbox(get_df_lat_lon(type_radio_data), lat='latitude', lon='longitude', radius=1,center=dict(lat=-21.1, lon=55.3), zoom=7,mapbox_style="stamen-terrain")
        fig3.update_layout(width=1200, height=800, mapbox_style="open-street-map")
        st.plotly_chart(fig3)

def main():
    st.sidebar.write("Please close the sidebar after choosing the datatset to get the best experience !")
    option_year = st.sidebar.selectbox('What year do you want to display?', ('Year 2017', 'Year 2018', 'Year 2019', 'Year 2020', 'Full'))

    df_2017 = load_data_by_year(2017)
    df_2018 = load_data_by_year(2018)
    df_2019 = load_data_by_year(2019)
    df_2020 = load_data_by_year(2020)
    df_all = load_all_data(df_2017, df_2018, df_2019, df_2020)



    if option_year == 'Year 2017':
        print_main(df_2017)
    
    elif option_year == "Year 2018":
        print_main(df_2018)
    
    elif option_year == "Year 2019":
        print_main(df_2019)
    
    elif option_year == "Year 2020":
        print_main(df_2020)
    
    elif option_year == "Full":
        print_main(df_all)


if __name__ == "__main__":
    main()


    


############ END FORMATTING #########