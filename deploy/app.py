import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

import warnings
warnings.filterwarnings(action='ignore')

st.header('Milestone 2 Phase 1')
st.write("Nama: Risqi Wahyu Permana")
st.write("Batch: HCK 006")

#load data
@st.cache_data
def fetch_data():
    df = pd.read_csv('https://raw.githubusercontent.com/wahyupermana-10/dataset/main/h8dsft_P1M2_risqi_wahyu_permana_train.csv')
    return df

df = fetch_data()

#make two pages for EDA and Prediction
page = st.sidebar.selectbox('Choose a page', ['EDA', 'Prediction'])

if page == 'EDA':
    st.title('Exploratory Data Analysis')

    st.subheader('Correlation RAM Capacity and Price Range')
    st.write('RAM (Random Access Memory) is a hardware device that allows information to be stored and retrieved on a computer.')
    plt.figure(figsize=(8, 6))
    plt.scatter(df['ram'], df['price_range'], alpha=0.5)
    plt.xlabel('RAM Capacity')
    plt.ylabel('Price Range')
    plt.title('Correlation Between RAM Capacity and Price Range')
    st.pyplot(plt)
    st.write('We can see that the higher the RAM capacity, the more expensive the mobile phone is.')
    st.write('')

    st.subheader('Correlation Battery Power and Price Range')
    st.write('Battery power is a measure of electric energy stored by the battery. mAhs (milliampere-hours) is the most common unit used to measure battery capacity. The higher the mAh rating, the longer the battery will last.')
    plt.figure(figsize=(8, 6))
    plt.scatter(df['battery_power'], df['price_range'], alpha=0.5)
    plt.xlabel('Battery Power')
    plt.ylabel('Price Range')
    plt.title('Correlation Between Battery Power and Price Range')
    st.pyplot(plt)
    st.write('We can see that mobile phones with battery capacity from small to large are in the small to large price range.')
    st.write('')

    st.subheader('Display Resolution')
    st.write('Display resolution is the number of distinct pixels in each dimension that can be displayed. It is usually quoted as height x width, with the units in pixels: for example, "1024 × 768" means the width is 1024 pixels and the height is 768 pixels.')
    plt.figure(figsize=(8, 6))
    plt.scatter(df['px_height'], df['px_width'], alpha=0.5)
    plt.xlabel('Pixel Height')
    plt.ylabel('Pixel Width')
    plt.title('Display Resolution')
    st.pyplot(plt)
    st.write('We can see that there is a correlation between px_height and px_width. The higher the px_height, the higher the px_width.')

    st.subheader('Correlation Between Number of Cores and Clock Speed')
    st.write('Clock speed is the rate at which a processor executes a task and is measured in Gigahertz (GHz). The higher the clock speed, the faster the processor.')
    st.write('Number of cores is the number of independent processors in a single computing component.')
    plt.figure(figsize=(8, 6))
    plt.scatter(df['n_cores'], df['clock_speed'], alpha=0.5)
    plt.xlabel('Number of Cores')
    plt.ylabel('Clock Speed')
    plt.title('Correlation Between Number of Cores and Clock Speed')
    st.pyplot(plt)
    st.write('We can conclude that the number of cores and clock speed are not related. There are processors that have a large number of cores but have low clock speeds and vice versa.')

    st.subheader('Price Range')
    st.write('Price range is the target variable with value of 0 (low cost), 1 (medium cost), 2 (high cost), and 3 (very high cost).')
    sns.countplot(x='price_range', data=df)
    plt.xticks([0, 1, 2, 3], ['Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'])
    st.pyplot(plt)
    st.write('The data is balanced.')
else:
    st.title('Prediction')
    st.write('Please input the following data to predict the price range of the mobile phone.')
    st.write('')
    st.subheader('Input Features')

    st.write('Battery Power: 500 - 2000')
    battery_power = st.slider('Battery Power', 500, 2000, 1000)

    st.write('Clock Speed: 1 - 3')
    clock_speed = st.slider('Clock Speed', 0.5, 3.0, 1.0)

    st.write('Front Camera Megapixels: 0 - 20')
    fc = st.slider('Front Camera Megapixels', 0, 20, 10)

    st.write('Primary Camera Megapixels: 0 - 20')
    pc = st.slider('Primary Camera Megapixels', 0, 20, 10)

    st.write('Supported 4G: Yes/No')
    four_g = st.selectbox('Supported 4G', ['Yes', 'No'])
    if four_g == 'Yes':
        four_g = 1
    else:
        four_g = 0

    st.write('Internal Memory: 2 - 64')
    int_memory = st.slider('Internal Memory', 2, 64, 32)

    st.write('Number of Cores: 1 - 8')
    n_cores = st.slider('Number of Cores', 1, 8, 4)

    st.write('RAM Capacity: 250 - 4000')
    ram = st.slider('RAM Capacity', 250, 4000, 2000)

    st.write('Screen Height Resolution: 500 - 2000')
    px_height = st.slider('Screen Height Resolution', 500, 2000, 1000)

    st.write('Screen Width Resolution: 500 - 2000')
    px_width = st.slider('Screen Width Resolution', 500, 2000, 1000)

    load_model = joblib.load('specPrice_pred.pkl')

    if st.button('Predict'):
        input_data = [[battery_power, clock_speed, fc, four_g, int_memory, n_cores, pc, ram, px_height, px_width]]
        input_df = pd.DataFrame(input_data, columns=['battery_power', 'clock_speed', 'fc', 'four_g', 'int_memory', 'n_cores', 'pc', 'ram', 'px_height', 'px_width'])
        pred = load_model.predict(input_df)
        if pred == 0:
            pred = 'Low Cost'
        elif pred == 1:
            pred = 'Medium Cost'
        elif pred == 2:
            pred = 'High Cost'
        else:
            pred = 'Very High Cost'
        st.write("The price range of the mobile phone is```", pred, "```")
