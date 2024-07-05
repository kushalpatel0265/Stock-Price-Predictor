import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2024-07-05'

st.title('Stock Future Predictor')

user_input = st.text_input('Enter stock Ticker', 'AAPL')

if st.button('Predict'):
    df = yf.download(user_input, start, end)

    # Describing data
    st.subheader('Data From 2010-2023')
    st.write(df.describe())

    # Closing Price VS Time Chart
    st.subheader('Closing Price VS Time Chart')
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], color='yellow')
    st.pyplot(fig)

    # Closing Price VS Time Chart with 100 moving average
    st.subheader('Closing Price VS Time Chart with 100 moving Average')
    ma100 = df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], color='yellow')
    plt.plot(ma100, color='red')
    st.pyplot(fig)

    # Closing Price VS Time Chart with 100 & 200 moving average
    st.subheader('Closing Price VS Time Chart with 100 & 200 moving Average')
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], color='yellow')
    plt.plot(ma100, color='red')
    plt.plot(ma200, color='green')
    st.pyplot(fig)

    # Splitting data into training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load model
    try:
        model = load_model('model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Testing past 100 days
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final graph
    st.subheader('Prediction vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
