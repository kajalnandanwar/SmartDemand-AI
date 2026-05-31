import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.title("📈 Demand Forecasting Dashboard")

uploaded_file = st.file_uploader(
    "Upload Sales CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df['date'] = pd.to_datetime(df['date'])

    store = st.selectbox(
        "Select Store",
        sorted(df['store'].unique())
    )

    item = st.selectbox(
        "Select Item",
        sorted(df['item'].unique())
    )

    filtered_df = df[
        (df['store'] == store) &
        (df['item'] == item)
    ]

    prophet_df = filtered_df[['date','sales']]
    prophet_df.columns = ['ds','y']

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=90
    )

    forecast = model.predict(future)

    st.subheader("Future Demand Forecast")

    fig1 = px.line(
        forecast,
        x='ds',
        y='yhat',
        title='Forecast Demand'
    )

    st.plotly_chart(fig1)

    st.subheader("Demand Distribution")

    fig2 = px.histogram(
        filtered_df,
        x='sales',
        nbins=30
    )

    st.plotly_chart(fig2)

    filtered_df['Month'] = filtered_df['date'].dt.month

    monthly = filtered_df.groupby(
        'Month'
    )['sales'].sum().reset_index()

    st.subheader("Monthly Demand")

    fig3 = px.bar(
        monthly,
        x='Month',
        y='sales'
    )

    st.plotly_chart(fig3)

    st.metric(
        "Average Demand",
        round(filtered_df['sales'].mean(),2)
    )

    st.metric(
        "Max Demand",
        filtered_df['sales'].max()
    )

    st.metric(
        "Predicted Demand (90 Days)",
        round(forecast['yhat'].iloc[-1],2)
    )
