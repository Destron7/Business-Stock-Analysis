import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
import holidays
import matplotlib.pyplot as plt
import datetime
# import plotly.graph_objs as go 




# Tabs section
tabs = st.sidebar.radio("Navigation", ("Stock Comparison", "Individual Stock Data",'Holidays'))

if tabs == "Stock Comparison":
            
        # Function to load data for a given ticker and date range

        #  Read ticker symbols from Excel sheet
        ticker_df = pd.read_excel('CodeList.xlsx')
        ticker_symbols = ticker_df['Yahoo Code'].tolist()

        # Sidebar section
        st.sidebar.header("Select Parameters")
        ticker1 = st.sidebar.selectbox('First Ticker', ticker_symbols, index=0)
        ticker2 = st.sidebar.selectbox('Second Ticker', ticker_symbols, index=1)
        ticker3 = st.sidebar.selectbox('Third Ticker', ticker_symbols, index=2)
        start_date1 = st.sidebar.date_input('Start Date', value=datetime.date(2014, 1, 1))
        end_date1 = st.sidebar.date_input('End Date')            

        # Function to plot stock comparison graph
        data1 = yf.download(ticker1, start_date1, end_date1)
        data2 = yf.download(ticker2, start_date1, end_date1)
        data3 = yf.download(ticker3, start_date1, end_date1)

        # Plot all three stocks on one graph
        st.header("Stocks Comparison")
        st.write('Close Price Comparison')
        plt.figure(figsize=(15, 7))
        data1['Close'].plot(label=ticker1)
        data2['Close'].plot(label=ticker2)
        data3['Close'].plot(label=ticker3)
        plt.ylabel('Stock Price')
        plt.title("Stock Price Comparison")
        plt.legend()
        st.pyplot(plt)

        st.write('Volume Comparison')
        plt.figure(figsize=(15, 7))
        data1['Volume'].plot(label=ticker1)
        data2['Volume'].plot(label=ticker2)
        data3['Volume'].plot(label=ticker3)
        plt.ylabel('Stock Price')
        plt.title("Stock Price Comparison")
        plt.legend()
        st.pyplot(plt)

        st.write('Total Traded Comparison')
        plt.figure(figsize=(15, 7))
        data1['Total Traded'] = data1['Open'] * data1['Volume']
        data2['Total Traded'] = data2['Open'] * data2['Volume']
        data3['Total Traded'] = data3['Open'] * data3['Volume']
        data1['Total Traded'].plot(label=ticker1)
        data2['Total Traded'].plot(label=ticker2)
        data3['Total Traded'].plot(label=ticker3)
        plt.ylabel('Total Traded')
        plt.title("Total Traded Comparison")
        plt.legend()
        st.pyplot(plt)

        data1['% Change'] = data1['Adj Close'] / data1['Adj Close'].shift(1) - 1
        data1.dropna(inplace=True)
        data2['% Change'] = data2['Adj Close'] / data2['Adj Close'].shift(1) - 1
        data2.dropna(inplace=True)
        data3['% Change'] = data3['Adj Close'] / data3['Adj Close'].shift(1) - 1
        data3.dropna(inplace=True)

        # Calculate annual return for all three tickers
        annual_return1 = data1['% Change'].mean() * 252 * 100
        annual_return2 = data2['% Change'].mean() * 252 * 100
        annual_return3 = data3['% Change'].mean() * 252 * 100

        # Display the annual returns
        st.write('Annual Return for', ticker1, 'is', annual_return1, '%')
        st.write('Annual Return for', ticker2, 'is', annual_return2, '%')
        st.write('Annual Return for', ticker3, 'is', annual_return3, '%')



elif tabs=='Individual Stock Data':
            st.title("Stock Dashboard")
            st.sidebar.header("Select Parameter from below")
            ticker = st.sidebar.text_input('ticker',value='ADANIENT.NS')
            start_date = st.sidebar.date_input('Start Date', value=datetime.date(2014, 1, 1))
            end_date = st.sidebar.date_input('End Date')
            
            st.write('Data from ',start_date,'to',end_date)
            data = yf.download(ticker, start=start_date, end=end_date)
            fig = px.line(data, y=data.columns, title=ticker,width=900,height=500)
            st.plotly_chart(fig)
            
            
            # To add indexing
            data.insert(1,'Date',data.index,True)
            data.reset_index(drop=True,inplace=True)
            
            
            
            pricing_data,fundamental_data,news,hundred_ma,twohundred_ma,Prediction_Graph=st.tabs(['Pricing Data','Fundamental Data','Top 10 News','100ma','200ma','Prediction Graph'])
            
            with pricing_data:
                st.header('Price Movements')
                data2=data
                data2['% Change']=data['Adj Close']/data['Adj Close'].shift(1)-1
                data2.dropna(inplace=True)
                st.write(data2)
            
                annual_return=data2['% Change'].mean()*252*100
                st.write('Annual Return is ',annual_return,'%')
                stdev=np.std(data2["% Change"])*np.sqrt(252)
                st.write("Standard Deviation is ",stdev*100,'%')
                st.write("Risk Adjustment Return is ",annual_return/(stdev*100))
            
            with fundamental_data:
                st.header('Fundamental Data from Yahoo Finance')
                ticker_info = yf.Ticker(ticker)
            
                st.subheader('Balance Sheet')
                balance_sheet = ticker_info.balance_sheet
                st.write(balance_sheet)
            
                st.subheader('Income Statement')
                income_statement = ticker_info.financials
                st.write(income_statement)
            
                st.subheader("Cash Flow Statement")
                cash_flow = ticker_info.cashflow
                st.write(cash_flow)
            
            from stocknews import StockNews
            with news:
                # pip install stocknews
                st.header(f'News of {ticker}')
                sn=StockNews(ticker,save_news=False)
                df_news=sn.read_rss()
                for i in range(10):
                    st.subheader(f'News {i+1}')
                    st.write(df_news['published'][i])
                    st.write(df_news['title'][i])
                    st.write(df_news['summary'][i])
                    # this can be used for sentiment analysis
                    title_sentiment=df_news['sentiment_title'][i]
                    st.write(f'Title Sentiment {title_sentiment}')
                    news_sentiment=df_news['sentiment_summary'][i]
                    st.write(f'News Sentiment {news_sentiment}')
            
           
            
            
            with hundred_ma:
                        st.subheader('Closing Price vs Time Chart with 100MA')
                        ma100 = data.Close.rolling(100).mean()
                        fig = plt.figure(figsize=(12,6))
                        plt.plot(ma100, label='100-day Moving Average', color='red')
                        plt.plot(data.Close, label='Closing Price', color='blue')
                        plt.legend()  # Add legend
                        st.pyplot(fig)

            with twohundred_ma:
                            st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
                            ma100 = data.Close.rolling(100).mean()
                            ma200 = data.Close.rolling(200).mean()
                            fig = plt.figure(figsize=(12,6))
                            plt.plot(ma100, label='100-day Moving Average', color='red')
                            plt.plot(ma200, label='200-day Moving Average', color='green')
                            plt.plot(data.Close, label='Closing Price', color='blue')
                            plt.legend()  # Add legend
                            st.pyplot(fig)

            
            from keras.models import load_model
            from sklearn.preprocessing import MinMaxScaler
            # Keras is api used for deeplearning  
            # Reinforcemt model - 1st collect data and then predict data works on itself
            # 1. Supervised - prediction using data from past data
            # 2. Unsupervised - we dont have any past it follows an algo for data prediction
            with Prediction_Graph:
                column=st.selectbox('Select the column to be used for predicting ',data.columns[2:])
                # subsetting the date
                data=data[['Date',column]]
                st.write(data)
                # Splitting data into Training and Testing
                data_training=pd.DataFrame(data[column][0:int(len(data)*0.7)])
                data_testing=pd.DataFrame(data[column][int(len(data)*0.7):int(len(data))])
            
            
            
                scaler=MinMaxScaler(feature_range=(0,1))
            
                data_training_array=scaler.fit_transform(data_training)
            
            
                # Splitting data into x_train and y_train
                x_train=[]
                y_train=[]
            
                for i in range(100,data_training_array.shape[0]):
                    x_train.append(data_training_array[i-100:i])
                    y_train.append(data_training_array[i,0])
            
                x_train,y_train=np.array(x_train),np.array(y_train)
            
            
                # Loading the model
                model=load_model('keras_model.h5')
            
                # Testing Part
                past_100_days=data_training.tail(100)
                final_data = pd.concat([past_100_days, data_testing], ignore_index=True)
            
                input_data = scaler.fit_transform(final_data)
            
                x_test=[]
                y_test=[]
            
                for i in range(100,input_data.shape[0]):
                    x_test.append(input_data[i-100:i])
                    y_test.append(input_data[i,0])
            
                x_test,y_test = np.array(x_test),np.array(y_test)
            
                # Making Predictions
            
            
                y_predicted = model.predict(x_test)
                scaler=scaler.scale_
            
                scale_factor =1/scaler[0]
                y_predicted=y_predicted*scale_factor
                y_test=y_test*scale_factor
            
            
                # Final Graph
            
                st.subheader("Predictions vs Original")
                fig2=plt.figure(figsize=(12,6))
                plt.plot(y_test,'b',label="Original Price")
                plt.plot(y_predicted,'r',label="Predicted Price")
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig2)

else:
                
                st.header('Stock Market Holidays')
                year = st.sidebar.number_input('Year',  value=datetime.datetime.now().year)
                in_holidays = holidays.India()
                holidays_list = []
                for date, name in sorted(holidays.India(years=(year)).items()):
                    holidays_list.append(f"{date}: {name}")
                st.write(holidays_list)            
            
