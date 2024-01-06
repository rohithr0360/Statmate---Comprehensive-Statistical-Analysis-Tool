import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import itertools
from PIL import Image as pl
import pymongo
from pymongo import MongoClient
import random

client = MongoClient("mongodb+srv://Str_2364353:mjo2h5KxnbV5EMJG@cluster0.yvnzvii.mongodb.net/?retryWrites=true&w=majority")
db = client['statmate']
collection = db['Data_collection1']

def loadlottieurl(url:str):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

lottie_url_teach = 'https://assets3.lottiefiles.com/packages/lf20_22mjkcbb.json'

lottie_teach = loadlottieurl(lottie_url_teach)

def get_data(prev,cv):
    v1 = cv

    ve = cv
    up,d = st.columns([60,40])
    with up:
        uploaded_file = st.file_uploader("Choose a file",type = "csv")
    with d:
        data_from_db = st.text_input("Retrive data from DB ")
        cv1 = cv + str(1001)
        st.write("Sample Code: ",cv1)

    if not uploaded_file and not data_from_db and cv == "COV - ":
        st.markdown("""
        ### Description:
        """,True) 
        st.write("Covariance matrix measures how much two variables vary together. It is used to determine the direction of the relationship between the variables and whether it is positive or negative. However, the magnitude of the covariance is difficult to interpret because it depends on the units of the variables.")

        ax = st.checkbox("Show sample dataset")
        if ax:
            str_path = "C:/CODES/Project/Images/" + "COV - " + ".PNG"
            img = pl.open(str_path)
            st.image(img, width=350)

    if not uploaded_file and not data_from_db and cv == "COR - ":
        st.markdown("""
        ### Description:
        """,True) 
        st.write("Correlation matrix, on the other hand, measures the strength and direction of the linear relationship between two variables. It is a standardized measure and takes values between -1 and 1, where -1 indicates a perfectly negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation. The correlation matrix is often used in multivariate analysis to summarize the correlations between several variables.") 
        ax = st.checkbox("Show sample dataset")
        if ax:
            str_path = "C:/CODES/Project/Images/" + "COR - " + ".PNG"
            img = pl.open(str_path)
            st.image(img, width=450)

    if uploaded_file and not data_from_db:

        df = pd.read_csv(uploaded_file,header = None)

        data = df.values.tolist()

        k = True

        data1 = collection.find_one({"_id": "Random_data"})["data_num"]
        rand_num = random.randint(1000, 9999 )
        
        while k:
            if rand_num in data1:
                rand_num = random.randint(1000, 9999 )
            else:
                cv2 = cv + str(rand_num)
                data1.append(cv2)

                collection.update_one({"_id":"Random_data"},{"$set":{"data_num":data1}})
                k = False
                

        data3 = collection.find_one({"_id": prev})["data"]

        if data != None and data != data3:
            cv21 = cv + str(rand_num)
            post= {
                "_id": cv21,
                "data" : data
                }
            collection.insert_one(post)
            collection.update_one({"_id":10001},{"$set":{"Previous":cv21}})

            up2,d5 = st.columns([75,25])
            with up2:
                ax = st.checkbox("Show dataset")
                if ax:
                    st.dataframe(data)
            
            with d5:

                st.write("Data code: ",cv + str(rand_num))

        else:
            up2,d5 = st.columns([75,25])
            with up2:
                ax = st.checkbox("Show dataset")
                if ax:
                    st.dataframe(data)
            with d5:
                print(ve + str(prev))
                st.write("Data code: ",prev)


        return data

    if data_from_db and not uploaded_file:
        
        data = collection.find_one({"_id": str(data_from_db)})["data"]

        up2,d5 = st.columns([75,25])
        with up2:
            ax = st.checkbox("Show dataset")
            if ax:
                st.dataframe(data)
        with d5:
            st.write("Data code: ",data_from_db)
        
        return data

    if data_from_db and  uploaded_file:
        st.info("Select anyone option")

    else:
        st.info("Upload a csv file or Enter a code")


def cov_mat():
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        data = get_data(prev,"COV - ")
        if data is not None:

            def compute_covariance_matrix(data):
                mean = np.mean(data, axis=0)
                deviation_matrix = data - mean
                covariance_matrix = np.dot(deviation_matrix.T, deviation_matrix)
                covariance_matrix = covariance_matrix / (data.shape[0] - 1)
                return covariance_matrix

            d = pd.DataFrame(data)
            covariance_matrix = compute_covariance_matrix(d)
            st.markdown('''
            ### Covariance matrix:
            ''')
            st.dataframe(covariance_matrix,width = 700)
    except Exception as e:
        st.error("Provide a valid dataset")

def corr_mat():
 
    try:

        prev = collection.find_one({"_id": 10001})["Previous"]
        data = get_data(prev,"COR - ")
        if data is not None:


            def compute_correlation_matrix(data):
                corr_matrix = data.corr()
                return corr_matrix



            d = pd.DataFrame(data)
            corr_matrix = compute_correlation_matrix(d)
            st.markdown('''
            ### Correlation matrix:
            ''')
            st.dataframe(corr_matrix,width = 700)

    except Exception as e:
        st.error("Provide a valid dataset")