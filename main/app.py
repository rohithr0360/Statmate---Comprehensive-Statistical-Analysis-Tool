import time
import math
import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image as pl
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from streamlit_option_menu import option_menu
import plotly.graph_objs as go
import json
import Chart,Plots,TimeSeries,anova,Tests,Hypo_test,addon,txttocsv
import anova as an
import random
import pymongo
from pymongo import MongoClient

st.markdown(''' 
<style>
.css-d1b1ld edgvbvh6
{
    visibility : hidden;
}
.css-lv8iw71.eknhn3m4
{
    visibility : hidden;
}
</style>
''',unsafe_allow_html = True)

@st.cache_data
def servermongo():
    client = MongoClient("mongodb+srv://Str_2364353:mjo2h5KxnbV5EMJG@cluster0.yvnzvii.mongodb.net/?retryWrites=true&w=majority")
    db = client['statmate']
    collection = db['Data_collection1']
servermongo()

plt.style.use("ggplot")

st.set_option('deprecation.showPyplotGlobalUse', False)

def loadlottieurl(url:str):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

head,mid,anim=st.columns([40,20,40])

def design():
    with head:
        st.write(" ")
        st.write(" ")

        str_path = "D:/statmate/Images/" + "logo" + ".PNG"
        img = pl.open(str_path)
        st.image(img, width=350)

    with anim:
        with open("D:/statmate/Assets/128445-project-config.json") as f:
            animation_data = json.load(f)
            st_lottie(animation_data)


with st.sidebar:
    sel=option_menu(
        menu_title='Main Menu',
        options=['Data Charts','Control Charts','Time Series','ANOVA','Covariance & Correlation','Hypothesis Testing','Text to CSV','Feedback and Add-on'],
        icons=['pie-chart','activity','graph-up-arrow','table','grid','check','pencil-square','calendar2-check']        
    )

if sel=='Control Charts':
    
    design()
    op=st.selectbox("Choose a method",['X-R Chart','X-SD Chart','NP Chart','P Chart','P Chart (varying UCL)','C Chart'])
    if(op=='X-R Chart'):
        a=1
        Chart.xr_chart(a)

    if(op=='X-SD Chart'):
        a=1
        Chart.xsd_chart(a)
            
    if(op=='NP Chart'):
        b=1
        Chart.np_chart(b)    

    if(op=='P Chart'):
        c=1
        Chart.p_chart(c)  

    if(op=='P Chart (varying UCL)'):
        c=1
        Chart.vary_ucl_chart(c)  

    if(op=='C Chart'):
        d=1
        Chart.c_chart(d)       
   
if sel=='Data Charts':
    design()
    op=st.selectbox("Choose a method",['Line Chart','Bar Chart','Histogram','Box Plot','Scatter Plot','Violin Plot','Heatmap','3D Scatter Plot','Pie Chart'])   
    
    if(op=='Bar Chart'):
        Plots.bar_chart()

    if(op=='Histogram'):
        Plots.histo()

    if(op=="Line Chart"):
        Plots.line_plot()

    if op == 'Heatmap':
        Plots.heatmap()

    if(op=='Box Plot'):
        Plots.box_plot()

    if(op=='Scatter Plot'):
        Plots.scatter_plot()

    if(op=='3D Scatter Plot'):
        Plots.scatter_3d()

    if(op=='Pie Chart'):
        Plots.pie_chart()

    if op == 'Violin Plot':
        Plots.violin_plot()
 
if sel=='Time Series':
    design()
    op=st.selectbox('Choose a method',['Forecasting Errors','Moving Average','Smoothing','Trend','Seasonal Indices'])

    if(op == 'Forecasting Errors'):
        TimeSeries.errors()

    if(op=='Moving Average'):
        rad=st.radio('Type',['Simple Moving Average','Weighted Moving Average'])
        if rad=="Simple Moving Average":
            TimeSeries.sim_mov_avg() 
        
        if rad=='Weighted Moving Average':
            TimeSeries.weighted_mov_avg()

    if(op=='Smoothing'):
        rad=st.radio('Type',['Simple Exponential Smoothing','Double Exponential Smoothing'])
       
        if rad=="Simple Exponential Smoothing":
            TimeSeries.simp_expo_smoothing() 
        
        if rad=='Double Exponential Smoothing':
            TimeSeries.double_expo_smoothing()

    if(op=='Trend'):
        c1, c2, c4 = st.columns([4,1,4])
        
        rad=st.radio('Choose a method',['Linear Trend','Parabolic Trend','Exponential Trend'])
        if rad=="Linear Trend":
            TimeSeries.linear_trend() 
            
        if rad=='Parabolic Trend':
            TimeSeries.para_trend()
            
        if rad=="Exponential Trend":
            TimeSeries.expo_trend() 
    
    if(op=='Seasonal Indices'):
        rad=st.radio('Choose a method',['Simple Averages','Ratio to Trend','Ratio to Moving Average'])
        
        
        if rad=="Simple Averages":
            TimeSeries.seas_index_sim_avg()
            
        if rad=='Ratio to Trend':
            TimeSeries.seas_ind_rat_to_trend()

        if rad=='Ratio to Moving Average':
            TimeSeries.seas_ind_rat_to_mov_avg()


if sel=='ANOVA':
    design()
    op = st.selectbox("Choose a methood",["Completely randomized design","Randomized block design","Latin square design"],index = 0)
    if op == "Completely randomized design":
        
        an.completely_randomized_design()

    if op == "Randomized block design":
        an.randomized_block_design()
    if op == "Latin square design":
        an.latin_block_design()

        
if sel=='Covariance & Correlation':
    design()
    op=st.selectbox("Choose a method",["Covariance Matrix","Correlation Matrix"])    
    if(op=='Covariance Matrix'):
        Tests.cov_mat() 
    if(op=='Correlation Matrix'):  
        Tests.corr_mat()
   
if sel=='Hypothesis Testing':
    design()
    op=st.selectbox("Choose a method",["Chi-Square", 'Z-Test', 'T-Test'])  
    if(op=='Chi-Square'):
        Hypo_test.chi_sq()
    if(op=='Z-Test'):
        Hypo_test.z_test()
    if(op=='T-Test'):
        Hypo_test.t_test()


if sel == 'Text to CSV':
    txttocsv.text_to_csv_converter()
        
if sel == 'Feedback and Add-on':
    addon.newadd()


    
    


