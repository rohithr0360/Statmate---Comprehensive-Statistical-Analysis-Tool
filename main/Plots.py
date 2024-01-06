import streamlit as st
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import pymongo
from pymongo import MongoClient
import random

client = MongoClient("mongodb+srv://Str_2364353:mjo2h5KxnbV5EMJG@cluster0.yvnzvii.mongodb.net/?retryWrites=true&w=majority")
db = client['statmate']
collection = db['Data_collection1']

def get_data(prev,cv):
    v1 = cv
    up,d = st.columns([60,40])
    with up:
        uploaded_file = st.file_uploader("Choose a file",type = "csv")
    with d:
        data_from_db = st.text_input("Retrive data from DB ")
        cv1 = cv + str(1001)
        st.write("Sample Code: ",cv1)

    if not uploaded_file and not data_from_db:
        ax = st.checkbox("Show sample dataset")
       # if ax:
    #        str_path = "C:/CODES/Project/Images/" + cv + ".PNG"
   #         img = pl.open(str_path)
    #        st.image(img, width=350)   

    if uploaded_file and not data_from_db:

        df = pd.read_csv(uploaded_file)
        
        data1 = collection.find_one({"_id": "Random_data"})["data_num"]
        rand_num = random.randint(1000, 9999 )

        while True:
            if rand_num in data1:
                rand_num = random.randint(1000, 9999 )
            else:
                cv2 = cv + str(rand_num)
                data1.append(cv2)

                collection.update_one({"_id":"Random_data"},{"$set":{"data_num":data1}})
                break

        data3 = collection.find_one({"_id": prev})["data"]

        data3 = pd.DataFrame(data3)


        if not (df.empty) and not (df.equals(data3)):
            cv21 = cv + str(rand_num)
            post= {
                "_id": cv21,
                "data" :  df.to_dict(orient='records')
                }
            collection.insert_one(post)
            collection.update_one({"_id":10001},{"$set":{"Previous":cv21}})

            up2,d5 = st.columns([75,25])
            with up2:
                ax = st.checkbox("Show dataset")
                if ax:
                    st.dataframe(df)
            
            with d5:
                st.write("Data code: ",cv + str(rand_num))

        else:
            up2,d5 = st.columns([75,25])
            with up2:
                ax = st.checkbox("Show dataset")
                if ax:
                    st.dataframe(df)
            with d5:

                st.write("Data code: ",str(prev))

        return df

    if data_from_db and not uploaded_file:
        
        
        retrieved_record = collection.find_one({"_id": data_from_db})
        if retrieved_record:
            df = pd.DataFrame(retrieved_record["data"])

            up2,d5 = st.columns([75,25])
            with up2:
                ax = st.checkbox("Show dataset")
                if ax:
                    st.dataframe(df)
            with d5:
                st.write("Data code: ",data_from_db)
            
            return df

        else:
            st.error('Enter a valid code')

    if data_from_db and  uploaded_file:
        st.info("Select one option")


a=1
if a==1:
    def bar_chart():
        
       
        try:
            prev = collection.find_one({"_id": 10001})["Previous"]
            df = get_data(prev,'DATA - ')
            if df is None:
                st.markdown("""
                ### Description:
                """) 
                st.write("A bar chart is a graphical representation of data that displays categorical data with rectangular bars. The height or length of each bar represents the frequency or proportion of the data in that category.")  

            if df is not None:

                columns = df.columns.tolist()

                x_axis,y_axis=st.columns(2)
                x_axis=st.selectbox('X-axis',columns)
                y_axis=st.selectbox('Y-axis',columns)
  

                fig = px.bar(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)

            else:
                st.info("Upload a csv file or Enter a code")   

        except Exception as e:
            st.error("Provide a valid dataset")


    def scatter_plot():
      
        try:
            prev = collection.find_one({"_id": 10001})["Previous"]
            df = get_data(prev,'DATA - ')
            if df is None:
                st.markdown("""
                ### Description:
                """) 
                st.write("A scatter plot is a graphical representation of data that shows the relationship between two continuous variables by placing data points on a Cartesian coordinate system.") 
        

            if df is not None:

                columns = df.columns.tolist()

                x_axis,y_axis=st.columns(2)
                x_axis=st.selectbox('X-axis',columns)
                y_axis=st.selectbox('Y-axis',columns)

                fig = px.scatter(df, x=x_axis, y=y_axis,title="Scatter Plot")
                st.plotly_chart(fig)

            else:
                st.info("Upload a csv file or Enter a code")
        except Exception as e:
            st.error("Provide a valid dataset")

    def scatter_3d():
        st.markdown("""
        ### Description:
        """) 
        st.write("A 3D scatter plot is a graphical representation of data that shows the relationship between three continuous variables by placing data points in a three-dimensional coordinate system.") 
        
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()

                x_axis,y_axis,z_axis=st.columns(3)
                x_axis=st.selectbox('X-axis',columns)
                y_axis=st.selectbox('Y-axis',columns)
                z_axis=st.selectbox('Z-axis',columns)

                fig = px.scatter_3d(df, x=x_axis, y=y_axis,z=z_axis,title="3D Scatter Plot")
                st.plotly_chart(fig)

            else:
                st.info("Upload a csv file or Enter a code")
        except Exception as e:
            st.error("Provide a valid dataset")

    def pie_chart():
       
        try:
            prev = collection.find_one({"_id": 10001})["Previous"]
            df = get_data(prev,'DATA - ')
            if df is None:
                st.markdown("""
                ### Description:
                """) 
                st.write("A pie chart is a graphical representation of data that displays the relative sizes of different categories as slices of a pie. It is often used to show the proportion of data in each category.") 
                
            if df is not None:
                columns = df.columns.tolist()

                cols=st.multiselect('Columns',columns)
                vals=[]
                for i in cols:
                    t=df[i].values.tolist()
                    for j in t:
                        vals.append(j)

                fig = px.pie(df, names=cols, values=vals,title="Pie Chart",template="seaborn")
                st.plotly_chart(fig)

            else:
                st.info("Upload a csv file or Enter a code")
        except Exception as e:
            st.error("Provide a valid dataset")

    def histo():
                
        try:
            prev = collection.find_one({"_id": 10001})["Previous"]
            df = get_data(prev,'DATA - ')
            if df is None:
                st.markdown("""
                ### Description:
                """) 
                st.write("A histogram is a graphical representation of data that displays the frequency or proportion of data in intervals or bins. It is often used to show the distribution of continuous data.") 
                
            if df is not None:
                columns = df.columns.tolist()

                x_axis,y_axis=st.columns(2)
                x_axis=st.selectbox('X-axis',columns)
                y_axis=st.selectbox('Y-axis',columns)
                

                fig = px.histogram(df,x=x_axis,y=y_axis)
                st.plotly_chart(fig)

            else:
                st.info("Upload a csv file or Enter a code")
        except Exception as e:
            st.error("Provide a valid dataset")
  
    def box_plot():
                
        try:
            prev = collection.find_one({"_id": 10001})["Previous"]
            df = get_data(prev,'DATA - ')
            if df is None:
                st.markdown("""
                ### Description:
                """) 
                st.write("A box plot is a graphical representation of data that displays the median, quartiles, and outliers of a distribution. It is often used to show the distribution and variability of data.") 
                        
            if df is not None:
                columns = df.columns.tolist()

                x_axis,y_axis=st.columns(2)
                x_axis=st.selectbox('X-axis',columns)
                y_axis=st.selectbox('Y-axis',columns)
                

                fig = px.box(df,x=x_axis,y=y_axis)
                st.plotly_chart(fig)

            else:
                st.info("Upload a csv file or Enter a code")
        except Exception as e:
            st.error("Provide a valid dataset")

    def line_plot():
                
        try:
            prev = collection.find_one({"_id": 10001})["Previous"]
            df = get_data(prev,'DATA - ')
            if df is None:
                st.markdown("""
                ### Description:
                """) 
                st.write("A line chart is a graphical representation of data that shows the relationship between two variables by connecting data points with lines. It is often used to display trends over time.") 
                        
            if df is not None:
                columns = df.columns.tolist()

                x_axis,y_axis=st.columns(2)
                x_axis=st.selectbox('X-axis',columns)
                y_axis=st.selectbox('Y-axis',columns)
                

                fig = px.line(df,x=x_axis,y=y_axis)
                st.plotly_chart(fig)

            else:
                st.info("Upload a csv file or Enter a code")
        except Exception as e:
            st.error("Provide a valid dataset")
            
    def violin_plot():
              
        try:
            prev = collection.find_one({"_id": 10001})["Previous"]
            df = get_data(prev,'DATA - ')
            if df is None:
                st.markdown("""
                ### Description:
                """) 
                st.write("A violin plot is a graphical representation of data that combines a box plot and a kernel density plot. It is often used to show the distribution and density of data.") 
                                
            if df is not None:
                columns = df.columns.tolist()
                x_axis,y_axis= st.columns(2)
                with x_axis:
                    x = st.selectbox('X-axis',columns)
                with y_axis:
                    y = st.selectbox('Y-axis',columns)

                fig = px.violin(df, y=y, x=x, box=True, points='all', title='Violen Chart')

                st.plotly_chart(fig)

            else:
                st.info("Upload a csv file or Enter a code")               
        except Exception as e:
            st.warning("Provide a valid dataset")

    def heatmap():
        try:
            prev = collection.find_one({"_id": 10001})["Previous"]
            df = get_data(prev,'DATA - ')
            if df is None:
                st.markdown("""
                ### Description:
                """) 
                st.write("A heatmap is a graphical representation of data that uses color-coding to visualize the values of a matrix. The colors in the heatmap represent the relative values of the data, with brighter colors indicating higher values and darker colors indicating lower values. Heatmaps are commonly used to visualize data in various fields, including data science, biology, and finance, and are useful for identifying patterns and trends in large datasets") 
            
            if df is not None:           
                columns = df.columns.tolist()
                x_axis,y_axis,z_axis= st.columns(3)
                with x_axis:
                    x = st.selectbox('X-axis',columns)
                with y_axis:
                    y = st.selectbox('Y-axis',columns)
                with z_axis:
                    z = st.selectbox('Z-axis',columns)


                fig = go.Figure(data=go.Heatmap(
                        z=df[z],
                        x=df[x],
                        y=df[y],
                        colorscale='Viridis'))

    
                fig.update_layout(
                    title='Heatmap',
                    xaxis_title='X Axis Title',
                    yaxis_title='Y Axis Title')

                st.plotly_chart(fig)
                
            else:
                st.info("Upload a csv file or Enter a code")

        except Exception as e:
            st.warning("Provide a valid dataset")





