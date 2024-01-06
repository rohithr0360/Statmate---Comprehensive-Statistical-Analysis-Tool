import time
import math
import requests
import json
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
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




def xr_chart(a):
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        df = get_data(prev,'XR - ')
        if df is None:
            st.markdown("""
            ### Description:
            """,True) 
            st.write("The X-R chart is a type of control chart that is used to monitor the process mean and variability. It consists of two charts: the X-chart, which plots the sample means, and the R-chart, which plots the sample ranges.") 

        if df is not None:

            columns = df.columns.tolist()

            col1,col2,col3 = st.columns([4,1,4])

            with col1:     
                x_axis,y_axis,no_of_sam=st.columns(3)
                x_axis=st.selectbox('X-Bar',columns)
                y_axis=st.selectbox('Range',columns)
                no_of_sam=st.number_input("Sample Size",min_value=1.0,max_value=30.0,step=1.0)

            with col3:
                with open("D:/statmate/Assets /84045-graph-lottie-animation.json") as f:
                    animation_data = json.load(f)
                    st_lottie(animation_data)
        
            xvals=df[x_axis].values.tolist()
            yvals=df[y_axis].values.tolist()

            A2vals =[0.0,1.880, 1.023, 0.729, 0.577, 0.483, 0.419, 0.373, 0.337, 0.308, 0.285, 0.266, 0.249, 0.235, 0.223, 0.212, 0.203, 0.194, 0.187, 0.180, 0.173, 0.167, 0.162, 0.157, 0.153]
            D3vals=[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.076, 0.136, 0.184, 0.223, 0.256, 0.283, 0.307, 0.328, 0.347, 0.363, 0.378, 0.391, 0.403, 0.415, 0.425, 0.434, 0.443, 0.451]
            D4vals=[ 0.0,3.267, 2.568, 2.266, 2.089, 1.970, 1.882, 1.815, 1.761, 1.716, 1.679, 1.646, 1.618, 1.594, 1.572, 1.552, 1.534, 1.518, 1.503, 1.490, 1.477, 1.466, 1.455]
            
            
            
            st.markdown("""
            ### _The table values are_ 
            """,True)

            m=int(no_of_sam)
            A2=A2vals[m-1]
            D3=D3vals[m-1]
            D4=D4vals[m-1]

            cl1,cl2,cl3,cl4 = st.columns(4)  

            with cl1:
                st.write("A2 : ",A2)
            with cl2:
                st.write("D3 : ",D3)
            with cl3:
                st.write("D4 : ",D4)
            st.write('')
            st.write('')
            n=0
            xbar=[]
            r=[]

            for i in xvals:
                if i==0 or i<0 or i>0:
                    xbar.append(i)
                    n+=1
                else:
                    break
                
            for j in yvals:
                if j==0 or j<0 or j>0:
                    r.append(j)
                else:
                    break
            
            df = pd.DataFrame({
            'X': xbar,
            'R': r
                })

            with st.container():
                st.write(df.reset_index(drop=True).T.reset_index().rename(columns={'index': ''}).astype(object).set_index(''))
                
            clx=sum(xbar)/n
            rbar=sum(r)/n

            uclx=clx+(rbar*A2)
            lclx=clx-(rbar*A2)
            if(lclx<0):
                lclx=0
            
            st.write(D3,D4)
            lclr=rbar*D3
            uclr=rbar*D4
                
            sample_num=[]
            for i in range(1,n+1):
                sample_num.append(i)

            a0=[]
            b0=[]
            c0=[]

            st.write('')
            st.markdown(""" 
                ### Control Limits for X-bar Chart   
                """)
            
            cl1=st.write("LCL : ",round(lclx,3))
            cl2=st.write("CL : ",round(clx,3))
            cl3=st.write("UCL : ",round(uclx,3))
            
            for i in range(0,n):
                a0.append(uclx)
                b0.append(lclx)
                c0.append(clx)

            st.write('')
            st.markdown(""" 
                ## X-BAR CHART  
                """)
            plt.plot(xbar, label = 'Data Points', marker = 'o')
            plt.plot(a0, label = 'UCL')
            plt.plot(b0, label = 'LCL')
            plt.plot(c0, label = 'CL')
            plt.xlabel("Sample Number")
            plt.ylabel("X-bar Values")
            plt.legend(loc = 'lower right', bbox_to_anchor =(1, 1.1) )

            st.pyplot()

            temp=1
            for i in xbar:
                if i<lclx or i>uclx:
                    st.warning("The process is not under control")
                    temp+=1
                    break
            if temp==1:
                st.success("The process is under control")

            a1=[]
            b1=[]
            c1=[]
        
            for i in range(0,n):
                a1.append(uclr)
                b1.append(lclr)
                c1.append(rbar)

            st.write('')
            st.markdown(""" 
                ### Control Limits for R Chart   
                """)
            clm1,clm2,clm3,clm4 = st.columns(4)

            with clm1:
                clrc1=st.write("LCL : ",round(lclr,3))
            with clm2:
                clrc2=st.write("CL : ",round(rbar,3))
            with clm3:
                clrc3=st.write("UCL : ",round(uclr,3))

            st.markdown(""" 
                ## R CHART  
                """)

            plt.plot(r, label = 'Data Points')
            plt.plot(a1, label ='UCL')
            plt.plot(b1, label = 'LCL')
            plt.plot(c1, label = 'CL')
            plt.xlabel("Sample Number")
            plt.ylabel("R Values")
            plt.legend(loc = 'lower right', bbox_to_anchor =(1, 1.1) )
            st.pyplot()
                    
            te1=1
            for j in r:
                if j<lclr or j>uclr:
                    st.warning("The process is not under control with respect to Range")
                    te1+=1
                    break
            
            if te1==1:
                st.success("The process is under control with respect to Range")

            st.write('')
            st.markdown(""" 
                #### Inference: 
                """)
            
            if temp==1 and te1==1:
                    st.success("The process is under control")
                    
            else:
                st.warning("The process is not under control")

        else:
            st.info("Upload a csv file or Enter a code")

    except Exception as e:
        st.error("Provide a valid dataset")

def xsd_chart(a):

    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        df = get_data(prev,'XSD - ')
        if df is None:
            st.markdown("""
            ### Description:
            """,True) 
            st.write("The X-SD chart is a type of control chart that is used to monitor the process mean and standard deviation. It consists of two charts: the X-chart, which plots the sample means, and the SD-chart, which plots the sample standard deviations.") 


        if df is not None:
            
            columns = df.columns.tolist()

            col1,col2,col3 = st.columns([4,1,4])
            
            with col1:    
                x_axis=st.selectbox('X-Bar',columns)
                y_axis=st.selectbox('Range',columns)
                no_of_sam=st.number_input("Sample Size",min_value=1.0,max_value=30.0,step=1.0)
            with col3:
                with open("D:/statmate/Assets /96007-signal-analysis.json") as f:
                    animation_data = json.load(f)
                    st_lottie(animation_data)
        
            xvals=df[x_axis].values.tolist()
            yvals=df[y_axis].values.tolist()

            A3vals = [1.880, 1.023, 0.729, 0.577, 0.483, 0.419, 0.373, 0.337, 0.308, 0.285, 0.266, 0.249, 0.235, 0.223, 0.212, 0.203, 0.195, 0.187]
            B3vals = [0, 0, 0.136, 0.184, 0.207, 0.223, 0.234, 0.243, 0.25, 0.253, 0.256, 0.258, 0.26, 0.261, 0.262, 0.263, 0.263, 0.264]
            B4vals = [3.267, 2.568, 2.266, 2.089, 1.97, 1.882, 1.815, 1.761, 1.716, 1.679, 1.646, 1.618, 1.594, 1.572, 1.552, 1.534, 1.518, 1.503]

            
            n=0
            xbar=[]
            sd=[]

            for i in xvals:
                if i==0 or i<0 or i>0:
                    xbar.append(i)
                    n+=1
                else:
                    break
                
            for j in yvals:
                if j==0 or j<0 or j>0:
                    sd.append(j)
                else:
                    break
            
            df = pd.DataFrame({
            'X': xbar,
            'SD': sd
                })

            with st.container():
                st.write(df.reset_index(drop=True).T.reset_index().rename(columns={'index': ''}).astype(object).set_index(''))
                
            st.markdown("""
            ### _The table values are_ 
            """,True)

            m=int(no_of_sam)
            A3=A3vals[m-1]
            B3=B3vals[m-1]
            B4=B4vals[m-1]  

            clm1, clm2, clm3, clm4 = st.columns([4,4,4,2])
            with clm1:
                st.write("A2 : ",A3)
            with clm2:
                st.write("D3 : ",B3)
            with clm3:
                st.write("D4 : ",B4)            

            clx=sum(xbar)/n
            sbar=sum(sd)/n

            uclx=clx+(sbar*A3)
            lclx=clx-(sbar*A3)
            if(lclx<0):
                lclx=0
            
            lcls=sbar*B3
            ucls=sbar*B4
                
            sample_num=[]
            for i in range(1,n+1):
                sample_num.append(i)

            a0=[]
            b0=[]
            c0=[]

            st.markdown(""" 
                ### Control Limits X-Bar Chart    
                """)
            
            cl1=st.write("LCL : ",round(lclx,3))
            cl2=st.write("CL : ",round(clx,3))
            cl3=st.write("UCL : ",round(uclx,3))
            
            for i in range(0,n):
                a0.append(uclx)
                b0.append(lclx)
                c0.append(clx)
            st.markdown(""" 
                ## X-BAR CHART  
                """)
            
            plt.plot(xbar, label = 'Data Points')
            plt.plot(a0, label = 'UCL')
            plt.plot(b0, label = 'LCL')
            plt.plot(c0, label = 'CL')
            plt.xlabel("Sample Number")
            plt.ylabel("X-bar")
            plt.legend(loc = 'lower right', bbox_to_anchor =(1, 1.1) )

            st.pyplot()

            temp=1
            for i in xbar:
                if i<lclx or i>uclx:
                    st.warning("The process is not under control with respect to mean")
                    temp+=1
                    break
            if temp==1:
                st.success("The process is under control with respect to mean")

            a1=[]
            b1=[]
            c1=[]
        
            for i in range(0,n):
                a1.append(ucls)
                b1.append(lcls)
                c1.append(sbar)

            st.markdown(""" 
                ### Control Limits for SD Chart    
                """)

            clrc1=st.write("LCL : ",round(lcls,3))
            clrc2=st.write("CL : ",round(sbar,3))
            clrc3=st.write("UCL : ",round(ucls,3))

            st.markdown(""" 
                ## R CHART  
                """)

            plt.plot(sd, label = 'Data Points', marker = 'o')
            plt.plot(a1, label = 'UCL')
            plt.plot(b1, label = 'LCL')
            plt.plot(c1, label = 'CL')
            plt.xlabel("Sample Number")
            plt.ylabel("R Values")
            plt.legend(loc = 'lower right', bbox_to_anchor =(1, 1.1) )

            st.pyplot()
                    
            te1=1
            for j in sd:
                if j<lcls or j>ucls:
                    st.warning("The process is not under control with respect to Standard Deviation")
                    te1+=1
                    break
            
            if te1==1:
                st.success("The process is under control with respect to Standard Deviation")

            st.markdown(""" 
                #### Inference: 
                """)
            
            if temp==1 and te1==1:
                    st.success("The process is under control")
                    time.sleep(7)
                    st.balloons()
                
            else:
                st.warning("The process is not under control")

        else:
            st.info("Upload a csv file or Enter a code")
    except Exception as e:
        st.error('Provide a valid dataset')

def np_chart(b):
   
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        df = get_data(prev,'NP - ')

        if df is None:
            st.markdown("""
            ### Description:
            """) 
            st.write("The NP chart is a type of control chart that is used to monitor the proportion of nonconforming items in a sample. It plots the number of nonconforming items versus the sample size.") 

        if df is not None:
 
            columns = df.columns.tolist()

            c1,c2,c3  = st.columns([4,1,4])
            with c1:
                st.write('')
                x_axis = st.selectbox('NP Values',columns)
                st.write('')
                st.write('')
                sam_size = st.number_input("Sample Size",0,1000,50,step=50)
                xval = df[x_axis].values.tolist()
            
            with c3:
                with open("D:/statmate/Assets /43312-graph-1.json") as f:
                    animation_data = json.load(f)
                    st_lottie(animation_data, height = 250)
                  
            n=0
            np=[]
            round_np = []

            for i in xval:
                if i==0 or i<0 or i>0:
                    np.append(i)
                    round_np.append(round(i,3))
                    n+=1
                else:
                    break
                
            clnp=sum(np)/n
            pbar=clnp/sam_size
            uclnp=clnp+(3*(math.sqrt((clnp*(1-pbar)))))
            lclnp=clnp-(3*(math.sqrt((clnp*(1-pbar)))))
            
            if(lclnp<0):
                lclnp=0
            
            sample_num=[]
            for i in range(1,n+1):
                sample_num.append(i)

            dnp = pd.DataFrame({'Mean Values': round_np}) 
            st.write(dnp.T)
            st.write('')
            st.write('')
            st.markdown(""" 
                ### Control Limits for NP Chart   
                """)
            
            cl1, cl2, cl3, cl4 = st.columns([4,4,4,2])

            with cl1:
                st.write("LCL : ",round(lclnp,3))
            with cl2:
                st.write("CL : ",round(clnp,3))
            with cl3:
                st.write("UCL : ",round(uclnp,3))

            a2=[]
            b2=[]
            c2=[]
            for i in range(0,n):
                a2.append(uclnp)
                b2.append(lclnp)
                c2.append(clnp)
            st.write('')
            st.markdown(""" 
                ## N-P CHART  
                """)
            plt.plot(np, label = 'Data Points', marker = 'o')
            plt.plot(a2, label = 'UCL')
            plt.plot(b2, label = 'LCL')
            plt.plot(c2, label = 'CL')
            plt.xlabel("Sample Number")
            plt.ylabel("N-P Values")
            plt.legend(loc = 'lower right', bbox_to_anchor =(1, 1.1) )

            st.pyplot()
            st.write('')
            
            temp1=1
            st.markdown(""" 
                #### Inference: 
                """)
            
            for i in np:
                if i<lclnp.real or i>uclnp.real:
                    st.warning("The process is not under control")
                    temp1+=1
                    break
            if temp1==1:
                st.success("The process is under control ")
                
        else:
            st.info("Upload a csv file or Enter a code")

    except Exception as e:
        st.error("Provide a valid dataset")

def p_chart(c):
    
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        df = get_data(prev,'P - ')

        if df is None:       
            st.markdown("""
            ### Description:
            """,True) 
            st.write("The P chart is a type of control chart that is used to monitor the proportion of nonconforming items in a sample when the sample size is constant. It plots the proportion of nonconforming items versus time or batch.") 

        if df is not None:
                
            columns = df.columns.tolist()
            c1,c2,c3 = st.columns([4,1,4])
            with c1: 
                st.write('')
                x_axis=st.selectbox('P Vlaues',columns)
                st.write('')
                sam_size=st.number_input("Sample Size",0,1000,50,step=50)
                xval=df[x_axis].values.tolist()

            with c3:
                 with open("D:/statmate/Assets /96883-smooth-chart.json") as f:
                    animation_data = json.load(f)
                    st_lottie(animation_data)
                               
            n=0
            np1=[]
            p=[]

            for i in xval:
                if i==0 or i<0 or i>0:
                    np1.append(i)
                    p.append(i/sam_size)
                    n+=1
                else:
                    break
            
            dp = pd.DataFrame({'P Values': p})   
            dp = dp.T      
            st.dataframe(dp, width = 900)

            clnp=sum(np1)/n
            pbar=clnp/sam_size
            uclp=pbar+(3*(math.sqrt((pbar*(1-pbar))/sam_size)))
            lclp=pbar-(3*(math.sqrt((pbar*(1-pbar))/sam_size)))

            if(lclp<0):
                lclp=0
        
            sample_num=[]
            for i in range(1,n+1):
                sample_num.append(i)

            st.markdown(""" 
                ### Control Limits for P Chart  
                """)
            
            cl1, cl2, cl3, cl4 = st.columns([4,4,4,2])
            with cl1:
                st.write("LCL : ",lclp)
            with cl2:
                st.write("CL : ",pbar)
            with cl3:
                st.write("UCL : ",uclp)

            a3=[]
            b3=[]
            c3=[]
            for i in range(0,n):
                a3.append(uclp)
                b3.append(lclp)
                c3.append(pbar)

            st.markdown(""" 
                ## P CHART  
                """)
            plt.plot(p, label = 'Data Points', marker= 'o' )
            plt.plot(a3, label = 'UCL')
            plt.plot(b3, label = 'LCL')
            plt.plot(c3, label = 'CL')
            plt.xlabel("Sample Number")
            plt.ylabel("P Values")
            plt.legend(loc = 'lower right', bbox_to_anchor =(1, 1.1) )

            st.pyplot()
            
            temp1=1
            st.markdown(""" 
                #### Inference: 
                """)
            for i in p:
                if i<lclp.real or i>uclp.real:
                    st.warning("The process is not under control")
                    temp1+=1
                    break
            
            if temp1==1:
                st.success("The process is under control ")
                time.sleep(7)
                st.balloons()

        else:
            st.info("Upload a csv file or Enter a code")
    except Exception as e:
        st.error("Provide a valid dataset")


def c_chart(d):
    
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        df = get_data(prev,'C - ')

        if df is None:
            st.markdown("""
            ### Description:
            """) 
            st.write("The C chart is a type of control chart that is used to monitor the count of nonconforming items in a sample. It plots the number of nonconforming items versus time or batch.") 
        
        if df is not None:
            c1,c2,c3 = st.columns([4,1,4])
            
            with c1:
                columns = df.columns.tolist() 
                st.write('')
                x_axis=st.selectbox('C Values',columns)
                xval=df[x_axis].values.tolist()
            with c3:
                with open("D:/statmate/Assets /86946-uiux-testing-the-app-on-the-phone.json") as f:
                    animation_data = json.load(f)
                    st_lottie(animation_data, height = 150)
                               
            n=0
            c=[]
            for i in xval:
                if i==0 or i<0 or i>0:
                    c.append(i)
                    n+=1
                else:
                    break        
            
            dc = pd.DataFrame({'P Values': c})         
            st.table(dc.T)

            ccl=sum(c)/n
            lclc=ccl-(3*(math.sqrt(ccl)))
            uclc=ccl+(3*(math.sqrt(ccl)))
            
            if(lclc<0):
                lclc=0

            st.markdown(""" 
                ### Control Limits for C Chart    
                """)
            co1, co2, co3, co4 =st.columns([4,4,1])

            with co1:
                st.write("LCL : ",round(lclc,3))
            with co2:
                st.write("CL : ",round(ccl,3))
            with co3:
                st.write("UCL : ",round(uclc,3))

            sample_num=[]
            for i in range(1,n+1):
                sample_num.append(i)
            
            a4=[]
            b4=[]
            c4=[]
            for i in range(0,n):
                a4.append(uclc)
                b4.append(lclc)
                c4.append(ccl)

            st.markdown(""" 
                ## C CHART  
                """)
            
            plt.plot(c, label = 'Data Points' , marker = 'o')
            plt.plot(a4, label = 'UCL')
            plt.plot(b4, label = 'LCL') 
            plt.plot(c4, label = 'CL')
            plt.xlabel("Sample Number")
            plt.ylabel("C")
            plt.legend(loc = 'lower right', bbox_to_anchor =(1, 1.1) )

            st.pyplot()

            temp1=1
            st.markdown(""" 
                #### Inference: 
                """)
            for i in c:
                if i<lclc.real or i>uclc.real:
                    st.warning("The process is not under control")
                    temp1+=1
                    break
            if temp1==1:
                st.success("The process is under control ")
                
        else:
            st.info("Upload a csv file or Enter a code")
    except Exception as e:
        st.error("Provide a valid dataset")                            

def vary_ucl_chart(c):
    
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        df = get_data(prev,'VUCL - ')

        if df is None:
            st.markdown("""
            ### Description:
            """,True) 
            st.write("The P chart with varying UCL is a type of control chart that is used when the sample size varies. The upper control limit (UCL) is adjusted for each sample size to account for the variability due to the sample size.") 
        
        if df is not None:
            columns = df.columns.tolist()
            cols = []
            for i in range(1,len(columns)):
                cols.append(columns[i])
                
            data, space, anim=st.columns([4,1,4])
            with data: 
                st.write('')
                x_axis=st.selectbox('Num of Inspected',cols)
                st.write('')
                st.write('')
                y_axis=st.selectbox('Num of Defectives',cols)

            with anim:
                with open('D:/statmate/Assets /135504-3d-statistics.json') as f:
                    animation_data = json.load(f)
                    st_lottie(animation_data)

            inspected = df[x_axis].values.tolist()
            defectives = df[y_axis].values.tolist()

            n=0
            ins = []
            defec = []

            for i in inspected:
                if i==0 or i<0 or i>0:
                    ins.append(i)
                    n+=1
                else:
                    break
                
            for j in defectives:
                if j==0 or j<0 or j>0:
                    defec.append(j)
                else:
                    break

            if st.checkbox("Show Data") :   

                dp = pd.DataFrame({
                    'No. of Inspected ' : ins,
                    'No. of Defectives ' : defec 
                })

                st.dataframe(dp.T, width = 900 )

            pbar = sum(defec) / sum(ins)
            p = []
            lclp = []
            uclp = []
            cl = []

            for i in range(n):
                lcl = ( pbar - 3 * (math.sqrt ((pbar * (1 - pbar)) / ins[i])))
                if lcl <= 0:
                    lcl = 0
                ucl = ( pbar + 3 * (math.sqrt ((pbar * (1 - pbar)) / ins[i])))
                pval = defec[i] / ins[i]

                lclp.append(lcl)
                uclp.append(ucl)
                cl.append(pbar)
                p.append(pval)

            sam_num = []
            for i in range (1,n+1):
                sam_num.append(i)

            st.markdown(""" 
                ### Control Limits    
                """)
            
            c1,c2,c3 = st.columns([1,2,1])

            with c2:
                st.metric('CL', round(pbar,3))

            df = pd.DataFrame({
                'Sample Number ':sam_num,
                'No of Defectives ':defec,
                'No of Inspected ': ins,
                'Fraction Defective (P)':p,
                'UCL ': uclp,
                'LCL ': lclp
            })

            st.dataframe(df,width = 1000)
            
            st.markdown(""" 
                ## P CHART (varying n values) 
                """)
            
            plt.plot(p, label = 'Data Points', marker = 'o')
            plt.plot(cl, label = 'CL')
            plt.plot(uclp, label = 'UCL')
            plt.plot(lclp, label ='LCL')
            plt.xlabel("Sample Number")
            plt.ylabel("P Values")
            plt.legend(loc = 'lower right', bbox_to_anchor =(1, 1.1) )

            st.pyplot()
            
            aa=1
            for i in p:
                if i < lcl or i > max(uclp):
                    aa+=1
                else:
                    pass

            st.markdown(""" 
                ## Inference 
                """)
            
            if aa == 1:
                st.success("The process is under control")
                
            else :
                st.warning("The process is under control")
        else:
            st.info("Upload a csv file or Enter a code")
    except Exception as e:
        st.error("Provide a valid dataset")