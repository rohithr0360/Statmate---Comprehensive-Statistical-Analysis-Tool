import numpy as np
import pandas as pd
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json
from scipy import stats
from scipy.stats import chi2_contingency
from PIL import Image as pl
import pymongo
from pymongo import MongoClient
import random

def loadlottieurl(url:str):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

client = MongoClient("mongodb+srv://Str_2364353:mjo2h5KxnbV5EMJG@cluster0.yvnzvii.mongodb.net/?retryWrites=true&w=majority")
db = client['statmate']
collection = db['Data_collection1']


def get_data(prev,cv):
    v1 = cv

    ve = cv
    up,d = st.columns([6,4])
    with up:
        uploaded_file = st.file_uploader("Choose a file",type = "csv")
    with d:
        data_from_db = st.text_input("Retrive data from DB ")
        cv1 = cv + str(1001)
        st.write("Sample Code: ",cv1)

    if not uploaded_file and not data_from_db and cv == 'T - ':
        st.markdown("""
        ### Description:
        """, True) 
        st.write("A t-test is a statistical hypothesis test that is used to determine whether there is a significant difference between the means of two groups of data. It is used when the sample size is small and the population standard deviation is unknown. The t-test calculates a t-value, which is then compared to a critical value from a t-distribution to determine whether the difference between the two means is statistically significant. The t-test is widely used in various fields, including science, medicine, and social sciences")
    
        ax = st.checkbox("Show sample dataset")
        if ax:
            str_path = "C:/CODES/Project/Images/" + "T - " + ".PNG"
            img = pl.open(str_path)
            st.image(img, width=400)

    if not uploaded_file and not data_from_db and cv == 'CHI - ':
        st.markdown("""
        ### Description:
        """,True) 
        st.write("The chi-square test is used to determine whether there is a significant association between two categorical variables. It compares the observed frequencies with the expected frequencies under the null hypothesis of no association. The test statistic follows a chi-square distribution, and the p-value is calculated based on the degrees of freedom.")
        ax = st.checkbox("Show sample dataset")
        if ax:
            str_path = "C:/CODES/Project/Images/" + "CHI - " + ".PNG"
            img = pl.open(str_path)
            st.image(img, width=500)

    if not uploaded_file and not data_from_db and cv == 'Z - ':
        st.markdown("""
        ### Description:
        """, True) 
        st.write("The z-test is a hypothesis test used to determine whether a sample mean is significantly different from a known population mean. It is used when the population standard deviation is known. The test statistic follows a standard normal distribution, and the p-value is calculated based on the standard normal distribution.")
        ax = st.checkbox("Show sample dataset")
        if ax:
            str_path = "C:/CODES/Project/Images/" + "Z -" + ".PNG"
            img = pl.open(str_path)
            st.image(img, width=300)

    if uploaded_file and not data_from_db:

        df = pd.read_csv(uploaded_file,header = None)

        if cv == 'Z - ':
            data = df.values.tolist()
        
        else:
            data = [df.columns.tolist()] + df.values.tolist()

        data = data[1:]

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
                if ax and cv == 'Z - ':
                    xyz = pd.DataFrame(data)
                    st.write(xyz)
                if ax and (cv == 'CHI - ' or cv == 'T - '):
                    xyz = pd.DataFrame(data[1:],columns = data[0])
                    st.write(xyz)
            
            with d5:

                st.write("Data code: ",cv + str(rand_num))

        else:
            up2,d5 = st.columns([75,25])
            with up2:
                ax = st.checkbox("Show dataset")
                if ax and cv == 'Z - ':
                    xyz = pd.DataFrame(data)
                    st.write(xyz)
                if ax and (cv == 'CHI - ' or cv == 'T - '):
                    xyz = pd.DataFrame(data[1:],columns = data[0])
                    st.write(xyz)
            with d5:
                st.write("Data code: ",prev)


        return data

    if data_from_db and not uploaded_file:
        
        data = collection.find_one({"_id": str(data_from_db)})["data"]

        up2,d5 = st.columns([75,25])
        with up2:
            ax = st.checkbox("Show dataset")
            if ax and cv == 'Z - ':
                xyz = pd.DataFrame(data)
                st.write(xyz)
            if ax and (cv == 'CHI - ' or cv == 'T - '):
                xyz = pd.DataFrame(data[1:],columns = data[0])
                st.write(xyz)
        with d5:
            st.write("Data code: ",data_from_db)
        
        return data

    if data_from_db and  uploaded_file:
        st.info("Select anyone option")

    else:
        st.info("Upload a csv file or Enter a code")

def chi_sq():
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        d = get_data(prev,"CHI - ")
        if d is not None:

            dt =  [[int(x) for x in sublist] for sublist in d[1:]]

            data = pd.DataFrame(dt, columns=d[0])
            stat, p, dof, expected = chi2_contingency(data)
            table1=pd.DataFrame({"Chi-square statistic" : [stat],"P-value" : [p],"Degrees of freedom" : [dof]})
            table2=pd.DataFrame(expected,index=data.index,columns=data.columns)
            st.markdown("""
        ##### Chi Square Test Values:
        """,True) 
            st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.markdown("""
        ##### Expected frequencies:
        """,True) 

            st.markdown(f'<center>{table2.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.markdown("""
        ##### Inference:
        """,True) 
            st.info("Therefore for 0.05 level of significance")
            if p < 0.05:
                st.success("There is a significant association between the variables.")
            else:
                st.warning("There is not enough evidence to conclude that there is a significant association between the variables.")
        
    except Exception as e:
        st.error("Provide a valid dataset")


def z_test():
    
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        d = get_data(prev,"Z - ")
        if d is not None:

            dt =  [[int(x) for x in sublist] for sublist in d]
            data = pd.DataFrame(dt)
            
            sample = data.values
            sample_size = len(sample)
            sample_mean = np.mean(sample)
            sample_std = np.std(sample)

            col1, col2, col3 = st.columns([4,1,4])
            with col1:
                st.write('')
        
                ttype=st.selectbox("Tail Type",["Two-Tailed","Right-Tailed","Left-Tailed"])
                st.write('')
            
                alpha = st.slider("Select the alpha level:", 0.01, 0.1, 0.05)
                st.write('')
                z_critical = stats.norm.ppf(1 - alpha)
            
            with col3:
                st.write('')
                st.write('')
                with open("D:/statmate/Assets/cloud-monitoring.json") as f:
                    animation_data = json.load(f)
                    st_lottie(animation_data)
                    
                
            if(ttype=='Two-Tailed'):
                with col1:
                    null_hypothesis_mean = st.number_input('Enter the population mean:')
                z_test = (np.mean(sample) - null_hypothesis_mean) / (sample_std / np.sqrt(sample_size))
                p_value = 2 * (1 - stats.norm.cdf(abs(z_test)))

                st.write('')
                st.write('')
                st.markdown("""
            #### Results:
            """,True) 
                table1=pd.DataFrame({
                "Sample Mean": [np.mean(sample)],
                "Null Hypothsis Mean": [null_hypothesis_mean],
                "T-statistic": [z_test],
                "P-value" : [p_value]})
                
                st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
                st.write('')
                
                st.markdown("""
                #### Inference
                """,True)              
                if abs(z_test) > z_critical:
                    st.warning("Reject null hypothesis : Sample is not drawn from the population")
                else:
                    st.success("Accept null hypothesis : Sample is drawn from the population")

        
                if abs(z_test) > z_critical and sample_mean == null_hypothesis_mean:
                    st.write("Type 1 error : Rejecting null hypothesis when it's True")
                elif abs(z_test) <= z_critical and sample_mean != null_hypothesis_mean:
                    st.write("Type 2 error : Accepting null hypothesis when it's False")
                else:
                    st.success("No error")

            
            if(ttype=='Right-Tailed'):
                with col1:
                    null_hypothesis_mean = st.number_input('Enter the null hypothesis mean:')
                alternative_hypothesis_mean = st.number_input('Enter the alternate hypothesis mean:')
            
                z_test = (np.mean(sample) - null_hypothesis_mean) / (sample_std / np.sqrt(sample_size))
                p_value = 1 - stats.norm.cdf(z_test)
                st.write('')
                
                st.markdown("""
            #### Results:
            """,True) 
                table1=pd.DataFrame({
                "Sample Mean": [np.mean(sample)],
                "Null Hypothsis Mean": [null_hypothesis_mean],
                "T-statistic": [z_test],
                "P-value" : [p_value]})
                st.write('')
                st.write('')
                st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
                st.markdown("""
                #### Inference
                """,True)

                if z_test > stats.norm.ppf(1 - alpha):
                    st.warning("Reject null hypothesis : Population mean greater than sample mean")
                else:
                    st.success("Accept null hypothesis : Population mean less than or equal to sample mean")

                if z_test > stats.norm.ppf(1 - alpha) and sample_mean == null_hypothesis_mean:
                    st.warning("Type 1 error : Rejecting null hypothesis when it's True")
                elif z_test <= stats.norm.ppf(1 - alpha) and sample_mean != alternative_hypothesis_mean:
                    st.warning("Type 2 error : Accepting null hypothesis when it's False")
                else:
                    st.success("No error")

            if(ttype=='Left-Tailed'):
                with col1:
                    null_hypothesis_mean = st.number_input('Enter the null hypothesis mean:')
                alternative_hypothesis_mean = st.number_input('Enter the alternate hypothesis mean:')
                
                z_test = (np.mean(sample) - null_hypothesis_mean) / (sample_std / np.sqrt(sample_size))
                p_value = stats.norm.cdf(z_test)
                st.write('')
                
                st.markdown("""
            #### Results:
            """,True) 
                table1=pd.DataFrame({
                "Sample Mean": [np.mean(sample)],
                "Null Hypothsis Mean": [null_hypothesis_mean],
                "T-statistic": [z_test],
                "P-value" : [p_value]})

                st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
                st.write('')
                st.write('')
                st.markdown("""
                #### Inference
                """,True)
                if z_test < stats.norm.ppf(alpha):
                    st.warning("Reject null hypothesis : Population mean less than sample mean")
                else:
                    st.success("Accept null hypothesis : Population mean greater than or equal to sample mean")


                if z_test < stats.norm.ppf(alpha) and sample_mean == null_hypothesis_mean:
                    st.warning("Type 1 error : Rejecting null hypothesis when it's True")
                elif z_test >= stats.norm.ppf(alpha) and sample_mean != alternative_hypothesis_mean:
                    st.warning("Type 2 error : Accepting null hypothesis when it's False")
                else:
                    st.success("No error")
        
    except Exception as e:
        st.error("Provide a valid dataset")



def t_test():
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        d = get_data(prev,"T - ")
        if d is not None:

            data = pd.DataFrame(d[1:], columns=d[0])
            c1, c2, c3 = st.columns([6,1,6])
            
            with c1:
                st.write(" ")
                group1_col = st.selectbox("Select the column for group 1:", data.columns)
                group2_col = st.selectbox("Select the column for group 2:", data.columns)
                alpha = st.slider("Select the alpha level:", 0.01, 0.1, 0.05)
        
            with c3:
                with open("D:/statmate/Assets/98571-testing-checking-animation.json") as f:
                    animation_data = json.load(f)
                    st_lottie(animation_data)


            group10 = data[group1_col]
            group20 = data[group2_col]

            group1 = [int(x) for x in group10]
            group2 = [int(x) for x in group20]

            t_stat, p_val = stats.ttest_ind(group1, group2)

            st.markdown("""
        ##### Results:
        """,True) 
            table1=pd.DataFrame({
            "Group 1 mean": [np.array(group1).mean()],
            "Group 2 mean": [np.array(group2).mean()],
            "T-statistic": [t_stat],
            "P-value" : [p_val]})
            
            st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.markdown("""
        ##### Conclusion:
        """,True) 
            if p_val < alpha:
                st.warning(f"Reject null hypothesis at the alpha level of {alpha}")
            else:
                st.success(f"Accept null hypothesis at the alpha level of {alpha}")

    except Exception as e:
        st.error("Provide a valid dataset")