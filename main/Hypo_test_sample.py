import numpy as np
import pandas as pd
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json
from scipy import stats
from scipy.stats import chi2_contingency
from PIL import Image as pl

def loadlottieurl(url:str):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

def chi_sq():

    try:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file,index_col=0)
            stat, p, dof, expected = chi2_contingency(data)
            table1=pd.DataFrame({"Chi-square statistic" : [stat],"P-value" : [p],"Degrees of freedom" : [dof]})
            table2=pd.DataFrame(expected,index=data.index,columns=data.columns)
            st.write('Chi Square Test Values:')
            st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.write("Expected frequencies: ")
            st.write(table2,use_column_width=True)
            st.write("")
            st.write("Therefore for 0.05 level of significance")
            if p < 0.05:
                st.write("There is a significant association between the variables.")
            else:
                st.write("There is not enough evidence to conclude that there is a significant association between the variables.")
            result=[table1.values.tolist(),table2.values.tolist()]

        if not uploaded_file:
            st.markdown("""
            ### Description:
            """,True) 
            st.write("The chi-square test is used to determine whether there is a significant association between two categorical variables. It compares the observed frequencies with the expected frequencies under the null hypothesis of no association. The test statistic follows a chi-square distribution, and the p-value is calculated based on the degrees of freedom.")
            ax = st.checkbox("Show sample dataset")
            if ax:
                str_path = "C:/CODES/Project/Images/" + "CHI - " + ".PNG"
                img = pl.open(str_path)
                st.image(img, width=500)
            st.info("Upload a csv file")

    except Exception as e:
        st.error("Provide a valid dataset")

def z_test():

    try:
        uploaded_file = st.file_uploader("Upload the CSV file containing the sample values", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file,header=0)
            
            sample = data.values
            sample_size = len(sample)
            sample_mean = np.mean(sample)
            sample_std = np.std(sample)

            col1, col2, col3 = st.columns([4,1,4])
            with col1:
                st.write('')
                st.write('')
                alpha = st.slider("Select the alpha level:", 0.01, 0.1, 0.05)
                z_critical = stats.norm.ppf(1 - alpha)
                st.write('')
                st.write('')
                ttype=st.selectbox("Tail Type",["Two-Tailed","Right-Tailed","Left-Tailed"])
            
            with col3:
                with open("D:/statmate/Assets/98571-testing-checking-animation.json") as f:
                    animation_data = json.load(f)
                    st_lottie(animation_data)
                
            if(ttype=='Two-Tailed'):
                null_hypothesis_mean = st.number_input('Enter the population mean:')
                z_test = (np.mean(sample) - null_hypothesis_mean) / (sample_std / np.sqrt(sample_size))
                p_value = 2 * (1 - stats.norm.cdf(abs(z_test)))

                st.markdown("""
                ### Inference
                """,True)              
                if abs(z_test) > z_critical:
                    st.write("Reject null hypothesis : Sample is not drawn from the population")
                else:
                    st.write("Accept null hypothesis : Sample is drawn from the population")

                st.write("The error is:")
                if abs(z_test) > z_critical and sample_mean == null_hypothesis_mean:
                    st.write("Type 1 error : Rejecting null hypothesis when it's True")
                elif abs(z_test) <= z_critical and sample_mean != null_hypothesis_mean:
                    st.write("Type 2 error : Accepting null hypothesis when it's False")
                else:
                    st.write("No error")

            
            if(ttype=='Right-Tailed'):
                null_hypothesis_mean = st.number_input('Enter the null hypothesis mean:')
                alternative_hypothesis_mean = st.number_input('Enter the alternate hypothesis mean:')
            
                z_test = (np.mean(sample) - null_hypothesis_mean) / (sample_std / np.sqrt(sample_size))
                p_value = 1 - stats.norm.cdf(z_test)

                st.markdown("""
                ### Inference
                """,True)

                if z_test > stats.norm.ppf(1 - alpha):
                    st.write("Reject null hypothesis : Population mean greater than sample mean")
                else:
                    st.write("Accept null hypothesis : Population mean less than or equal to sample mean")

                st.write("The error is:")
                if z_test > stats.norm.ppf(1 - alpha) and sample_mean == null_hypothesis_mean:
                    st.write("Type 1 error : Rejecting null hypothesis when it's True")
                elif z_test <= stats.norm.ppf(1 - alpha) and sample_mean != alternative_hypothesis_mean:
                    st.write("Type 2 error : Accepting null hypothesis when it's False")
                else:
                    st.write("No error")

            if(ttype=='Left-Tailed'):
                null_hypothesis_mean = st.number_input('Enter the null hypothesis mean:')
                alternative_hypothesis_mean = st.number_input('Enter the alternate hypothesis mean:')
                
                z_test = (np.mean(sample) - null_hypothesis_mean) / (sample_std / np.sqrt(sample_size))
                p_value = stats.norm.cdf(z_test)
                
                st.markdown("""
                ### Inference
                """,True)
                if z_test < stats.norm.ppf(alpha):
                    st.write("Reject null hypothesis : Population mean less than sample mean")
                else:
                    st.write("Accept null hypothesis : Population mean greater than or equal to sample mean")

               
                st.write("The error is:")
                if z_test < stats.norm.ppf(alpha) and sample_mean == null_hypothesis_mean:
                    st.write("Type 1 error : Rejecting null hypothesis when it's True")
                elif z_test >= stats.norm.ppf(alpha) and sample_mean != alternative_hypothesis_mean:
                    st.write("Type 2 error : Accepting null hypothesis when it's False")
                else:
                    st.write("No error")
            
            result=[z_test,p_value]

        else:
            st.markdown("""
            ### Description:
            """, True) 
            st.write("The z-test is a hypothesis test used to determine whether a sample mean is significantly different from a known population mean. It is used when the population standard deviation is known. The test statistic follows a standard normal distribution, and the p-value is calculated based on the standard normal distribution.")
            ax = st.checkbox("Show sample dataset")
            if ax:
                str_path = "C:/CODES/Project/Images/" + "Z -" + ".PNG"
                img = pl.open(str_path)
                st.image(img, width=300)
            st.info("Upload a csv file")

    except Exception as e:
        st.error("Provide a valid dataset")

def t_test():

    try:
        data_file = st.file_uploader("Upload a CSV file containing the data for the two groups.", type=["csv"])
        if data_file is not None:
            data = pd.read_csv(data_file)
            st.write("Data:")
            st.dataframe(data, width = 900)
            c1, c2, c3 = st.columns([6,1,6])
            with c1:
                group1_col = st.selectbox("Select the column for group 1:", data.columns)
            with c3:
                group2_col = st.selectbox("Select the column for group 2:", data.columns)
            c4, c5 = st.columns([15,3])
            with c4:
                alpha = st.slider("Select the alpha level:", 0.01, 0.1, 0.05)
            group1 = data[group1_col]
            group2 = data[group2_col]
            t_stat, p_val = stats.ttest_ind(group1, group2)
            st.write("Results:")
            table1=pd.DataFrame({
            "Group 1 mean": [group1.mean()],
            "Group 2 mean": [group2.mean()],
            "T-statistic": [t_stat],
            "P-value" : [p_val]})
            
            st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.write("Conclusion:")
            if p_val < alpha:
                st.write("Reject null hypothesis at the alpha level of", alpha)
            else:
                st.write("Accept null hypothesis at the alpha level of", alpha)
            result=table1.values.tolist()

        else:
            st.markdown("""
            ### Description:
            """, True) 
            st.write("A t-test is a statistical hypothesis test that is used to determine whether there is a significant difference between the means of two groups of data. It is used when the sample size is small and the population standard deviation is unknown. The t-test calculates a t-value, which is then compared to a critical value from a t-distribution to determine whether the difference between the two means is statistically significant. The t-test is widely used in various fields, including science, medicine, and social sciences")
        
            ax = st.checkbox("Show sample dataset")
            if ax:
                str_path = "C:/CODES/Project/Images/" + "T - " + ".PNG"
                img = pl.open(str_path)
                st.image(img, width=400)
            st.info("Upload a csv file")


    except Exception as e:
        st.error("Provide a valid dataset")
