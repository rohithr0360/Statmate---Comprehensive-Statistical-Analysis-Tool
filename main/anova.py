import scipy.stats
import streamlit as st
import pandas as pd
import pymongo
from pymongo import MongoClient
import random
from PIL import Image as pl
import certifi
ca = certifi.where()

#shaun_434546

client = MongoClient('mongodb+srv://Str_2364353:mjo2h5KxnbV5EMJG@cluster0.yvnzvii.mongodb.net/',  tls=True, tlsAllowInvalidCertificates=True)

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

    if not uploaded_file and not data_from_db:
        st.markdown("""
        ### Description:
        """,True) 
        if cv == 'CRD - ':
            st.write("One-way ANOVA or Completely Randomised Design ANOVA table is used to compare the means of three or more groups. It tests whether the variation among group means is significantly greater than the variation within groups.")
        if cv == 'RBD - ':
            st.write("Two-way ANOVA or Randomised Block Design is used when there are two independent variables. It tests whether the main effects of each independent variable and the interaction effect between them are significant.")
        if cv == 'LSD - ':
            st.write("Repeated measures ANOVA or Latin Square Design is used when the same group of subjects is measured multiple times under different conditions. It tests whether there is a significant difference between the means of the repeated measures and whether the effect of the independent variable is significant.")

        ax = st.checkbox("Show sample dataset")
        if ax:
            str_path = "C:/CODES/Project/Images/" + cv + ".PNG"
            img = pl.open(str_path)
            st.image(img, width=350)


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

    
def completely_randomized_design():
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        data = get_data(prev,"CRD - ")
        if data is not None:
            row_total = []
            for i in data:
                row_total.append(sum(i))

            x_bar = sum(row_total)/len(data)
            T = sum(row_total)
            h,k, =  len(data),len(data[1])
            n = h*k

            V = 0
            for i in data:
                for j in i:
                    V += j**2

            V = V - (T**2)/(h*k)

            Vb = 0
            for i in row_total:
                Vb += i**2

            Vb = Vb/k - (T**2)/(h*k)

            Vw = V - Vb

            sum_of_squares,dof = [Vb,Vw],[h-1,h*k-h]

            mean_square = []
            for i in range(len(dof)):
                mean_square.append(sum_of_squares[i]/dof[i])

            F = mean_square[0]/mean_square[i]   

            a = scipy.stats.f.ppf(q = 1-0.05,dfn = dof[0],dfd = dof[1])

            table1 = pd.DataFrame({
                "SUM OF SQUARES" : sum_of_squares,
                "DEGREES OF FREEDOM" : dof,
                "MEAN SQUARE" : mean_square
            })

            table2 = pd.DataFrame({
                "F": [F],
                "P": [a]
            }, index=["row1"])

            st.write(" ")
 

     
            st.markdown("""
                ### ANOVA Table 
            """)
            st.write("")

            st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.markdown(f'<center>{table2.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.write("")

            st.markdown("""
                ### OUTCOME
            """)
            st.write("")
            if F < a:

                st.info("Calculated value[F]     <     Table value[P]")
                st.success("Accept H0")
            else:
                st.info("Calculated value[F]     >     Table value[P]")
                st.warning("Reject H0")

    except Exception as e:
        st.error("Provide a valid dataset")
        st.write(e)


def randomized_block_design():
    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        d = get_data(prev,"RBD - ")
        if d is not None:
            min_element = d[0][0]
            data = []
            for i in d:
                d1 = []
                for j in i:
                    temp = j-min_element
                    d1.append(temp)
                data.append(d1)

            
            h,k, =  len(data),len(data[1])
            n = h*k

            row_total = []
            Ti_square = []
            Tj_square = []
            Xij_square = []

            for i in data:
                row_total.append(sum(i))
                Ti_square.append((sum(i)**2)/k)
                total = 0
                for j in i:
                    total += j**2
                Xij_square.append(total)

            for i in range(len(data[0])):
                total = 0
                for j in range(len(data)):
                    total += data[j][i]
                Tj_square.append(total**2/h)

            x_bar = sum(row_total)/len(data)
            T = sum(row_total)


            V = sum(Xij_square) - (T**2)/(n)

            Vr = (sum(Ti_square)) - (T**2)/(n)

            Vl = (sum(Tj_square)) - (T**2)/(n)

            Ve = V - Vr- Vl

            sum_of_squares,dof = [Vr,Vl,Ve],[h-1,k-1,(k-1)*(h-1)]
            mean_square = []
            for i in range(len(dof)):
                mean_square.append(sum_of_squares[i]/dof[i])

            f = []

            if mean_square[0] <  mean_square[2]:
                f.append((mean_square[2])/mean_square[0])


            if mean_square[1] < mean_square[2]:
                f.append((mean_square[2])/mean_square[1])


            if mean_square[0] >= mean_square[2]:
                f.append((mean_square[0])/mean_square[2])

            
            if mean_square[1] >= mean_square[2]:
                f.append((mean_square[1])/mean_square[2])

            a = [] 

            if (mean_square[0] >= mean_square[2] and mean_square[1] >= mean_square[2]) or (mean_square[0] == mean_square[2] and mean_square[1] < mean_square[2]) or (mean_square[0] < mean_square[2] and mean_square[1] == mean_square[2]):

                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[0],dfd = dof[2]))
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[1],dfd = dof[2]))

            #or (dof[0] == dof[2] and dof[1] < dof[2]) or (dof[0] < dof[2] and dof[1] == dof[2])
                
            elif mean_square[0] < mean_square[2]:
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[2],dfd = dof[0]))
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[2],dfd = dof[1]))



        # or (dof[0] == dof[2] and dof[1] > dof[2]) or (dof[0] > dof[2] and dof[1] == dof[2]

            table1 = pd.DataFrame({
                "SUM OF SQUARES" : sum_of_squares,
                "DEGREES OF FREEDOM" : dof,
                "MEAN SQUARE" : mean_square
            })

            table2 = pd.DataFrame({
                "F": f,
                "P": a
            })

            st.write(" ")
            st.markdown("""
            ### ANOVA Table 
            """)
            st.write("")

            st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.markdown(f'<center>{table2.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.write("")

            st.markdown("""
            ### OUTCOME
            """)

            st.write(" ")

            if f[0] < a[0] and f[1] < a[1] :
                st.info(" F1 < P1     and     F2 < P2")
                st.success("Accept H0")

            if f[0] > a[0] and f[1] < a[1] :
                st.info(" F1 > P1     and     F2 < P2")
                st.error("Reject H0")

            if f[0] < a[0] and f[1] > a[1] :
                st.info(" F1 < P1     and     F2 > P2")
                st.error("Reject H0")
                
            if f[0] > a[0] and f[1] > a[1] :
                st.info(" F1 > P1     and     F2 > P2")
                st.error("Reject H0")

    except Exception as e:
        st.error("Provided dataset is not valid")



def latin_block_design():

    try:
        prev = collection.find_one({"_id": 10001})["Previous"]
        d = []
        d1 = get_data(prev,"LSD - ")

        if d1 is not None:

            xy = d1
            for i in d1:
                temp = []
                for j in i:
                    temp.append(int(j[1:]))
                d.append(temp)


            lists = []
            for i in range(len(d1)):
                lists.append(list())

            samples = []

            
            for i in range(len(d1)):
                val = d1[i][0]
                samples.append(val[0])
        
            for i in d1:
                for j in i:
                    x = j[0]
                    index = samples.index(x)
                    x = lists[index]
                    x.append(int(j[1:]))

            data = []
            min_element = d[0][0]
            for i in d:
                d1 = []
                for j in i:
                    temp = j-min_element
                    d1.append(temp)
                data.append(d1)

            empty = []
            min_element = d[0][0]
            for i in lists:
                d1 = []
                for j in i:
                    temp = j-min_element
                    d1.append(temp)
                empty.append(d1)

            lists = empty

            
            h,k, =  len(data),len(data[1])
            n = h
            N = h*k

            row_total = []
            Ti_square = []
            Tj_square = []
            Tk_square = []
            Xij_square = []
            latin_row_total = []
            
            for i in data:
                row_total.append(sum(i))
                Ti_square.append((sum(i)**2)/k)
                total = 0
                for j in i:
                    total += j**2
                Xij_square.append(total)

            for i in range(len(data[0])):
                total = 0
                for j in range(len(data)):
                    total += data[j][i]
                Tj_square.append(total**2/h)
            
            for i in lists:
                latin_row_total.append(sum(i))
                Tk_square.append(((sum(i))**2)/n)


                

            x_bar = sum(row_total)/len(data)
            T = sum(row_total)

            sum_latin = sum(latin_row_total)

            V = sum(Xij_square) - (T**2)/(h*k)

            Vr = (sum(Ti_square)) - (T**2)/(h*k)

            Vc = (sum(Tj_square)) - (T**2)/(h*k)

            Vl = (sum(Tk_square)) - (T**2)/(h*k)


            Ve = V - Vr- Vc - Vl

            sum_of_squares,dof =[Vr,Vc,Vl,Ve],[n-1,n-1,n-1,(n-1)*(n-2)] 

            mean_square = []
            for i in range(len(dof)):
                mean_square.append(sum_of_squares[i]/dof[i])

            f = []

            if mean_square[0] <  mean_square[3]:
                f.append((mean_square[3])/mean_square[0])
            if mean_square[0] >  mean_square[3]:
                f.append((mean_square[0])/mean_square[3])

            if mean_square[1] < mean_square[3]:
                f.append((mean_square[3])/mean_square[1])
            if mean_square[1] >  mean_square[3]:
                f.append((mean_square[1])/mean_square[3])

            if mean_square[2] <  mean_square[3]:
                f.append((mean_square[3])/mean_square[2])
            if mean_square[2] > mean_square[3]:
                f.append((mean_square[2])/mean_square[3])

            a = [] 

            if (mean_square[0] >= mean_square[3] and mean_square[1] >= mean_square[3]) or (mean_square[0] == mean_square[3] and mean_square[1] < mean_square[2]) or (mean_square[0] < mean_square[2] and mean_square[1] == mean_square[2]):

                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[0],dfd = dof[2]))
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[1],dfd = dof[2]))

                
            if mean_square[0] < mean_square[3]:
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[3],dfd = dof[0]))
            if mean_square[0] > mean_square[3]:
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[0],dfd = dof[3]))

            if mean_square[1] < mean_square[3]:
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[3],dfd = dof[1]))
            if mean_square[1] > mean_square[3]:
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[1],dfd = dof[3]))

            if mean_square[2] < mean_square[3]:
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[3],dfd = dof[2]))
            if mean_square[2] > mean_square[3]:
                a.append(scipy.stats.f.ppf(q = 1-0.05,dfn = dof[2],dfd = dof[3]))

            table1 = pd.DataFrame({
                "SUM OF SQUARES" : sum_of_squares,
                "DEGREES OF FREEDOM" : dof,
                "MEAN SQUARE" : mean_square
            })

            table2 = pd.DataFrame({
                "F": f,
                "P": a
            })

            st.write(" ")
            st.markdown("""
            ### ANOVA Table 
            """)
            st.write("")

            st.markdown(f'<center>{table1.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.markdown(f'<center>{table2.to_html(index=False)}</center>', unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.write("")

            st.markdown("""
            ### OUTCOME
            """)
            st.write("")


            if f[0] < a[0] and f[1] < a[1] and f[2]<a[2] :
                st.info(" F1 < P1     and     F2 < P1     and     F3 < P3") 
                st.success("Accept H0")

            if f[0] > a[0] and f[1] < a[1] and f[2]<a[2] :
                st.info(" F1 > P1     and     F2 < P1     and     F3 < P3") 
                st.error("Reject H0")

            if f[0] < a[0] and f[1] > a[1] and f[2]<a[2] :
                st.info(" F1 < P1     and     F2 > P1     and     F3 < P3") 
                st.error("Reject H0")

            if f[0] < a[0] and f[1] < a[1] and f[2] > a[2] :
                st.info(" F1 < P1     and     F2 < P1     and     F3 > P3") 
                st.error("Reject H0")

            if f[0] < a[0] and f[1] > a[1] and f[2] > a[2] :
                st.info(" F1 < P1     and     F2 > P1     and     F3 > P3") 
                st.error("Reject H0")
                        
            if f[0] > a[0] and f[1] < a[1] and f[2] > a[2] :
                st.info(" F1 > P1     and     F2 < P1     and     F3 > P3") 
                st.error("Reject H0")

            if f[0] > a[0] and f[1] > a[1] and f[2] < a[2] :
                st.info(" F1 < P1     and     F2 < P1     and     F3 > P3") 
                st.error("Reject H0")

            if f[0] > a[0] and f[1] > a[1] and f[2] > a[2] :
                st.info(" F1 > P1     and     F2 > P1     and     F3 > P3") 
                st.error("Reject H0")

    except Exception as e:
        st.error("Provided dataset is not valid")
