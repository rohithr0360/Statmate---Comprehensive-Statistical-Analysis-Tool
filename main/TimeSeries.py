import streamlit as st
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sympy import symbols, Eq, solve
import plotly.graph_objs as go
import json
from streamlit_lottie import st_lottie

a=1

if a==1:

    def linear_trend():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()

                c1, c2, c3 = st.columns([4, 1, 4])

                with c1:
                    st.write('')
                    st.write('')
                    x_axis=st.selectbox('X',columns)
                    st.write('')
                    y_axis=st.selectbox('Y',columns)
                    
                    xvals=df[x_axis].values.tolist()
                    yvals=df[y_axis].values.tolist()
                    
                with c3:
                    with open("D:/statmate/Assets/137960-bar-chart-with-arrow-and-a-star.json") as f:
                        animation_data = json.load(f)
                        st_lottie(animation_data)
                
                n=0
                xv=[]
                yv=[]

                for i in xvals:
                    if i==0 or i<0 or i>0:
                        xv.append(i)
                        n+=1
                    else:
                        break
                    
                for j in yvals:
                    if j==0 or j<0 or j>0:
                        yv.append(j)
                    else:
                        break

                sumx=sum(xv)
                sumy=sum(yv)

                xy=[]
                x2=[]
                
                for i in range(0,n):
                    temp=(xv[i] * yv[i])
                    xy.append(temp)
                    t=(xv[i] * xv[i])
                    x2.append(t)
                
                sumxy=sum(xy)
                sumx2=sum(x2)
                            
                a1 = sumx
                b1 = n
                c1 = sumy

                a2 = sumx2
                b2 = sumx
                c2 = sumxy
                
                # Create a coefficient matrix and a constant matrix
                A = np.array([[a1, b1], [a2, b2]])
                B = np.array([c1, c2])

                # Solve the system of equations
                xsol = np.linalg.solve(A, B)

                b = xsol[0]
                a = xsol[1]
                st.write
                st.write('')

                aval,bval=st.columns(2)
                with aval:
                    st.write("a =", round(a,2))
                with bval:
                    st.write("b =", round(b,2))

                tnd = []

                for i in xv:
                    t = a + ( i * b )
                    tnd.append(t)

                df = pd.DataFrame({
                'X': xv,
                'Y': yv,
                'X2':x2,
                'XY':xy,
                'Trend':tnd
                })

                st.dataframe(df,width=900)
                
                st.markdown(""" 
                ### Summation    
                """)
                
                data = {'X': [sumx], 'Y': [sumy], 'X2': [sumx2], 'XY': [sumxy]}
                df = pd.DataFrame(data)

                st.dataframe(df,width=900)

                plt.plot(yv)
                plt.plot(tnd)
                plt.title("Trend Chart")
                plt.xlabel("X Values")
                plt.ylabel("Y Values")
                plt.legend()

                st.pyplot()

        except Exception as e:
            st.warning("Provide a valid dataset")
            st.write(e)
        
    def para_trend():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()
        
                c1, c2, c3 = st.columns([4, 1, 4])

                with c1:
                    st.write('')
                    st.write('')
                    x_axis=st.selectbox('X',columns)
                    st.write('')
                    st.write('')
                    y_axis=st.selectbox('Y',columns)
                    
                    year = df[x_axis].values.tolist()
                    yvals = df[y_axis].values.tolist()
                    
                with c3:
                    with open("D:/statmate/Assets/82738-chart-grow-up.json") as f:
                        animation_data = json.load(f)
                        st_lottie(animation_data)

                n=0
                xv=[]
                yv=[]

                for i in year:
                    if i==0 or i<0 or i>0:
                        xv.append(i-(year[0]-1))
                        n+=1
                    else:
                        break
                    
                for j in yvals:
                    if j==0 or j<0 or j>0:
                        yv.append(j)
                    else:
                        break

                sumx=sum(xv)
                sumy=sum(yv)

                xy=[]
                x2=[]
                x2y=[]
                x3=[]
                x4=[]
                
                for i in range(0,n):
                    temp=(xv[i] * yv[i])
                    xy.append(temp)
                    t1=(xv[i] * xv[i])
                    x2.append(t1)
                    t2=(t1 * yv[i])
                    x2y.append(t2)
                    t3=(t1 * xv[i])
                    x3.append(t3)
                    t4=(t3 * xv[i])
                    x4.append(t4)

                sumxy=sum(xy)
                sumx2=sum(x2)
                sumx2y=sum(x2y)
                sumx3=sum(x3)
                sumx4=sum(x4)
    
                # get user input for the coefficients of the equations
                a11 = n
                a12 = sumx
                a13 = sumx2
                b1 = sumy

                a21 = sumx
                a22 = sumx2
                a23 = sumx3
                b2 = sumxy

                a31 = sumx2
                a32 = sumx3
                a33 = sumx4
                b3 = sumx2y

                # define symbols for the equations
                x, y, z = symbols('x,y,z')

                # create equations using user input
                eq1 = Eq((a11*x + a12*y + a13*z), b1)
                eq2 = Eq((a21*x + a22*y + a23*z), b2)
                eq3 = Eq((a31*x + a32*y + a33*z), b3)

                # solve equations and extract values of x, y, and z
                solutions = solve((eq1, eq2, eq3), (x, y, z))
                x_value = solutions[x]
                y_value = solutions[y]
                z_value = solutions[z]

                x_val = round(float(x_value),3)
                y_val = round(float(y_value),3)
                z_val = round(float(z_value),3)

                x_eq = str(x_val)
                y_eq = str(y_val)
                z_eq = str(z_val)

                tnd = []
                for i in xv:
                    eqn = (x_value + (i * y_value) + ((i)**2) * z_value)
                    tnd.append(eqn)


                df = pd.DataFrame({
                'X': xv,
                'Y': yv,
                'X2':x2,
                'XY':xy,
                'X2Y':x2y,
                'X3':x3,
                'X4':x4,
                'Trend':tnd
                })

                st.write('')
                st.dataframe(df, width = 900)
                
                st.markdown(""" 
                ### Summation    
                """)
                
                df = pd.DataFrame({
                    'X': [sumx],
                    'Y': [sumy],
                    'X2': [sumx2],
                    'XY': [sumxy],
                    'X2Y': [sumx2y],
                    'X3': [sumx3],
                    'X4': [sumx4]
                })

                # Create a DataFrame from the dictionary
                st.dataframe(df,width = 900)

                a1,a2,a3 = st.columns(3)
                st.write('')

                with a1:
                    st.write("a =", x_val  )
                with a2:
                    st.write("b =", y_val )
                with a3:
                    st.write("c =", z_val )

                st.write('')
                st.write('') 

                st.write("The equation is : y = ", x_val , ' + ', y_val,' x  + ',z_val," x^2")
            
                plt.plot(yv,label = 'Data Points')
                plt.plot(tnd,label= 'Trend Line' )
                plt.title("Trend Chart")
                plt.xlabel("X Values")
                plt.ylabel("Y Values")
                plt.legend()

                st.pyplot()

        except Exception as e:
            st.warning("Provide a valid dataset")
            
    def expo_trend():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()

                c1, c2, c3 = st.columns([4, 1, 4])

                with c1:
                    st.write('')
                    st.write('')
                    x_axis=st.selectbox('X',columns)
                    st.write('')
                    st.write('')
                    y_axis=st.selectbox('Y',columns)
                    
                    year = df[x_axis].values.tolist()
                    yvals = df[y_axis].values.tolist()
                    
                with c3:
                    with open("D:/statmate/Assets/126403-statistics.json") as f:
                        animation_data = json.load(f)
                        st_lottie(animation_data)

                n=0
                xv=[]
                yv=[]

                for i in year:
                    if i==0 or i<0 or i>0:
                        n+=1
                    else:
                        break
                
                for i in range(1,n+1):
                    xv.append(i)
                    
                for j in yvals:
                    if j==0 or j<0 or j>0:
                        yv.append(j)
                    else:
                        break

                sumx=sum(xv)
                sumy=sum(yv)

                log_y=[]
                log_xy=[]
                x2=[]
                
                for i in range(0,n):
                    temp = (xv[i] * math.log(yv[i],10))
                    log_xy.append(temp)
                    
                    t = (xv[i] * xv[i])
                    x2.append(t)

                    te = math.log(yv[i],10)
                    log_y.append(te)

                sumlog_xy = sum(log_xy)
                sumx2 = sum(x2)
                sumlog_y = sum(log_y)   
                
                loga = sumlog_y / n
                logb = sumlog_xy / sumx2

                tnd = []

                for i in xv:
                    t = loga + ( i * logb )
                    tnd.append(t)

                df = pd.DataFrame({
                'X': xv,
                'Y': yv,
                'log(Y)' : log_y,
                'log(XY)': log_xy,
                'X2':x2,
                'Trend':tnd
                })

                st.dataframe(df,width=900)
                
                st.markdown(""" 
                ### Summation    
                """)
                
                data = {
                    'X': [sumx],
                    'log(Y)': [sumlog_y],
                    'X2': [sumx2],
                    'X log(Y)': [sumlog_xy],
                    'log a': [loga],
                    'log b': [logb]
                }
                df = pd.DataFrame(data)

                st.dataframe(df)            
                
                st.write('')
                st.write('')

                st.write("The equation is : y = ", round(loga,2), '+ ', round(logb,2),'x')
                
                plt.plot(yv, label = 'Data Points')
                plt.plot(tnd, label = 'Trend Line')
                plt.title("Trend Chart")
                plt.xlabel("X Values")
                plt.ylabel("Y Values")
                plt.legend()

                st.pyplot()
        except Exception as e:
            st.warning("Provide a valid dataset")

    
    def seas_index_sim_avg():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()
                m=len(columns)
                year = df[columns[0]].values.tolist()

                qvals=[]
                for i in range(1,m):       
                    qtr = df[columns[i]].values.tolist()
                    qtr_floats = [round(float(val), 3) for val in qtr]  # Convert each element to float
                    qvals.append(qtr_floats)
                    
                n=len(qvals)
            
                sums=[]
                for sublist in qvals:
                    a=sum(sublist)
                    m=len(sublist)
                    sums.append(a)

                if(st.checkbox("Show Table")):
                
                    df=pd.DataFrame(qvals)  

                    with st.container():
                        st.write(df.reset_index(drop=True).T.reset_index().rename(columns={'index': ''}).astype(object).set_index(''),width=1000)

                avg = []
                for i in sums:
                    sublist_avg = round(i/m, 2)  # Calculate the average of the current sublist
                    avg.append(sublist_avg)
                    
                G = sum(avg) / n
                
                seas_ind = []
                for i in range(0, n):
                    temp = (avg[i] / G) * 100
                    seas_ind.append(round(temp, 2))

                dp=pd.DataFrame({
                    'Total':sums,
                    'Average':avg,
                    'Seasonal Indices':seas_ind
                })

                with st.container():
                    st.write(dp.reset_index(drop=True).T.reset_index().rename(columns={'index': ''}).astype(object).set_index(''),width=1000)
            
                st.write('Grand Average :',round(G,2))

                if(((sum(seas_ind))//(n*100))==1):
                    st.success("No alteration needed")

                else:
                    alt_seas_ind=[]
                    for i in seas_ind:
                        tem =( i/(sum(seas_ind)) (n*100) )
                        alt_seas_ind.append(tem)
                    
                    st.info('Altered Seasonal Indices')
                    st.dataframe(alt_seas_ind,width=900)

        except Exception as e:
            st.warning("Provide a valid dataset")

    def seas_ind_rat_to_trend():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()
                n=len(columns)
                cols=[]
                for i in range(1,n):
                    cols.append(columns[i])

                if len(cols) != 4:
                    st.warning("Please upload a valid csv file!!")

                else:
                    interval = st.select_slider("Enter the time interval",["Week",'Month','Quater','Half Yearly','Year'])
                    year = df[columns[0]].values.tolist()
                    
                    qtr1 = df[columns[1]].values.tolist()
                    qtr2 = df[columns[2]].values.tolist()
                    qtr3 = df[columns[3]].values.tolist()
                    qtr4 = df[columns[4]].values.tolist()

                    if(st.checkbox("Show Table")):
                        st.dataframe(df,width=9000)

                    x = []
                    m = n//2
                    for i in range(0,n):
                        temp = year[i] - year[m]  
                        x.append(temp)

                    yr=[]
                    for i in year:
                        yr.append(str(i))

                    y = []
                    for i in range(n):
                        s = qtr1[i] + qtr2[i] + qtr3[i] + qtr4[i]
                        y.append(s/4)

                    xy = []
                    x2 = []

                    for i in range(n):
                        xy.append( x[i] * y[i] )
                        x2.append( x[i] * x[i] )

                    df = pd.DataFrame({
                    'X': x,
                    'Y': y,
                    'X2':x2,
                    'XY':xy
                    })

                    xval,yval,xyval,x2val=st.columns(4)
                    with xval:
                        st.write('X : ',sum(x))
                    with yval:        
                        st.write('Y : ',sum(y))
                    with xyval:
                        st.write("XY : ",sum(xy))
                    with x2val:
                        st.write('X2 : ',sum(x2))
                                    
        

                    a1 = sum(x)
                    b1 = n
                    c1 = sum(y)

                    a2 = sum(x2)
                    b2 = sum(x)
                    c2 = sum(xy)
                    
                    # Create a coefficient matrix and a constant matrix
                    A = np.array([[a1, b1], [a2, b2]])
                    B = np.array([c1, c2])

                    # Solve the system of equations
                    xsol = np.linalg.solve(A, B)

                    b = xsol[0]
                    a = xsol[1]

                    tnd = []

                    for i in x:
                        t = a + ( i * b )
                        tnd.append(t)
                
                    dp = pd.DataFrame({
                        "Year" : yr,
                        "X" : x,
                        "Y" : y,
                        "XY" : xy,
                        "X2" : x2,
                        "Trend" : tnd
                    })

                    st.write
                    st.write('')

                    t1,aval,bval,t2 = st.columns(4)
                    with aval:
                        st.write("a =", round(a,2))
                    with bval:
                        st.write("b =", round(b,2))

                    st.write('')
                    st.write('')
                    st.write('')

                    a1,a2,a3 = st.columns([1,3,1])
                    
                    with a2:
                        if (b<0):
                            st.write("The equation is : y = ", a ,' - ',(-b)," x")
                
                        else:
                            st.write("The equation is : y = ", a ,' + ',b," x")

                    icr=b/4

                    q1 = []
                    q2 = []
                    q3 = []
                    q4 = []

                    x3 = round((tnd[0] - (icr/2)),2)
                    x2 = round((tnd[0] + (icr/2)),2)
                    q2.append(x3)
                    q3.append(x2)

                    x1 = round((q2[0] - icr),2)
                    x4 = round((q3[0] + icr),2)

                    q1.append(x1)
                    q4.append(x4)

                    for i in range(0,n-1):
                        te1=round((q1[i] + b),2)
                        q1.append(te1)
                        
                        te2=round((q2[i] + b),2)
                        q2.append(te2)
                        
                        te3=round((q3[i] + b),2)
                        q3.append(te3)
                        
                        te4=round((q4[i] + b),2)
                        q4.append(te4)

                    si_q1 = []
                    si_q2 = []
                    si_q3 = []
                    si_q4 = []

                    for i in range(n):
                        p1 = round(((qtr1[i] / q1[i]) *100),2)
                        si_q1.append(p1)
                        p2 = round(((qtr2[i] / q2[i]) *100),2)
                        si_q2.append(p2)
                        p3 = round(((qtr3[i] / q3[i]) *100),2)
                        si_q3.append(p3)
                        p4 = round(((qtr4[i] / q4[i]) *100),2)
                        si_q4.append(p4)

                    tot = [sum(si_q1),sum(si_q2),sum(si_q3),sum(si_q4)]
                    avg = []
                    for i in tot:
                        avg.append(i/4)

                    G = sum(avg) / n
                
                    seas_ind = []
                    for i in range(4):
                        temp = (avg[i] / G) * 100
                        seas_ind.append(round(temp, 2))

                    si_tab=[si_q1,si_q2,si_q3,si_q4]

                    dd = pd.DataFrame( si_tab )

                    st.dataframe(dd,width=900)

                    dg = pd.DataFrame({
                            'Total':tot,
                            'Average':avg,
                            'Seasonal Indices':seas_ind
                        })
                            
                    with st.container():
                        st.dataframe(dg.reset_index(drop=True).T.reset_index().rename(columns={'index': ''}).astype(object).set_index(''),width=900)

                    st.write('')
                    st.write('')
                    GA = round((sum(seas_ind ) / 4),2)
                                
                    st.metric("Grand Average :", GA)

        except Exception as e:
            st.warning("Provide a valid dataset")

    def seas_ind_rat_to_mov_avg():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()
                m1=len(columns)
                interval = st.select_slider("Enter the time interval",["Week",'Month','Quater','Half Yearly','Year'])
                year = df[columns[0]].values.tolist()
                cols=[]

                qvals=[]
                for i in range(1,m1):       
                    qtr = df[columns[i]].values.tolist()
                    qtr_floats = [round(float(val), 3) for val in qtr]  # Convert each element to float
                    qvals.append(qtr_floats)
                
                n1=len(qvals)

                for sublist in qvals:
                    n2=len(sublist)   
            
                qv = [list(x) for x in zip(*qvals)]

                q = [item for sublist in qv for item in sublist]

                arr = np.array(q)

                window_size = 4
                mv_avg = np.convolve(arr, np.ones(window_size), mode='valid').tolist()

                moving_avg1=[]

                for i in mv_avg:
                    moving_avg1.append(float(i))

                moving_avg1 = ['-']*(2) + [round(float(i),5) for i in mv_avg] + ['-']*(1)
                n=len(moving_avg1)
                moving_avg2 = ['-','-']
                
                for i in range(2, n):
                    if i >= n-2:
                        moving_avg2.append('-')
                    elif all(isinstance(x, float) for x in moving_avg1[i:i+2]):
                        s = sum(moving_avg1[i:i+2])
                        moving_avg2.append(round(s, 5))
                    else:
                        moving_avg2.append('-')

                cen_tot = []
                for i in moving_avg2:
                    if i=='-':
                        cen_tot.append('-')
                    else:
                        cen_tot.append(i/8)

                yvals=[]
                for i in range(0,len(cen_tot)):
                    if not isinstance(cen_tot[i], (int, float)):
                        yvals.append('-')
                        continue
                    y=(q[i]/cen_tot[i]*100)
                    yvals.append(round(y,3))

                year1=[]
            
                for i in range(0,len(year)):
                    year1.append(year[0])
                
                    for j in range(0,n2):
                        year1.append('-')        

                df=pd.DataFrame({
                    interval : year1,
                    'Values' : q,
                    '4 Period Moving Total': moving_avg1,
                    'Centred Total': moving_avg2,
                    'Period M' : cen_tot,
                    'Y': yvals

                })

                st.dataframe(df,width=800)     
                num_rows = n2
                num_cols = n1

                tab_val = [[] for _ in range(num_rows)]
                for i in range(num_rows):
                    for j in range(num_cols):
                        index = i * num_cols + j
                        if index < len(yvals):
                            tab_val[i].append(yvals[index])
            
                for i in range(len(tab_val)):
                    for j in range(len(tab_val[i])):
                        if tab_val[i][j] == "-":
                            tab_val[i][j] = 0

                seas_ind_table=[]
                my_list = tab_val
                num_rows = len(tab_val)
                num_cols = len(tab_val[0])

                for i in range(num_cols):
                    temp=[]
                    for j in range(num_rows):
                        t=tab_val[j][i]
                        temp.append(t)
                    seas_ind_table.append(temp)

                                
                my_list = tab_val
                num_rows = len(tab_val)
                num_cols = len(tab_val[0])

                col_sums = [0] * num_cols

                for i in range(num_rows):
                    for j in range(num_cols):
                        col_sums[j] += my_list[i][j]

                denom=[]
                
                for i in range(num_cols):
                    den=0
                    for j in range(num_rows):
                        if tab_val[j][i]==0:
                            pass
                        else:
                            den+=1
                    denom.append(den)
                
                si_avg=[]
                si_sum=[]
                d=0
                for sub in seas_ind_table:
                    s=sum(sub)
                    avg=s/denom[d]
                    si_avg.append(avg)
                    d+=1

                G=sum(si_avg)/num_cols
                si=[]
                for i in si_avg:
                    r=(i/G)*100
                    si.append(round(r,3))

                st.table(tab_val)

                dp=pd.DataFrame({
                    "Average ":si_avg,
                    "Seasonal Indices ":si
                })
                
                with st.container():
                    st.dataframe(dp.reset_index(drop=True).T.reset_index().rename(columns={'index': ''}).astype(object).set_index(''),width=900)
            
        except Exception as e:
            st.warning("Provide a valid dataset")


    def sim_mov_avg():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()
                n=len(columns)
                year = df[columns[0]].values.tolist()
                cols=[]
                for i in range(1,n):
                    cols.append(columns[i])

                col1,space,col2=st.columns([6,1,6])

                with col1:
                    cname=st.selectbox('Values',columns)
                    val=df[cname].values.tolist()
                    m=st.number_input("Enter the moving average : ",1,12,step=0)

                with col2:
                    with open("D:/statmate/Assets/50399-dashboard.json") as f:
                        animation_data = json.load(f)
                        st_lottie(animation_data)
                   
                arr = np.array(val)
                n=len(val)

                window_size = m
                mv_avg = np.convolve(arr, np.ones(window_size)/window_size, mode='valid').tolist()

                moving_avg=[]

                for i in mv_avg:
                    moving_avg.append(float(i))

                if (m%2!=0):
                    moving_avg = ['-']*((m//2)) + [round(float(i),5) for i in mv_avg] + ['-']*((m//2))

                elif(m%2==0):
                    moving_avg = ['-']*((m//2)) + [round(float(i),5) for i in mv_avg] + ['-']*((m//2)-1)


                df=pd.DataFrame({
                    'Year':year,
                    "Moving Average":moving_avg
                })
            
                st.dataframe(df,width=800)
        except Exception as e:
            st.warning("Provide a valid dataset")

    def weighted_mov_avg():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()
                n=len(columns)
                year = df[columns[0]].values.tolist()
                cols=[]
                for i in range(1,n):
                    cols.append(columns[i])

                col1,col2=st.columns(2)

                with col1:
                    cname=st.selectbox('Values',columns)
                    val=df[cname].values.tolist()
                    m=st.number_input("Enter the moving average : ",1,12,step=1)
                
                with col2:
                    with open("D:/statmate/Assets/50399-dashboard.json") as f:
                        animation_data = json.load(f)
                        st_lottie(animation_data)
                   

                n=len(val)                
                weight=1
                for i in range(1,m+1):
                    weight*=i

                w=[]
                for i in range(1,m+1):
                    w.append(i)
            
                mov_avg=[]
                for i in range(0,n-3):
                    avg=round((((val[i]*1)+(val[i+1]*2)+(val[i+2]*3))/weight),4)
                    mov_avg.append(avg)

                res_avg=[]
                for i in range(0,m):
                    res_avg.append('-')

                for j in range(m, n):
                    if j - m >= 0 and j - m < len(mov_avg):
                        res_avg.append(mov_avg[j-m])
                              
                df=pd.DataFrame({
                    "Time Interval": year,
                    "Weighted Moving Average": res_avg
                })

                st.dataframe(df,width=900)

        except Exception as e:
            st.warning("Provide a valid dataset")


    def simp_expo_smoothing():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()
                n=len(columns)
                year = df[columns[0]].values.tolist()

                cols=[]
                for i in range(1,n):
                    cols.append(columns[i])

                col1,space,col2=st.columns([5,1,5])

                cols=[]
                for i in range(1,len(columns)):
                    cols.append(columns[i])

                with col1:
                    cname=st.selectbox('Values',cols)
                    val=df[cname].values.tolist()
                    st.write('')
                    al=st.number_input("Enter Alpha value : ",0.0,5.0,step=0.1)
                    st.write('')
                    interval = 'period'
                    forecast=st.number_input(f'Enter the forecast for the first {interval} : ',0.0,step = 1.0)

                with col2:
                    with open("D:/statmate/Assets/99797-data-management.json") as f:
                        animation_data = json.load(f)
                        st.write('')
                        st.write('')
                        st_lottie(animation_data) 
                
                interval=st.select_slider("Enter the time interval",["Week",'Month','Quater','Half Yearly','Year'])
                    
                st.write('')
                st.write('')
                n=len(val)
                
                if interval=='Half Yearly':
                    interval='Year'
                expo=[forecast]
                fo=float(forecast)

                for i in range(0,n-1):
                    ft=(al*(val[i]))+((1-al)*fo)
                    expo.append(ft)
                    fo=ft

                df=pd.DataFrame({
                    interval:year,
                    "Actual Values":val,
                    "Smoothened Values":expo
                })

                st.dataframe(df,width=960)

        except Exception as e:
            st.warning("Provide a valid dataset")

    def double_expo_smoothing():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()
                n=len(columns)
                year = df[columns[0]].values.tolist()

                cols=[]
                for i in range(1,n):
                    cols.append(columns[i])
             
                col1,space,col2=st.columns([5,1,5])

                with col1:
                    st.write('')
                    st.write('')
                    st.write('')
                    cname=st.selectbox('Values',cols)
                    val=df[cname].values.tolist()
                    st.write('')
                    st.write('')
                    interval=st.select_slider("Enter the time interval",["Week",'Month','Quater','Half Yearly','Year'])

                with col2:
                    with open("D:/statmate/Assets/99797-data-management.json") as f:
                        animation_data = json.load(f)
                        st.write('')
                        st_lottie(animation_data) 
                
                col3,space,col4 = st.columns([4,1,4])

                st.write('')
                st.write('')
                with col3:
                    a = st.number_input("Enter Alpha value : ",0.0,5.0,step=0.1)
                with col4:
                    b = st.number_input("Enter Beta value : ",0.0,6.0,step=0.1)
                
                st.write('')
                st.write('')
                n = len(val)

                if interval == 'Half Yearly':
                    interval = 'Year'
                
                ctvals = ['-']
                ttvals = ['-']
                ftvals = ['-']
                
                ct = val[0]
                ctvals.append(ct)
                
                tt = (val[1]-val[0])
                ttvals.append(tt)
                
                ft = val[0]
                ftvals.append(ft)

                for i in range(2,n):
                    x = ((a*val[i])+((1-a)*(ctvals[i-1] + ttvals[i-1])))
                    ctvals.append(round(x,3))

                    y = ((b*(ctvals[i] - ctvals[i-1])) + ((1-b) * ttvals[i-1]))
                    ttvals.append(round(y,3))
                    z = ctvals[i-1] + ttvals[i-1]
                    ftvals.append(round(z,3))

            
                df = pd.DataFrame({
                        interval : year,
                        "Actual Values" : val,
                        "C(t) Values" : ctvals,
                        "T(t) Values" : ttvals,
                        "F(t) Values" : ftvals
                    })
                    
                st.dataframe(df, width=1000)
        except Exception as e:
            st.warning("Provide a valid dataset")
        
    def errors():
        try:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:

                df = pd.read_csv(uploaded_file)
                columns = df.columns.tolist()
                n=len(columns)
                year = df[columns[0]].values.tolist()
            
                cols=[]
                for i in range(1,n):
                    cols.append(columns[i])
            
                c1,c2=st.columns(2)

                with c1:
                    x_axis = st.selectbox('Actual Value ',cols)
                with c2:
                    y_axis = st.selectbox('Forecast Value ',cols)
                
                xvals=df[x_axis].values.tolist()
                yvals=df[y_axis].values.tolist()
                
                n=0
                xv=[]
                str_xv = []
                yv=[]
                str_yv = []

                for i in xvals:
                    if i==0 or i<0 or i>0:
                        xv.append(i)
                        n+=1
                    else:
                        break
                    
                for j in yvals:
                    if j==0 or j<0 or j>0:
                        yv.append(j)
                    else:
                        break

                str_year = []
                for i in range(n):
                    str_year.append(str(year[i]))
                    str_xv.append(str(xv[i]))
                    str_yv.append(str(yv[i]))

                c1 , c2, c3 = st.columns([40,1,77])

                with c1:
                    st.write('')
                    op = st.radio("Error Types",['Mean Absolute Deviation Error', 'Mean Absolute Percentage Error', 'Mean Square Error', 'Root Mean Square Error'],key="radio1")

                with c3:
                    with open("D:/statmate/Assets/98642-error-404.json") as f:
                        animation_data = json.load(f)
                        st_lottie(animation_data, height = 220)

                
                if op == 'Mean Absolute Deviation Error':            
                    mad_vals = []
                    str_mad_vals = []
                    for i in range(n):
                        if yv[i] == 0:
                            mad_vals.append(0)
                            str_mad_vals.append(str_mad_vals)
                            n-= 1
                        else:
                            temp = xv[i] - yv[i]
                            mad_vals.append(temp)

                    if st.checkbox("Show Table"):
                        df = pd.DataFrame({
                        'Year' : str_year,
                        "Acutual Values" : xv,
                        "Forecasts" : yv,
                        "Errors" : mad_vals
                        })

                        st.dataframe(df, width = 900)

                    res = sum(mad_vals) / n
                    st.metric("Mean Absolute Deviation : ", round(res,3))

                if op == 'Mean Absolute Percentage Error':            
                    mape_vals = []
                    for i in range(n):
                        if yv[i] == 0:
                            mape_vals.append(0)
                            n-=1
                        else:
                            temp = (xv[i] - yv[i]) / xv[i]
                            mape_vals.append(temp)
                   
                    if st.checkbox("Show Table"):
                        df = pd.DataFrame({
                            'Year' : str_year,
                            "Acutual Values" : xv,
                            "Forecasts" : yv,
                            "Errors" : mape_vals
                        })
                        st.dataframe(df, width = 900)

                    res = sum(mape_vals) / n
                    st.metric("Mean Absolute Percentage Error : ", round(res,3))

                if op == 'Mean Square Error':            
                    mse_vals = []
                    for i in range(n):
                        if yv[i] == 0:
                            mse_vals.append(0)
                            n-=1
                        else:
                            temp = ((xv[i] - yv[i]) **2)
                            mse_vals.append(temp)
                    
                    if st.checkbox("Show Table"):
                        df = pd.DataFrame({
                            'Year' : str_year,
                            "Acutual Values" : xv,
                            "Forecasts" : yv,
                            "Errors" : mse_vals
                        })
                        st.dataframe(df, width = 900)

                    res = sum(mse_vals) / n
                    st.metric("Mean Square Error : ", round(res,3))
                    
                if op == 'Root Mean Square Error':            
                    mse_vals = []
                    for i in range(n):
                        if yv[i] == 0:
                            mse_vals.append(0)
                            n-=1
                        else:
                            temp = ((xv[i] - yv[i]) **2)
                            mse_vals.append(temp)
                   
                    if st.checkbox("Show Table"):
                        df = pd.DataFrame({
                        'Year' : str_year,
                        "Acutual Values" : xv,
                        "Forecasts" : yv,
                        "Errors" : mse_vals
                        })
                        st.dataframe(df, width = 900)

                    st.write('')
                    res = sum(mse_vals) / n
                    st.metric("Mean Square Error : ", round(math.sqrt(res),3))
                    
                    
        except Exception as e:
            st.warning("Provide a valid dataset")
            st.write(e)


            

                


   