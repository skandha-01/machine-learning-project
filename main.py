import streamlit as st
import numpy as np
import pandas as pd
import datetime
import xgboost as xgb

import joblib
date_time = datetime.datetime.now()
model = xgb.XGBRegressor()
model.load_model('hello.json')

def main(): 
    html_temp="""
     <div style = "background-color:red;padding:20px">
     <h2 style="color:black;text-align:center;"> Car Price Prediction Using ML</h2>
     <h4 style="color:black;text-align:right;"> by skandha.s</h4>
     </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
   
    st.markdown("##### Are you planning to sell your car !?\n##### So let's try evaluating the price..")
    
    st.write('')
    st.write('')
    p1 = st.number_input('What is the current ex-showroom price of the car ?  (In Lakhs)',2.5,25.0,step=1.0) 
    
    
    p2 = st.number_input('What is distance completed by the car in Kilometers ?',100,50000000,step=100)

    s1 = st.selectbox('What is the fuel type of the car ?',('Petrol','Diesel', 'CNG'))
    if s1=="Petrol":
        p3=0
    elif s1=="Diesel":
        p3=1
    elif s1=="CNG":
        p3=2
        
    s2 = st.selectbox('Are you a dealer or an individual ?', ('Dealer','Individual'))
    
    if s2=="Dealer":
        p4=0
        st.info("Please note: There will be a 2% commission on the predicted price.")
    elif s2=="Individual":
        p4=1
        
    s3 = st.selectbox('What is the Transmission Type ?', ('Manual','Automatic'))
    if s3=="Manual":
        p5=0
    elif s3=="Automatic":
        p5=1
        
    p6 = st.slider("Number of Owners the car previously had",1,3)
    
    
    years = st.number_input('In which year car was purchased ?',1990,date_time.year,step=1)
    p7 = date_time.year-years
    
    data_new = pd.DataFrame({
    'Present_Price':p1,
    'Kms_Driven':p2,
    'Fuel_Type':p3,
    'Seller_Type':p4,
    'Transmission':p5,
    'Owner':p6,
    'Age':p7
},index=[0])
    try: 
        if st.button('Predict'):
            prediction = model.predict(data_new)
            if prediction>0:
                st.balloons()
                st.success('You can sell the car for {:.2f} lakhs'.format(prediction[0]))
            else:
                st.warning("You will be not able to sell this car !!")
    except:
        st.warning("Opps!! Something went wrong\nTry again")

if __name__ == '__main__':
    main()