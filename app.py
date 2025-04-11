import streamlit as st
import numpy as np
import pickle
lr_model=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor App")
st.write("This app is created using the data taken from kaggle website.")
st.write("The prediction here is based on the data available, hence might not align perfectly with the real-life data.")

company=st.selectbox("Laptop Manufacturer",df['Company'].unique(),index=4)
typename=st.selectbox("Type of Laptop",df['TypeName'].unique(),index=1)
cpu=st.selectbox("Processor",df['Cpu'].unique(),index=0)
cpu_speed=st.slider("CPU Clock Speed",min_value=0.8,max_value=4.0,step=0.1,value=2.5)
ram=st.radio("Amount of RAM on the system",[4,6,8,12,16,24,32,64,128],index=2,horizontal=True)
gpu=st.selectbox("Graphics Card on the system",df['Gpu'].unique(),index=1)
os=st.selectbox("Operating System",df['OpSys'].unique(),index=2)
weight=st.slider("Weight of the laptop(in kg)",min_value=0.6,max_value=5.0,step=0.2,value=2.0)
ips=st.selectbox("Does the laptop display have IPS Panel?",["Yes",'No'],index=1)
touchscreen=st.selectbox("Does the laptop have a touchscreen?",["Yes",'No'],index=1)
ssd=st.selectbox("What is the SSD storage on the system? If system has HDD storage only, select 0",
                 [0,128,256,512,1024],index=3)
hdd=st.selectbox("What is the HDD storage on the system? If system has SSD only, then select 0",
                 [0,128,256,512,1024,2048],index=0)
screen_size=st.slider("Screen size(in inches, measured diagonally)",min_value=10.0,max_value=18.5,step=0.1,value=15.6)
screen_resolution=st.selectbox("Screen Resolution(in horizontal x vertical pixels)",
                               ["2560x1600","1440x900","1920x1080","2880x1800","1366x768","2304x1440",
                                "3200x1800","1920x1200","2256x1504","3840x2160","2160x1440","2560x1440",
                                "1600x900","2736x1824","2400x1600"],index=2)
if st.button("PREDICT PRICE"):
    X_res=int(screen_resolution.split('x')[0])
    Y_res=int(screen_resolution.split('x')[1])
    ppi=(X_res**2+Y_res**2)**0.5/screen_size
    if ips=='Yes':
        ips=1
    else:
        ips=0
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    query=np.array([[company,typename,cpu,ram,gpu,os,weight,ips,touchscreen,cpu_speed,hdd,ssd,ppi]])
    op=lr_model.predict(query)
    st.subheader("The predicted price of the laptop with the above mentioned specification is ₹"+ str(int(round(op[0],-2))))
