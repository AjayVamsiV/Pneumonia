import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from pred import efficientNet,resNet
import base64

#@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


c1 = st.container()
with c1:
    set_png_as_page_bg('images/lung.webp')
    c1col1, c1col2, c1col3 = st.columns(3)
    with c1col2:
        c1c1 = st.container()
        with c1c1:
            c1c1col1, c1c1col2,c1c1col3 = st.columns(3)
            with c1c1col2:
                st.image('images/sircrr.jpeg', width=70)


st.markdown('<h1 style="color:#FF1100;text-align:center">SIR C R R COLLEGE OF ENGINEERING</h1>', unsafe_allow_html=True)

st.markdown('<h1 style="color:#4D4DFF;text-align:center">Pneumonia Detection from CT-Scan</h1>', unsafe_allow_html=True)
left_co, cent_co,last_co = st.columns(3)

with cent_co:
    st.image('images/lung.jpg')
# Header description
st.markdown('<h2 style="color:black;text-align:center">This classification model classifies CT-Scan into Pneumonic and Normal</h2>', unsafe_allow_html=True)

upload = st.file_uploader('Upload CT-Scan Image for classification', type=['png', 'jpg'])
c1, c2 = st.columns(2)
if upload is not None:
    im = Image.open(upload)
    img = np.asarray(im)
    img = cv2.resize(img, (180, 180))
    img = np.expand_dims(img, 0)
    img2 = np.asarray(im)
    img2 = cv2.resize(img2, (160, 160))
    img2 = np.expand_dims(img2, 0)
    c1.header('Input Image')
    c1.image(im)
    print(img2.shape)
    path1 = 'ResNet.weights.h5'
    path2 = 'EfficientNet.weights.h5'
    model1 = resNet(path1)
    model2 = efficientNet(path2)
    model3 = tf.keras.models.load_model('Pneumonia.h5')
    outp1 = model1.predict(img2)
    outp2 = model2.predict(img2)
    outp3 = model3.predict(img)
    outp = (outp1+outp2+outp3)/3
    print(outp1,outp2,outp3,outp)
    if outp > 0.5:
        label = "Pneumonic"
    else:
        label = "Normal"
    c2.header('Output')
    c2.subheader('Predicted class :')
    if(label == 'Pneumonic'):
        color = 'red'
    else:
        color = 'green'
    
    formatted_text = f"<span style='color: {color};'>{label}</span>"
    c2.write(formatted_text, unsafe_allow_html=True)
    if(label == 'Pneumonic'):
        text = str(int(outp[0][0]*100))+'% matched with Pneumonic scans'
        #c2.write(str(int(outp[0][0]*100))+'% matched with Pneumonic scans')
    else:
        text = str(100-int(outp[0][0]*100))+'% matched with Normal scans'
        #c2.write(str(100-int(outp[0][0]*100))+'% matched with Normal scans')
    formatted_text2 = f"<span style='color: {color};'>{text}</span>"
    c2.write(formatted_text2, unsafe_allow_html=True)
