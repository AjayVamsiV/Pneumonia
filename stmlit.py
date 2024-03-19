import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from pred import efficientNet,resNet


st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

header_container = st.container()

header_container.markdown("""
<style>
    .header-container {
        background: linear-gradient(to bottom right, #ff7e5f, #feb47b);
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Create two columns for header elements
with header_container:
    #header_col1, header_col2 = st.columns([1, 3])
    left, cent,last = st.columns(3)
    div_container = st.container()
    with div_container:
        st.image('images/sircrr.jpeg', width=70)
    with cent:
        st.image('images/sircrr.jpeg', width=70)
    # Place the image in the first column
    #with header_col1:
      # Adjust width as needed

    # Place the header text in the second column and center it
    #with header_col2:
    st.markdown('<h1 style="color:#FF1100;text-align:center">Sir C.R.R college of Engineering</h1>', unsafe_allow_html=True)

st.markdown('<h1 style="color:#4D4DFF;text-align:center">Pneumonia Detection From CT-Scan</h1>', unsafe_allow_html=True)
left_co, cent_co,last_co = st.columns(3)

with cent_co:
    st.image('images/lung.jpg')
# Header description
st.markdown('<h2 style="color:gray;text-align:center">This classification model classifies CTScan into Pneumonic and Normal</h2>', unsafe_allow_html=True)

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
    c2.write(label)
    c2.write(outp[0][0])
