import tensorflow as tf
from PIL import Image
import streamlit as st
import numpy as np 

#Importing pre-trained model:

model = tf.keras.models.load_model("riceplant_model.keras")


#side bar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header(":green[Rice Plant Disease Recognition]")
    image_path = "rice-fields.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""Welcome to the Rice Plant Disease Recognition System! 🌿🔍
                Our mission is to help in identifying plant diseases efficiently. 
                Upload an image of a rice plant, and our system will analyze it to detect any signs of diseases. 
                Together, let's protect our crops and ensure a healthier harvest!""")
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                The "RiceLeafDisease" dataset: This article presents an image collection that was recently
                 collected and features eight different rice leaf diseases from other areas of Bangladesh. 
                There are eight classifications: Bacterial Leaf Blight (Xanthomonas oryzae pv. oryzae), Brown Spot (Cochliobolus miyabeanus), 
                Leaf scald (Microdochium oryzae), Narrow Brown Leaf Spot (Cercospora janseana), Rice Hispa (Dicladispa armigera),
                 Sheath Blight (Rhizoctonia solani), Leaf Blast (Pyricularia oryzae) and Healthy Rice Leaf. 
                The collection comprises 1701 original images and 5188 augmented photos of Rice Leaf Diseases.
                 Each original image was taken in adequate natural light against a suitable backdrop.
               

                """)
# (Further markdown omitted for brevity...)

elif app_mode == "Disease Recognition":
    data_cat = ['Bacterial Leaf Blight',
 'Brown Spot',
 'Healthy Rice Leaf',
 'Leaf Blast',
 'Leaf scald',
 'Narrow Brown Leaf Spot',
 'Rice Hispa',
 'Sheath Blight'
]
    #upload file
    img_height = 180
    img_width = 180
    image =st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
   
    if image is not None:
        # Convert the uploaded file into a format compatible with PIL
        image = Image.open(image)
        
        if st.button("Predict"):

         # Preprocess the image
           image = image.resize((img_height, img_width))  # Resize to the target size
           img_arr = np.array(image)  # Convert to array
           img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension

        # Predict
           predict = model.predict(img_arr)
           score = tf.nn.softmax(predict[0])
           predicted_class = data_cat[np.argmax(score)]
        
        # Display results
           st.image(image, width=200)
       
           st.write(f'Predicted Disease: **{predicted_class}**')
           st.write(f'Accuracy: **{np.max(score) * 100:0.2f}%**')
            
            