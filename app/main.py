import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pickle as pickle
import numpy as np


def get_prediction(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))  # Add "rb" to open the file in binary mode
    scaler = pickle.load(open("model/scaler.pkl", "rb"))  # Add "rb" here as well

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster predication")
    st.write("The cluster is: ")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>",unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicius'>Malicious</span>", unsafe_allow_html=True)

    # Get the probability of being benign
    probability_benign = model.predict_proba(input_array_scaled)[0][0]
    st.write("Probability of being benign:", probability_benign)

    st.write("Probability of being Malicious: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app ca assist medical progessionals in making a diagonsis, but should not be used as a substitue for doctors")


    
def get_clean_data():
    # Read the data from the CSV file for clearning purpose
    data = pd.read_csv("data/data.csv")
    #drop the Nul variable and unneccessary variable
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    # Encode the diagosis variable
    data['diagnosis'] = data['diagnosis'].map({'M' : 1, 'B' : 0})
    
    return data

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop("diagnosis", axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()

        scaled_value = (value - min_val) / (max_val - min_val)

        scaled_dict[key] = scaled_value

    return scaled_dict



def add_sidebar():
    st.sidebar.header("Cell Nucleus Measurements")
    data = get_clean_data()

    slider_labels = [
       ("Radius (Means)", "radius_mean"),
       ("Texture (Means)", "texture_mean"),
       ("Preimeter (Means)", "perimeter_mean"),
       ("Area (Means)", "area_mean"),
       ("Smoothness (Means)", "smoothness_mean"),
       ("Compactness (Means)", "compactness_mean"),
       ("Concavity (Means)", "concavity_mean"),
       ("Concave Points (Means)", "concave points_mean"),
       ("Symetry (Means)", "symmetry_mean"),
       ("Fractal Dimenion (Means)", "fractal_dimension_mean"),
       ("Radius (Se)", "radius_se"),
       ("Texture (Se)", "texture_se"),
       ("Preimeter (Se)", "perimeter_se"),
       ("Area (Se)", "area_se"),
       ("Smoothness (Se)", "smoothness_se"),
       ("Compactness (Se)", "compactness_se"),
       ("Concavity (Se)", "concavity_se"),
       ("Concave Points (Se)", "concave points_se"),
       ("Symetry (Se)", "symmetry_se"),
       ("Fractal Dimenion (Se)", "fractal_dimension_se"),
       ("Radius (Worst)", "radius_worst"),
       ("Texture (Worst)", "texture_worst"),
       ("Preimeter (Worst)", "perimeter_worst"),
       ("Area (Worst)", "area_worst"),
       ("Smoothness (Worst)", "smoothness_worst"),
       ("Compactness (Worst)", "compactness_worst"),
       ("Concavity (Worst)", "concavity_worst"),
       ("Concave Points (Worst)", "concave points_worst"),
       ("Symetry (Worst)", "symmetry_worst"),
       ("Fractal Dimenion (Worst)", "fractal_dimension_worst"),
       
  ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

def get_radar_chart(input_data):
   categories = ['Radius', 'Texture', 'Perimeter' , 'Area' , 'Smoothness' , 'Compactness', 'Concavity' , 'Concave   Point', 'Symmetry' , 'Fractal Dimension']

   fig = go.Figure()

   fig.add_trace(go.Scatterpolar(
         r=[
             input_data["radius_mean"],
             input_data["texture_mean"],
             input_data["perimeter_mean"],
             input_data["smoothness_mean"],
             input_data["compactness_mean"],
             input_data["concavity_mean"],
             input_data["concave points_mean"],
             input_data["symmetry_mean"],
             input_data["fractal_dimension_mean"],
            ],
         theta=categories,
         fill='toself',
         name='Mean Value'
   ))
   fig.add_trace(go.Scatterpolar(
         r=[
             input_data["radius_se"],
             input_data["texture_se"],
             input_data["perimeter_se"],
             input_data["area_se"],
             input_data["smoothness_se"],
             input_data["compactness_se"],
             input_data["concavity_se"],
             input_data["concave points_se"],
             input_data["symmetry_se"],
             input_data["fractal_dimension_se"],
          ],
         theta=categories,
         fill='toself',
         name='Standard Error'
   ))
   fig.add_trace(go.Scatterpolar(
         r=[
             input_data["radius_worst"],
             input_data["texture_worst"],
             input_data["perimeter_worst"],
             input_data["area_worst"],
             input_data["smoothness_worst"],
             input_data["compactness_worst"],
             input_data["concavity_worst"],
             input_data["concave points_worst"],
             input_data["symmetry_worst"],
             input_data["fractal_dimension_worst"],
         ],
         theta=categories,
         fill='toself',
         name='Worst Value'
   ))
    
   fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
      )),
   showlegend=True
   )

   return fig



def main(): 
   st.set_page_config(
      page_title= "Breast Cancer Prediction",
      page_icon=":female-doctor:",
      layout= "wide",
      initial_sidebar_state="expanded"
      )
   
   with open("assets/style.css") as f:
       st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
        
   

        
        
   

   input_data = add_sidebar()

   scaled_value = get_scaled_values(input_data)

   
   with st.container():
       st.title("Breast Cancer Predictor")
       st.write("Early detection and personalized treatment plans have significantly improved the prognosis for breast cancer patients. It's important to consult with a healthcare professional for a thorough diagnosis and appropriate treatment.")

   col1, col2 = st.columns([4,1])

   with col1:
       radar_chart = get_radar_chart(scaled_value)
       st.plotly_chart(radar_chart)
   
   with col2:
       get_prediction(input_data)
       





if __name__ == '__main__':
    main()
