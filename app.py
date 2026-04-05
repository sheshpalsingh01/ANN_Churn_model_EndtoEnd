import streamlit as st
import pandas as pd
import pickle

from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from tensorflow.keras.models import load_model
from streamlit_tensorboard import st_tensorboard
from keras.models import load_model
# Load the Train model
model = load_model('04_model.h5', compile=False)

with open('01_encoded_gender.pkl','rb') as f:
    encoded_gender=pickle.load(f)
with open('02_encoded_geo.pkl','rb') as f:
    encoded_geo=pickle.load(f)
with open('03_scaler.pkl','rb') as f:
    scaler=pickle.load(f)
    

## Streamlit app

st.title('ANN Customer Churn Prediction')

# User Input
geography = st.selectbox('Geography',encoded_geo.categories_[0])
gender=st.selectbox('Gender',encoded_gender.classes_)
age = st.slider('Age',min_value=18,max_value=92)
balance= st.number_input('Balance')
credit_score =st.number_input('CreditScore')
estimated_salery= st.number_input('EstimatedSalary')
tenure=st.slider('Tenure',min_value=0,max_value=10)
num_of_product=st.slider('NumOfProducts',min_value=1,max_value=6)
has_cr_card=st.selectbox('HasCrCard',[0,1])
is_active_memb=st.selectbox('IsActiveMember',[0,1])
  

   

input_data =pd.DataFrame({
    'CreditScore':[credit_score], 
    'Gender':[encoded_gender.fit_transform([gender])[0]], 
    'Age':[age], 
    'Tenure':[tenure], 
    'Balance':[balance], 
    'NumOfProducts':[num_of_product],
    'HasCrCard':[has_cr_card], 
    'IsActiveMember':[is_active_memb], 
    'EstimatedSalary':[estimated_salery]
    
}
)

# Geo encoded
geo_encoded=encoded_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=encoded_geo.get_feature_names_out(['Geography']))

## Combine data
input_df=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

## Scale the input data
input_data_scaled= scaler.transform(input_df)

## Prediction
prediction= model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write("the Probability ",prediction_prob*100)
if prediction_prob>=0.5:
    st.success("🔥")
    st.write("The Customer is likely to churn")
else:
    st.warning("🚨")
    st.write("The Customer is not likely to churn")

# TensorBoard Integration
st.header("TensorBoard Visualization")
st_tensorboard(logdir="logs")
    
    
