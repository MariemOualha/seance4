import streamlit as st
import joblib      #load and save les fichiers conetenant les modeles
import pickle
import datetime    #modifier les dates  et meme compter et calculer les jours les heures..
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
st.sidebar.image("./hpp.jpeg")             #load image
tab=st.sidebar.columns(2)
today=datetime.date.today()
tab[0].date_input("today",today)
tab[1].time_input("time")
st.sidebar.file_uploader("upload file",accept_multiple_files=True, type=["png","jpg","pdf"])

#Separate categorical values and Numerical Values
Cat_Col = ['category','city','type']
Num_Col = ['room_count','bathroom_count' , 'size']
Pipeline = ColumnTransformer([
    ("num", StandardScaler(), Num_Col),
    ('cat', OrdinalEncoder(),Cat_Col)
])
df = pd.read_csv("./Immobiliers.csv")
Pipeline.fit_transform((df.drop(['price','log_price','region'],axis=1)))

st.title("house price prediction")
room_numbers=st.number_input("number of rooms",min_value=1,max_value=30)
bathroom_number=st.number_input("number of bathrooms",min_value=1,max_value=30)
category = st.selectbox("Choose the category",['Terrains et Fermes', 'Appartements', 'Locations de vacances',
       'Magasins, Commerces et Locaux industriels', 'Maisons et Villas',
       'Colocations', 'Bureaux et Plateaux'])
surface=st.number_input("surface",min_value=30)
types = st.selectbox("put the caracteristics",['À Vendre', 'À Louer'])
city = st.selectbox("the Localisation:",['Ariana', 'Béja', 'Ben arous', 'Bizerte', 'Gabès', 'Gafsa',
       'Jendouba', 'Kairouan', 'Kasserine', 'Kébili', 'La manouba',
       'Le kef', 'Mahdia', 'Médenine', 'Monastir', 'Sidi bouzid',
       'Siliana', 'Sousse', 'Tataouine', 'Tozeur', 'Zaghouan', 'Sfax',
       'Nabeul', 'Tunis'])

c=st.columns(10)
pred=c[4].button("predict")
with open ("pipe.h5","rb") as f:
    pip=pickle.load(f)

df_test = pd.DataFrame([[str(category),float(room_numbers),int(bathroom_number),float(surface),str(types),str(city)]],columns=["category","room_count","bathroom_count","size","type","city"])
df_test = Pipeline.transform(df_test)

model=joblib.load("best.h5")
prediction=model.predict(df_test)
if pred:
    c=st.columns(10)
    st.balloons()
    c[4].success("{:.2f}".format(prediction[0]))








