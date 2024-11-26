import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from joblib import dump, load

# Chargement des données
df = pd.read_csv('energydata_complete.csv')

# Ajout de styles CSS et affichage des titres
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 2.5em; /* Taille du texte (optionnel) */
        color: #4CAF50; /* Couleur verte pour le titre principal (optionnel) */
    }
    .centered-title1 {
        text-align: left;
        font-size: 1.5em; /* Taille du texte (optionnel) */
        color: #8e44ad; /* Couleur sombre pour le sous-titre (optionnel) */
    }
    </style>
    <h1 class="centered-title">Energy Data</h1>
    <h3 class="centered-title1">Data Visualisation</h3>
    """,
    unsafe_allow_html=True
)

# Conversion de la colonne 'date' en format datetime (si elle existe)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Afficher les premières lignes du DataFrame pour validation
st.dataframe(df.head())

# Ajouter un sélecteur de plage de dates
st.subheader("Select Date Range")
min_date = df['date'].min()
max_date = df['date'].max()
default_start_date = max_date - pd.Timedelta(days=90)  # Par défaut : 3 derniers mois

# Sélecteur de plage de dates
date_range = st.date_input(
    "Select date range:",
    [default_start_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Vérification des dates
if len(date_range) != 2:
    st.warning("Please select a valid start and end date!")
else:
    start_date, end_date = date_range
    # Filtrer les données en fonction de la plage de dates sélectionnée
    filtered_df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]

    # Liste des colonnes disponibles pour la sélection (incluant 'date')
    columns = [col for col in df.columns if col != 'Appliances']

    # Sélection de la colonne par l'utilisateur
    selected_column = st.selectbox("Choose a column to visualize against Appliances:", columns)

    # Sélection de la couleur de la ligne pour Appliances
    line_color = st.selectbox("Choose line color for Appliances:", ["red", "green", "blue", "orange", "purple"])

    # Vérification de la sélection
    if selected_column:
        fig, ax = plt.subplots(figsize=(10, 6))  # Créer un graphique avec des dimensions personnalisées
        if selected_column == 'date':
            # Si 'date' est sélectionné, afficher Appliances en fonction de la date
            ax.plot(filtered_df['date'], filtered_df['Appliances'], color=line_color)
            ax.set_title("Appliances vs Date")
            ax.set_xlabel("Date")
            ax.set_ylabel("Appliances")
            ax.tick_params(axis='x', rotation=45)  # Incliner les labels de l'axe x
        else:
            # Si une autre colonne est sélectionnée, afficher Appliances et la colonne sélectionnée
            ax.plot(filtered_df[selected_column], filtered_df['Appliances'], color=line_color)
            ax.set_title(f"Appliances vs {selected_column}")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Appliances")
            ax.tick_params(axis='x', rotation=45)  # Incliner les labels de l'axe x

        st.pyplot(fig)  # Afficher le graphique dans Streamlit
    else:
        st.warning("Please select a column to visualize.")

# Ajout de la section Distribution
st.markdown(
    """
    <style>
    .centered-title {
        text-align: left;
        font-size: 2.5em; /* Taille du texte (optionnel) */
        color: #8e44ad; /* Couleur sombre pour le titre principal (optionnel) */
    }
    </style>
    <h3 class="centered-title">Data Distribution</h3>
    """,
    unsafe_allow_html=True
)

# Sélectionner une couleur pour l'histogramme
hist_color = st.selectbox("Choose color for histogram:", ["blue", "green", "red", "orange", "purple"])

# Vérifier si des colonnes existent avant la sélection
columns = [col for col in df.columns if col != 'Appliances']
selected_columnhisto = st.selectbox("Choose a column to visualize distribution:", columns)

# Afficher la distribution de la colonne sélectionnée
if selected_columnhisto:
    fig, ax = plt.subplots(figsize=(10, 6))  # Créer une figure pour l'histogramme
    sns.histplot(filtered_df[selected_columnhisto], kde=True, color=hist_color, ax=ax)  # Histogramme avec courbe de densité
    ax.set_title(f"Distribution of {selected_columnhisto}")
    ax.set_xlabel(selected_columnhisto)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)  # Afficher l'histogramme dans Streamlit
else:
    st.warning("Please select a column for distribution visualization.")

# Ajout de la section Heatmap
st.markdown(
    """
    <style>
    .centered-title {
        text-align: left;
        font-size: 2.5em; /* Taille du texte (optionnel) */
        color: #8e44ad; /* Couleur sombre pour le titre principal (optionnel) */
    }
    </style>
    <h3 class="centered-title">Heatmap</h3>
    """,
    unsafe_allow_html=True
)

# Calcul de la matrice de corrélation des colonnes numériques
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_cols].corr()

# Créer et afficher la heatmap
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=True)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)  # Afficher la heatmap dans Streamlit

# Handling Outliers & Outlier treatments
def handle_outliers(df):
    for ftr in df:
        print(ftr,'\n')
        q_25= np.percentile(df[ftr], 25)
        q_75 = np.percentile(df[ftr], 75)
        iqr = q_75 - q_25
        print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q_25, q_75, iqr))
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower = q_25 - cut_off
        upper = q_75 + cut_off
        print(f"\nlower = {lower} and upper = {upper} \n ")
        # identify outliers
        outliers = [x for x in df[ftr] if x < lower or x > upper]
        print('Identified outliers: %d' % len(outliers))
        #removing outliers
        if len(outliers)!=0:

            def bin(row):
                if row[ftr]> upper:
                    return upper
                if row[ftr] < lower:
                    return lower
                else:
                    return row[ftr]

            df[ftr] =  df.apply (lambda row: bin(row), axis=1)
            print(f"{ftr} Outliers Removed")
        print("\n-------\n")
    return df

# Appliquer la fonction de traitement des outliers
df = handle_outliers(df)
df.rename(columns={'T1': 'temp_kitchen', 'RH_1':'hu_Kitchen', 'T2':'temp_living_room', 'RH_2': 'hu_living', 'T3':'temp_Laundry_room',
       'RH_3':'hu_laundry', 'T4':'temp_office_room', 'RH_4':'hu_office', 'T5':'temp_bathroom', 'RH_5':'hu_bath', 'T6':'temp_build_out'
       , 'RH_6':'hu_build_out', 'T7':'temp_ironing_room', 'RH_7':'hu_ironing_room', 'T8':'temp_teen_room',
       'RH_8':'hu_teen', 'T9':'temp_parents_room', 'RH_9':'hu_parent', 'T_out':'temp_out', 'RH_out':'out_humidity'},inplace = True)
# Définir les features (X) et la target (y)
X = df.drop(columns=['Appliances','hu_Kitchen','temp_kitchen','lights','hu_teen', 'temp_parents_room', 'hu_parent', 'temp_out', 'Press_mm_hg', 'out_humidity', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2','date'], axis=1)  # Features
y = df[['Appliances']]  # Target, utilisez un DataFrame pour une compatibilité avec MinMaxScaler

# Initialiser les scalers pour les entrées et la sortie
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Normaliser les données
X_scaled = scaler_X.fit_transform(X)  # Normaliser les entrées
y_scaled = scaler_y.fit_transform(y)  # Normaliser la sortie


et_model= load('et_model.joblib')











print(type(et_model))






rf_model=load('rf_model.joblib')



with st.form(key='my_form'):
 
 element_error=0
 temp_living_room = st.number_input("temp_living_room (°C)",value=19.230000)
 hu_living = st.number_input("hu_living (%)",value=44.400000)
 temp_Laundry_room = st.number_input("temp_Laundry_room",value=19.790000)
 hu_laundry=st.number_input("hu_laundry",value=44.863333 )
 temp_office_room=st.number_input("temp_office_room",value=18.890000)
 hu_office=st.number_input("hu_office",value=46.096667)
 temp_bathroom=st.number_input("temp_bathroom",value= 17.100000)
 hu_bath=st.number_input("hu_bath",value=55.000000)
 temp_build_out=st.number_input("temp_build_out",value= 6.190000)
 hu_build_out=st.number_input("hu_build_out",value=87.626667)
 temp_ironing_room=st.number_input("temp_ironing_room",value=17.200000)
 hu_ironing_room=st.number_input("hu_ironing_room",value=41.500000)
 temp_teen_room=st.number_input("temp_teen_room",value=18.100000 )
 element_tester = np.array([[temp_living_room, hu_living, temp_Laundry_room, hu_laundry, temp_office_room,hu_office,temp_bathroom,hu_bath,temp_build_out,hu_build_out,
                            temp_ironing_room, hu_ironing_room,temp_teen_room]])
    # Normaliser les données avec scaler_X
 element_tester_scaled = scaler_X.transform(element_tester)
 btn=st.form_submit_button("Prédire")
 if btn:
    element_tester = np.array([[temp_living_room, hu_living, temp_Laundry_room, hu_laundry, temp_office_room,hu_office,temp_bathroom,hu_bath,temp_build_out,hu_build_out,
                            temp_ironing_room, hu_ironing_room,temp_teen_room]])
    # Normaliser les données avec scaler_X
    element_tester_scaled = scaler_X.transform(element_tester)

    # Faire une prédiction avec Extra Trees
    element_tester_scaled = scaler_X.transform(element_tester)  # Normaliser les features

# Faire une prédiction initiale avec Extra Trees
    prediction_scaled_et = et_model.predict(element_tester_scaled)  # Prédiction normalisée
    prediction_et = scaler_y.inverse_transform(prediction_scaled_et.reshape(-1, 1))  # Inverser la normalisation

# Vérifier si cette prédiction dépasse le seuil d'erreur (calculé précédemment)
    element_error = np.abs(element_tester - prediction_et[0][0])
    threshold = np.percentile(element_error, 80)  # Seuil à 80e percentile
    indices_with_high_error = np.where(element_error >= threshold)[0]
    print(element_error)
    print(prediction_et)
    print(threshold)
    
    if (element_error >= threshold).all():  # Vérifie si tous les éléments >= seuil
      
      prediction_scaled_rf = rf_model.predict(element_tester_scaled)  # Prédiction corrigée
      prediction_combined = scaler_y.inverse_transform(prediction_scaled_rf.reshape(-1, 1))  # Inverser la normalisation
      print("combined1",prediction_combined)
    else:
     prediction_combined = prediction_et  # Sinon, utiliser la prédiction initiale
     print("combined2",prediction_combined)
    print(prediction_combined)
    st.success(f'valeur f: {prediction_combined[0]}')



