import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import joblib

import time
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, classification_report, confusion_matrix, accuracy_score

st.markdown(
    """
    <style>
    /* Changer la couleur des titres */
    h1 {
        color: #C00000;
        font-weight: bold; /* Mettre les titres en gras */
    }

    /* Changer la couleur des titres */
    h2 {
        color: #0F4761;
        font-weight: bold; /* Mettre les titres en gras */
    }

    /* Changer la couleur des titres */
    h3 {
        color: #0F4761;
        font-weight: normal; /* Mettre les titres h2 sans gras */
    }
    """,
    unsafe_allow_html=True)

LFB = pd.read_csv("FINAL_LFB Mobilisation & Incident data from 2018 - 2023.zip", compression='zip')
@st.cache_data
def load_data():
    df_final = pd.read_csv("FINAL_LFB Mobilisation & Incident data from 2018 - 2023.zip", compression='zip', encoding='ISO-8859-1', sep=',', on_bad_lines='skip')
    return df_final

df_final = load_data()

st.title("Projet de prédiction du temps de réponse de la Brigade des Pompiers de Londres")
st.sidebar.title("Sommaire")
pages=["Introduction ⛑️", "Exploration des données 🔎", "DataVizualization 📊", "Modélisation par Régression 🛠️", 
       "Modélisation par Classification 🛠️","Conclusion 📌"]
page=st.sidebar.radio("Aller vers", pages)

st.sidebar.title("Cursus")

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Data Analyst</p>
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Bootcamp - Septembre 2024</p>
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Participants au projet")

# URL de l'image du logo LinkedIn
linkedin_logo_url = "https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg"

# URL vers votre profil LinkedIn Julie Adeline
linkedin_profile_julieA_url = "https://www.linkedin.com/in/julie-adeline/"  # Remplacez par votre URL LinkedIn
# Utiliser Markdown pour afficher le logo et le nom sur la même ligne
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Julie ADELINE</p>
    <a href="{linkedin_profile_julieA_url}" target="_blank" style="margin-left: 8px;">
        <img src="{linkedin_logo_url}" alt="LinkedIn" style="width:20px; height:20px; position: relative; top: -2px;">
    </a>
</div>
""", unsafe_allow_html=True)

# URL vers votre profil LinkedIn Elena Stratan
linkedin_profile_Elena_url = "https://www.linkedin.com/in/elena-stratan/"  # Remplacez par votre URL LinkedIn
# Utiliser Markdown pour afficher le logo et le nom sur la même ligne
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Elena STRATAN</p>
    <a href="{linkedin_profile_Elena_url}" target="_blank" style="margin-left: 8px;">
        <img src="{linkedin_logo_url}" alt="LinkedIn" style="width:20px; height:20px; position: relative; top: -2px;">
    </a>
</div>
""", unsafe_allow_html=True)

# URL vers votre profil LinkedIn Julie Deng
linkedin_profile_JulieD_url = "https://www.linkedin.com/in/jjdeng/"  # Remplacez par votre URL LinkedIn
# Utiliser Markdown pour afficher le logo et le nom sur la même ligne
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Julie DENG</p>
    <a href="{linkedin_profile_JulieD_url}" target="_blank" style="margin-left: 8px;">
        <img src="{linkedin_logo_url}" alt="LinkedIn" style="width:20px; height:20px; position: relative; top: -2px;">
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Données")
st.sidebar.markdown("""
        [Ville de Londres](https://data.london.gov.uk/)
    """)

if page==pages[0]:
    st.image("lfb.png")
    st.header("Introduction ⛑️")
    
    st.write("Portant sur les interventions des Pompiers de Londres entre 2018 et 2023, notre projet\
        cherche à révéler les facteurs déterminant le temps de réponse des pompiers\
        et à réaliser des prédictions à partir de ces variables explicatives.")
    st.subheader("Variable cible: le temps de réponse de la LFB")
    st.image('response_time.png')
   
if page==pages[1] :
    st.image("lfb.png")
    st.header("Exploration des données🔎")
    st.subheader("Aperçu général des données")
    st.write("Cette section présente un aperçu des différentes données utilisées pour l'analyse.")
    
# Sélection du jeu de données
    dataset_choice = st.radio("Choisissez un jeu de données à explorer", 
                              ["Incidents (2018-2024)", "Mobilisation (2015-2020)", "Mobilisation (2021-2024)", "Jeu de données final : 2018 - 2023"])
    
    if dataset_choice == "Incidents (2018-2024)":
        st.subheader("1. Source")
        st.markdown("Le jeu de données provient du site du gouvernement de Londres : [London Fire Brigade Incident Records](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)")
        
        st.subheader("2. Période")
        st.markdown("Les données sont mises à jour régulièrement  et débutent en 2018.")
        
        st.subheader("3. Exploration des données")
        st.markdown("**Langage utilisé** : Python")
        st.markdown(f"**Taille du DataFrame** : 759359 lignes x 39 colonnes")

        # Affichage des premières lignes
        st.markdown("**Les premières lignes de ce jeu de données :**")
        df_preview_incidents = pd.read_csv("df_incident_head.csv")
        st.write(df_preview_incidents.head())

        # Affichage du résumé statistique
        df_describe_incidents = pd.read_csv("describe_df_incident.csv")
        st.markdown("**Résumé statistique de tout le jeu de données :**")
        st.write(df_describe_incidents)

        df_nan_incidents = pd.read_csv("NaN_df_incident.csv")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Valeurs manquantes :**")
            st.write(df_nan_incidents)
        with col2:
            st.markdown("**Informations :**")
            st.image("Inc2018.png")

    elif dataset_choice == "Mobilisation (2015-2020)":
        st.subheader("1. Source")
        st.markdown("Le jeu de données provient du site du gouvernement de Londres : [Mobilisation Data](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)")

        st.subheader("2. Période")
        st.markdown("Les données couvrent la période de 2015 à 2020.")
        
        st.subheader("3. Exploration des données")
        st.markdown("**Langage utilisé** : Python")
        st.markdown(f"**Taille du DataFrame** : 883641 lignes x 22 colonnes")
        df_head_mob_1520 = pd.read_csv("head_df_mob_1520.csv")
        st.write(df_head_mob_1520)
        st.markdown("**Résumé statistique de tout le jeu de données :**")
        df_describe_mob_1520 = pd.read_csv("describe_df_mob_1520.csv")
        st.write(df_describe_mob_1520)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Valeurs manquantes :**")
            df_nan_mob_1520 = pd.read_csv("NaN_df_mob_1520.csv")
            st.write(df_nan_mob_1520)
        with col2:
            st.markdown("**Informations :**")
            st.image("Mob2015.png")
        
    elif dataset_choice == "Mobilisation (2021-2024)":
        st.subheader("1. Source")
        st.markdown("Le jeu de données provient du site du gouvernement de Londres : [Mobilisation Data](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)")
        st.subheader("2. Période")
        st.markdown("Les données couvrent la période de 2021 à 2024.")
        
        st.subheader("3. Exploration des données")
        st.markdown("**Langage utilisé** : Python")
        st.markdown(f"**Taille du DataFrame** : 659200 lignes x 24 colonnes")
        st.markdown("**Les premières lignes de ce jeu de données :**")
        df_head_mob_2124 = pd.read_csv("head_df_mob_2124.csv")
        st.write(df_head_mob_2124)
        st.markdown("**Résumé statistique de tout le jeu de données :**")
        df_describe_mob_2124 = pd.read_csv("describe_df_mob_2124.csv")
        st.write(df_describe_mob_2124)

        col1, col2 = st.columns(2)
        with col1: 
            st.markdown("**Valeurs manquantes :**")
            df_nan_mob_2124 = pd.read_csv("NaN_df_mob_2124.csv")
            st.write(df_nan_mob_2124)
        with col2:
            st.markdown("**Informations :**")
            st.image("Mob2021.png")
        
    elif dataset_choice == "Jeu de données final : 2018 - 2023":
        st.subheader("1. Source")
        st.markdown("Le jeu de données est une fusion des 3 jeux de données précédents regroupant les incidents et les mobilisations.")        
        st.subheader("2. Période")
        st.markdown("Les données couvrent la période de 2018 à 2023.")
        st.subheader("3. Feature Engineering")
        st.markdown("Afin de continuer notre étude sur le temps de réponse de la Brigade des Pompiers de Londres, nous avons réalisé des modifications sur nos jeux de données pour obtenir un fichier final exploitable et pertinent.")
        st.markdown("Pour cela, nous avons :")
        st.markdown("- Supprimé les colonnes non nécessaire à notre étude,")
        st.markdown("- Créé la variable cible 'ResponseTimeSeconds'")

        st.subheader("4. Fusion des données")
        st.markdown("**Langage utilisé** : Python")
        st.markdown(f"**Taille du DataFrame** : 634025 lignes x 22 colonnes")
        st.markdown("**Les premières lignes de ce jeu de données :**")
        df_head_final = pd.read_csv("head_df_final.csv")
        st.write(df_head_final)
        st.markdown("**Résumé statistique de tout le jeu de données :**")
        df_describe_final = pd.read_csv("describe_df_final.csv")
        st.write(df_describe_final)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Valeurs manquantes :**")
            df_nan_final = pd.read_csv("NaN_df_final.csv")
            st.write(df_nan_final)
        with col2:
            st.markdown("**Informations :**")
            st.image("final.png")

if page==pages[2] :
    st.image("lfb.png")
    st.header("DataVizualization 📊")
    
    categories=["Analyses Univariées", "Analyses Multivariées", "Analyses Statistiques"]
    categorie=st.selectbox("Types d'analyses", categories)

    if categories[0] in categorie :

        types=["Incidents par types", "Incidents par périodes", "Incidents par localisation"]
        type=st.multiselect("Sélection", types)

        if types[0] in type : 
            
            #Création d'un groupe avec le décompte par type d'incident par années
            incident_counts = LFB.groupby(['CalYear'])['IncidentGroup'].value_counts().unstack()
        
            #Création d'un graphique en barre empilée pour visualisier la répartition des incidents par type par années
            fig=plt.figure(figsize=(5,4))
            incident_counts.plot(kind='bar', stacked=True, ax=fig.gca())
            plt.xlabel("Années")
            plt.ylabel("Nombre d'incidents")
            plt.title("Nombre d'incidents par années par type d'incident")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=360)
            st.pyplot(fig)

            #Création d'un graphique pour visualiser la répartition des types d'incidents
            fig=plt.figure(figsize=(7,4))
            sns.countplot(x='IncidentGroup',data=LFB)
            plt.xlabel("Types d'incidents")
            plt.ylabel("Nombre d'incidents")
            plt.xticks(["Special Service","Fire","False Alarm"],["Special Service","Feu","Fausse alarme"])
            plt.title("Types d'incidents")
            st.pyplot(fig)

            #Ajout des explications des deux graphiques
            st.write("Nous constatons que la répartition des types d’incident par année est sensiblement la même avec environ 50% de déplacements pour fausses alertes, 30% de déplacements pour causes de services spéciaux et 20% de déplacements pour des incendies.")
            st.write("De plus, il ne semble pas y avoir de différences entre le nombre d'incidents par années.")
            st.write("Il semble y avoir eu un léger recul du nombre d'incidents en 2020 pendant le COVID suivie d'une légère hausse des incidents en 2021,2022 et 2023.")

            # Retirer de la colonne SpecialService les entrées qui ne sont pas des SpecialService
            Special_Service=LFB.loc[LFB['SpecialServiceType']!='Not a special service']

            #Faire le décompte de toutes les valeurs de SpecialService
            counts = Special_Service['SpecialServiceType'].value_counts().reset_index()

            #Trier cette liste dans l'ordre décroissant
            counts.columns = ['Special_Service', 'nombre']
            counts = counts.sort_values(by='nombre', ascending=False)

            #Création d'un graphique pour visualiser les types de special service triés par ordre d'importance
            fig=plt.figure(figsize=(7,5))
            sns.barplot(y='Special_Service', x='nombre', data=counts, order=counts['Special_Service'])
            plt.xlabel('Types de Special Service')
            plt.ylabel('Nombre')
            plt.title("Nombre d'incidents par type de Special Service")
            st.pyplot(fig)

            #Ajouter des explications
            st.write("Nous avons pu constater qu'environ 30% des activités des pompiers de la Brigade de Londres concernent des services spéciaux autre que la gestion des incendies.")
            st.write("On peut constater que la majorité de ces services spéciaux concernent des ouvertures de portes (hors incendies et urgences vitales) des inondations, des accidents de la route (RTC) et de l'assistance aux personnes bloqués dans des ascenseurs. ")

            #Création d'une variable pour visualiser le TOP5 des lieux avec le plus d'incidents
            Top_PropertyCategory=LFB['PropertyCategory'].value_counts().head()

            #Création d'un graphique pour visualiser le TOP5 des lieux avec le plus d'incidents
            fig=plt.figure(figsize=(8,5))
            sns.barplot(x=Top_PropertyCategory.values,y=Top_PropertyCategory.index)
            plt.ylabel("Types de lieux")
            plt.xlabel("Nombre d'incidents")
            plt.title("Top 5 des catégories de lieux avec le plus d'incidents")
            st.pyplot(fig)

            #Ajout des explications
            st.write("On constate dans cette variable une disparité importante avec une très forte majorité des incidents ayant lieu dans les habitations résidentielles.")

        if types[1] in type : 
            #Création d'un graphique en barre pour visualisier la distribution des incidents par mois
            fig=plt.figure(figsize=(10,6))
            sns.countplot(x='CalMonth',data=LFB,hue='CalMonth',legend=False,palette='rainbow')
            plt.xlabel("Mois")
            plt.ylabel("Nombre d'incidents")
            plt.title("Nombre d'incidents par mois")
            plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Aout","Septembre","Octobre","Novembre","Décembre"],
                    rotation=45, ha='right')
            st.pyplot(fig)

            #Création d'un graphique en barre pour visualisier la distribution des incidents par jours de la semaine
            fig=plt.figure(figsize=(8,4))
            sns.countplot(x='CalWeekday',data=LFB,hue='CalWeekday',legend=False,palette='rainbow')
            plt.xlabel("Mois")
            plt.ylabel("Nombre d'incidents")
            plt.title("Nombre d'incidents par jours de la semaine")
            plt.xticks([0,1,2,3,4,5,6],["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"],rotation=45,ha='right')
            st.pyplot(fig)

            #Création d'un graphique en barre pour visualisier la distribution des incidents par heures de la journée
            fig=plt.figure(figsize=(8,5))
            sns.countplot(x='HourOfCall',data=LFB,hue='HourOfCall',legend=False,palette='rainbow')
            plt.xlabel("Mois")
            plt.ylabel("Nombre d'incidents")
            plt.title("Nombre d'incidents par heures de la journée")
            plt.xticks([0,6,12,18,23],['Minuit','6h','Midi','18h','23h'])
            st.pyplot(fig)

            #Ajout des explications des graphiques temporels
            st.write("Le nombre d'incident semble également répartis sur l'ensemble des mois de l'année à part un léger pic en été.")
            st.write("Le nombre d'incident semble également répartis sur l'ensemble de la semaine.")
            st.write("On constate que une disparité de la répartition du nombre d'incidents selon l'heure de la journée.")
            st.write("Il semble y avoir plus d'incidents en journée que la nuit, avec un pic aux heures de pointes entre 17h et 20h.")
        
        if types[2] in type :
           #Création d'une variable pour visualiser le TOP10 des quartiers avec le plus d'incidents
            Top_Borough=LFB['IncGeo_BoroughName'].value_counts().head(10)

            #Création d'un graphique pour visualiser le TOP10 des lieux avec le plus d'incidents
            fig=plt.figure(figsize=(8,5))
            sns.barplot(x=Top_Borough.values,y=Top_Borough.index)
            plt.ylabel("Noms de quartier")
            plt.xlabel("Nombre d'incidents")
            plt.title("Top 10 des quartiers de Londres avec le plus d'incidents")
            st.pyplot(fig)

            #Création d'une variable avec les lignes du quartier de Westminster
            West=LFB[(LFB['IncGeo_BoroughName']=='WESTMINSTER')]

            #Création d'une variable pour visualiser le TOP5 des ward avec le plus d'incidents dans le quartier de Westminster
            West_no_other=West[West['IncGeo_WardName']!='OTHERS']
            Top_Ward=West_no_other['IncGeo_WardName'].value_counts().head(5)

            #Création d'un graphique pour visualiser le TOP5 des ward de westminster avec le plus d'incidents
            fig=plt.figure(figsize=(7,5))
            sns.barplot(x=Top_Ward.values,y=Top_Ward.index)
            plt.ylabel("Noms des ward")
            plt.xlabel("Nombre d'incidents")
            plt.title("Top 5 des wards de Westminster avec le plus d'incidents")
            st.pyplot(fig)

            #Ajout des explications 
            st.write("Une dernière observation intéressante est que les incidents sont répartis de manière équitable entre les différents quartiers, à l'exception du quartier de Westminster qui recense près du double d’incidents en plus que les autres quartiers.")
            st.write("Nous sommes allées plus loin et dans le quartier de Westminster, il y a deux wards (ou circonscriptions) qui ont une plus forte densité d'incidents : West End et St James's.")
            st.write("Cela peut s'expliquer par le fait que St James's et West End sont deux zones de Londres qui concentrent beaucoup de commerces et de lieux touristiques, politiques et culturels. En effet, on peut y trouver le Parlement britannique, Soho, Mayfair, la National Gallery et l'abbaye de Westminster.")

    if categories[1] in categorie :
        
        types=["Distribution du temps de réponse", "Temps de réponse par périodes", "Temps de réponse par lieux de déploiements"]
        type=st.multiselect("Sélection", types)

        if types[0] in type :   
            
            #Visualisation de la distribution du temps de réponse en secondes
            fig=plt.figure(figsize=(10,5))
            sns.boxplot(x=LFB['ResponseTimeSeconds'])
            plt.xticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200])
            plt.xlabel("Temps de réponse en secondes")
            plt.title("Boxplot des temps de réponse")
            st.pyplot(fig)

            #Information sur la donnée
            st.dataframe(LFB['ResponseTimeSeconds'].describe())

            #Ajout des explications
            st.write("Les valeurs au-delà de 500 semblent plutôt des valeurs extrêmes lors d'opérations plus longues. La distribution du temps de réponse pose une moyenne à 313 secondes et une médiane à 300 secondes. Soit environ 5 minutes entre le moment où les pompiers sont mobilisés et le moment où ils arrivent sur les lieux de l'incident.")

        if types[1] in type : 

            #Création d'un graphique permettant de visualiser l'évolution du temps de réponse par mois et par années
            fig=plt.figure(figsize=(10,6))
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2018],errorbar=None,color='blue',alpha=0.8,label=2018)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2019],errorbar=None,color='green',alpha=0.8,label=2019)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2020],errorbar=None,color='red',alpha=0.8,label=2020)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2021],errorbar=None,color='violet',alpha=0.8,label=2021)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2022],errorbar=None,color='yellow',alpha=0.8,label=2022)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2023],errorbar=None,color='orange',alpha=0.8,label=2023)
            plt.legend()
            plt.ylim(260,360)
            plt.xlim(1,12)
            plt.xlabel("Mois")
            plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12],
                       ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Aout","Septembre","Octobre","Novembre","Décemnbre"],rotation=45,ha='right')
            plt.title("Evolution du temps de réponse moyen par mois et par années")
            st.pyplot(fig)

            #Ajout des explications
            st.write("On constate que le temps de réponse est généralement situé entre 300 et 320 secondes (soit entre 5 minutes et 5 minutes trente).")
            st.write("Ce temps de réponse augmente lors des mois d'été, et en particulier les mois de Juillet avec des pics jusqu'à 340 secondes.")
            st.write("L'année 2020 est visiblement une anomalie avec, entre Avril et Aout ,des temps de réponses bien inférieurs à la moyenne des autres années.")
            st.write("Cela peut s'expliquer par l'impact des restrictions sanitaires du COVID. Les gens étant confinés chez eux, le nombre d'incidents a baissé sur cette période.")

            #Création d'un graphique permettant de visualiser l'évolution du temps de réponse par jours de la semaine et par années
            fig=plt.figure(figsize=(10,6))
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2018],errorbar=None,color='blue',alpha=0.8,label=2018)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2019],errorbar=None,color='green',alpha=0.8,label=2019)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2020],errorbar=None,color='red',alpha=0.8,label=2020)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2021],errorbar=None,color='violet',alpha=0.8,label=2021)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2022],errorbar=None,color='yellow',alpha=0.8,label=2022)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2023],errorbar=None,color='orange',alpha=0.8,label=2023)
            plt.legend()
            plt.ylim(290,340)
            plt.xlim(0,6)
            plt.ylabel("Temps de réponse en secondes")
            plt.xlabel("Jours de la semaine")
            plt.xticks([0,1,2,3,4,5,6],["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"],rotation=360)
            plt.title("Evolution du temps de réponse moyen par jours de la semaine")
            st.pyplot(fig)

            #Ajout des explications
            st.write("Comme sur le graphique précédent, on constate que le temps de réponse de 2020 est bien inférieur à ceux des autres années.")
            st.write("Cependant, on distingue une tendance avec un temps de réponse en hausse le vendredi et en baisse le weekend.")
            st.write("Le dimanche semble être le jour de la semaine avec le temps de réponse le plus rapide. Probablement car c'est un jour de repos où les gens restent chez eux.")

            #Création d'un graphique permettant de visualiser l'évolution du temps de réponse par heures de la journée et par années
            fig=plt.figure(figsize=(10,6))
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2018],errorbar=None,color='blue',alpha=0.8,label=2018)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2019],errorbar=None,color='green',alpha=0.8,label=2019)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2020],errorbar=None,color='red',alpha=0.8,label=2020)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2021],errorbar=None,color='violet',alpha=0.8,label=2021)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2022],errorbar=None,color='yellow',alpha=0.8,label=2022)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2023],errorbar=None,color='orange',alpha=0.8,label=2023)
            plt.legend()
            plt.ylim(260,360)
            plt.xlim(0,23)
            plt.xlabel("Heures")
            plt.xticks([0,3,6,9,12,15,18,21,23],['Minuit','3h','6h','9h','Midi','15h','18h','21h','23h'])
            plt.title("Evolution du temps de réponse moyen par heures")
            st.pyplot(fig)

            #Ajout des explications
            st.write("Le temps de réponse varie selon le moment de la journée.")
            st.write("En effet, on constate qu'il est en moyenne entre 300 et 340 secondes entre 1h et 6h du matin puis de 11h à 18h. Il baisse drastiquement autour de 9h et de 22h.")
            st.write("Sur l'année 2020, le temps de réponse semble similaire à celui des autres années à la différence de la journée entre 10h et 21h où il est bien inférieur.")

        if types[2] in type :

            #Création d'un graphique pour visualiser la relation entre la caserne de déploiement et le temps de réponse
            fig=plt.figure(figsize=(4,4))
            sns.boxplot(x='DeployedFromLocation',y='ResponseTimeSeconds',data=LFB)
            plt.ylim(0,1300)
            plt.ylabel("Temps de réponse en secondes")
            plt.title("Relation entre la caserne de déploiement et le temps de réponse")
            st.pyplot(fig)

            #Ajout des explications
            st.write("La médiane du temps de réponse ne semble pas fluctuer selon le lieu de déploiement des pompiers. Cependant, les valeurs sont plus dispersées lorsque les pompiers sont déployés depuis une caserne qui n'est pas la leur.")

            #Création d'un graphique pour visualiser la relation entre le quartier de l'incident et le temps de réponse
            fig=plt.figure(figsize=(20,4))
            sns.boxplot(x='IncGeo_BoroughName',y='ResponseTimeSeconds',data=LFB)
            plt.ylim(0,1300)
            plt.xticks(rotation=45,ha='right')
            plt.ylabel("Temps de réponse en secondes")
            plt.xlabel("Quarties de Londres")
            plt.title("Relation entre les types d'incident et le temps de réponse")
            st.pyplot(fig)

            #Ajout des explications
            st.write("On constate que la médiane du temps de réponse semble être assez similaire dans la plupart des quartiers à l'exception de Bromley,Enfield,Tower Hamlets, Kensington and Chealsea.")
            st.write("Cependant, on constate qu'il existe des disparités au niveau des quartiles plus ou moins importants.")
    
    if categories[2] in categorie :
              
        # Liste des variables catégorielles
        var_cat = ['CalMonth', 'CalWeekday', 'HourOfCall', 'DeployedFromLocation', 'IncGeo_BoroughName']

        # Supression des NaNs
        LFB_anova = LFB.dropna(subset=['ResponseTimeSeconds'] + var_cat)

        # Formule du modèle
        formule  = 'ResponseTimeSeconds ~ C(CalMonth) + C(CalWeekday) + C(HourOfCall) + C(DeployedFromLocation) + C(IncGeo_BoroughName)'
        model = ols(formule, data=LFB_anova).fit()

        # ANOVA
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Sauvegarde du tableau ANOVA dans un fichier
        joblib.dump(anova_table, 'anova_file')

        # Formatage des valeurs de la table en notation scientifique
        anova_table['sum_sq'] = anova_table['sum_sq'].apply(lambda x: f"{x:.3e}")
        anova_table['PR(>F)'] = anova_table['PR(>F)'].apply(lambda x: f"{x:.3e}")

        #Affichage du tableau ANOVA
        st.table(anova_table)

        #Ajout des explications
        st.write("On remarque que pour l'ensemble des variables catégorielles, la p-value est inférieure à 5%.")
        st.write("Nous pouvons en déduire que chaque variable a un effet significatif sur le temps de réponse.")

if page==pages[3] :
    st.image("lfb.png")    
    st.header("Modélisation par Régression 🛠️")

    st.subheader("Objectif")
    st.write("Prédire le temps de réponse de la Brigade des Pompiers de Londres")
 
    st.subheader("1. Étapes de Preprocessing & Modélisation")
    st.markdown(""" 
                1. Séparation en jeux de d'entrainement (75%) et de test (25%)
                2. Gestion des valeurs nulles
                3. Standardisation des données numériques
                4. Encodage des valeurs catégorielles avec OneHotEncoder
                5. Transformation des variables circulaires (CalMonth, CalHour, CalWeekday)
                6. Instanciation & entrainement des modèles
                7. Prédictions de chaque modèle sur le jeu de test 
                8. Calcul des métriques de performance
                """)
    

    # Fonction de préparation des données avec mise en cache
    @st.cache_data
    def prepare_data(LFB):
        #Séparation des variable explicatives et de la variable cible
        X=LFB.drop('ResponseTimeSeconds',axis=1)
        y=LFB['ResponseTimeSeconds']
            
        #Création des ensembles d'entrainement et de test
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

        #Remplacement des valeurs manquantes de DeployedFromLocation & DelayCodeId par le mode
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X_train[['DeployedFromLocation','DelayCode_Description']] = imputer.fit_transform(X_train[['DeployedFromLocation','DelayCode_Description']])
        X_test[['DeployedFromLocation','DelayCode_Description']] = imputer.transform(X_test[['DeployedFromLocation','DelayCode_Description']])

        #Remplacement des valeurs manquantes de SpecialServiceType par la valeur 'Not a Special Service'
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value='Not a Special Service')
        X_train['SpecialServiceType'] = imputer.fit_transform(X_train[['SpecialServiceType']]).ravel()
        X_test['SpecialServiceType'] = imputer.transform(X_test[['SpecialServiceType']]).ravel()    

        #Suppression des lignes restantes avec des valeurs manquantes non remplaçables
        X_train=X_train.dropna()
        X_test=X_test.dropna() 

        #Suppression des lignes correspondantes dans y_train & y_test
        y_train = y_train.loc[X_train.index]
        y_test = y_test.loc[X_test.index] 

        #Vérification que les index sont alignés avant la transformation
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True) 

        #Stocker les variables numériques dans des dataframes nommés num_train & num_test
        num_train=X_train[['Easting_rounded','Northing_rounded',"PumpCount"]].astype(np.int8)

        num_test=X_test[['Easting_rounded','Northing_rounded','PumpCount']].astype(np.int8)

        #Stocker les variables catégorielles dans des dataframes nommés cat_train & cat_test
        cat_train=X_train[['IncidentGroup','StopCodeDescription','SpecialServiceType','PropertyCategory','PropertyType','AddressQualifier',
                               'Postcode_district','IncGeo_BoroughName','IncGeo_WardName','IncidentStationGround', 'NumStationsWithPumpsAttending',
                               'DeployedFromStation_Code','DeployedFromLocation','DelayCode_Description','CalYear']].astype(str)

        cat_test=X_test[['IncidentGroup','StopCodeDescription','SpecialServiceType','PropertyCategory','PropertyType','AddressQualifier', 
                             'Postcode_district','IncGeo_BoroughName','IncGeo_WardName','IncidentStationGround','NumStationsWithPumpsAttending',
                             'DeployedFromStation_Code','DeployedFromLocation','DelayCode_Description','CalYear']].astype(str)

        # Stocker les variables circulaires
        circ_train=X_train[['HourOfCall','CalWeekday','CalMonth']]
        circ_test=X_test[['HourOfCall','CalWeekday','CalMonth']]

        # Standardiser les variables numériques
        scaler = StandardScaler()

        # Ajuster le scaler sur les données d'entraînement et transformer les données d'entraînement et de test
        num_train_sc = scaler.fit_transform(num_train)
        num_test_sc = scaler.transform(num_test)

        # Création des DataFrames avec les noms de colonnes appropriés
        num_train_sc_df = pd.DataFrame(num_train_sc, columns=num_train.columns, index=num_train.index)
        num_test_sc_df = pd.DataFrame(num_test_sc, columns=num_test.columns, index=num_test.index) 

        # Encoder les variables catégorielles
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

        # Ajuster l'encodeur sur les données d'entraînement et transformer les données d'entraînement et de test
        cat_train_enc = ohe.fit_transform(cat_train)
        cat_test_enc = ohe.transform(cat_test)

        # Récupération des noms des variables encodées
        cat_feature_names = ohe.get_feature_names_out(input_features=cat_train.columns)

        # Création des DataFrames avec les noms de colonnes appropriés
        cat_train_enc_df = pd.DataFrame(cat_train_enc, columns=cat_feature_names, index=cat_train.index)
        cat_test_enc_df = pd.DataFrame(cat_test_enc, columns=cat_feature_names, index=cat_test.index)

        # Transformation des variables circulaires
        ## Mois (CalMonth: 1 à 12)
        circ_train.loc[:,'CalMonth_sin']=np.sin(2 * np.pi * circ_train['CalMonth'] / 12)
        circ_train.loc[:,'CalMonth_cos']=np.cos(2 * np.pi * circ_train['CalMonth'] / 12)

        circ_test.loc[:,'CalMonth_sin']=np.sin(2 * np.pi * circ_test['CalMonth'] / 12)
        circ_test.loc[:,'CalMonth_cos']=np.cos(2 * np.pi * circ_test['CalMonth'] / 12)

        ## Jours de la semaine (CalWeekday: 1 à 7)
        circ_train.loc[:,'CalWeekday_sin']=np.sin(2 * np.pi * circ_train['CalWeekday'] / 7)
        circ_train.loc[:,'CalWeekday_cos']=np.cos(2 * np.pi * circ_train['CalWeekday'] / 7)

        circ_test.loc[:,'CalWeekday_sin']=np.sin(2 * np.pi * circ_test['CalWeekday'] / 7)
        circ_test.loc[:,'CalWeekday_cos']=np.cos(2 * np.pi * circ_test['CalWeekday'] / 7)

        ## Heures de la journée (HourOfCall: 0 à 23)
        circ_train.loc[:,'HourOfCall_sin']=np.sin(2 * np.pi * circ_train['HourOfCall'] / 24)
        circ_train.loc[:,'HourOfCall_cos']=np.cos(2 * np.pi * circ_train['HourOfCall'] / 24)

        circ_test.loc[:,'HourOfCall_sin']=np.sin(2 * np.pi * circ_test['HourOfCall'] / 24)
        circ_test.loc[:,'HourOfCall_cos']=np.cos(2 * np.pi * circ_test['HourOfCall'] / 24)

        # Concaténer les DataFrames pour former X_train et X_test
        X_train = pd.concat([cat_train_enc_df, num_train_sc_df, circ_train], axis=1)
        # Reconstitution des jeux de données
        X_test = pd.concat([cat_test_enc_df, num_test_sc_df, circ_test], axis=1)

        return X_train, X_test, y_train, y_test

    # Chargement des données préparées avec cache
    X_train, X_test, y_train, y_test = prepare_data(LFB)

    st.subheader("2 .Choix du modèle")

    regressions=["lineaire", "Ridge", "SVR", "Arbre", "RandomForest", "GradientBoosting"]
    regression=st.radio("⚠️ Attention : cela peut prendre quelques minutes", regressions, key="modele_regression")

    if regression=="lineaire":

        if "last_selected_model" not in st.session_state:
            st.session_state["last_selected_model"] = None
        if regression != st.session_state["last_selected_model"]:
            st.cache_resource.clear()  # Nettoie le cache lorsque le modèle change
            st.session_state["last_selected_model"] = regression

        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        model_filename = 'model_lin_reg.joblib'

        try:
            # Tenter de charger le modèle existant
            lin_reg = joblib.load(model_filename)
        except FileNotFoundError:
            # Si le modèle n'existe pas, on l'entraîne
            st.write("Entraînement du modèle de régression linéaire...")
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)
                
            # Sauvegarder le modèle entraîné
            joblib.dump(lin_reg, model_filename)
            st.write("Modèle de régression linéaire entraîné et sauvegardé.")
            
        # Prédictions sur les données de test
        y_pred_lin = lin_reg.predict(X_test)

        # Évaluation du modèle
        # Calcul des scores R² sur l'entraînement et le test
        train_score_lin = lin_reg.score(X_train, y_train)
        test_score_lin = lin_reg.score(X_test, y_test)

        # Calcul de l'écart de performance
        ecart_performance_lin = train_score_lin - test_score_lin

        mae = mean_absolute_error(y_test, y_pred_lin)
        mse = mean_squared_error(y_test, y_pred_lin)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_lin)

        #Créer un dictionnaire avec les résultats
        performances = {
            "Métrique": ["Train Score (R²)", "Test Score (R²)", "Écart de Performance", 
                            "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
            "Valeur": [f"{train_score_lin:.4f}", f"{test_score_lin:.4f}", f"{ecart_performance_lin:.4f}", 
                        f"{mae:.4f}", f"{mse:.4f}", f"{rmse:.4f}"]
        }

        # Créer un DataFrame pour structurer les données
        lin_performance = pd.DataFrame(performances)

        # Afficher le tableau dans Streamlit
        st.write("Performances du modèle de Régression Linéaire :")
        st.table(lin_performance)

    if regression=="Ridge":

        if "last_selected_model" not in st.session_state:
            st.session_state["last_selected_model"] = None
        if regression != st.session_state["last_selected_model"]:
            st.cache_resource.clear()  # Nettoie le cache lorsque le modèle change
            st.session_state["last_selected_model"] = regression

        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        # Entraînement du modèle si le fichier n'existe pas
        alpha_value = st.select_slider('Alpha', options=[0.01, 0.1, 1, 10, 100, 1000], key="alpha_slider")
        ridge_reg = Ridge(alpha=alpha_value, random_state=42)
        ridge_reg.fit(X_train, y_train)

        # Prédictions sur les données de test
        y_train_pred_ridge = ridge_reg.predict(X_train)
        y_pred_ridge = ridge_reg.predict(X_test)

        # Évaluation du modèle
        train_score_ridge = ridge_reg.score(X_train, y_train)
        test_score_ridge = ridge_reg.score(X_test, y_test)
        ecart_performance_ridge = train_score_ridge - test_score_ridge

        mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        rmse_ridge = np.sqrt(mse_ridge)

        # Créer un dictionnaire avec les résultats
        performances_ridge = {
            "Métrique": ["Train Score (R²)", "Test Score (R²)", "Écart de Performance", 
                            "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
            "Valeur": [f"{train_score_ridge:.4f}", f"{test_score_ridge:.4f}", f"{ecart_performance_ridge:.4f}", 
                        f"{mae_ridge:.4f}", f"{mse_ridge:.4f}", f"{rmse_ridge:.4f}"]
        }

        # Créer un DataFrame pour structurer les données
        ridge_performance = pd.DataFrame(performances_ridge)

        # Afficher le tableau dans Streamlit
        st.write("Performances du modèle de Régression Ridge :")
        st.table(ridge_performance)

    if regression=="SVR":

        if "last_selected_model" not in st.session_state:
            st.session_state["last_selected_model"] = None
        if regression != st.session_state["last_selected_model"]:
            st.cache_resource.clear()  # Nettoie le cache lorsque le modèle change
            st.session_state["last_selected_model"] = regression

        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        # Entraînement du modèle
        kernel_options = {"Linéaire": "linear", "Non-Linéaire (RBF)": "rbf"}
        selected_kernel = st.selectbox("Type de kernel", list(kernel_options.keys()))
        kernel_type = kernel_options[selected_kernel]  # Map selected option to kernel value
        epsilon_value=st.slider("Epsilon", 0.01, 0.05, 0.01)
        C_value=st.slider("C", 1, 10, 1)
        Max_iter_value=st.slider("Max_iter", 1, 50, 10)
        svr_ln = SVR(kernel=kernel_type, epsilon=epsilon_value, C=C_value, max_iter=Max_iter_value)
        svr_ln.fit(X_train, y_train)

        # Prédire la cible à partir du test set des variables explicatives
        y_pred_SVR = svr_ln.predict(X_test)

        # Évaluation du modèle
        # Calcul des scores R² sur l'entraînement et le test
        train_score_SVR = svr_ln.score(X_train, y_train)
        test_score_SVR = svr_ln.score(X_test, y_test)

        # Calcul de l'écart de performance
        ecart_performance_SVR = train_score_SVR - test_score_SVR

        mae_SVR = mean_absolute_error(y_test, y_pred_SVR)
        mse_SVR = mean_squared_error(y_test, y_pred_SVR)
        rmse_SVR = np.sqrt(mse_SVR)
        
        #Créer un dictionnaire avec les résultats
        performances = {
            "Métrique": ["Train Score (R²)", "Test Score (R²)", "Écart de Performance", 
                        "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
            "Valeur": [f"{train_score_SVR:.4f}", f"{test_score_SVR:.4f}", f"{ecart_performance_SVR:.4f}", 
                    f"{mae_SVR:.4f}", f"{mse_SVR:.4f}", f"{rmse_SVR:.4f}"]
        }

        # Créer un DataFrame pour structurer les données
        SVR_performance = pd.DataFrame(performances)

        # Afficher le tableau dans Streamlit
        st.write("Performances du modèle de Régression par vecteurs de support :")
        st.table(SVR_performance)

    if regression=="Arbre":

        if "last_selected_model" not in st.session_state:
            st.session_state["last_selected_model"] = None
        if regression != st.session_state["last_selected_model"]:
            st.cache_resource.clear()  # Nettoie le cache lorsque le modèle change
            st.session_state["last_selected_model"] = regression

        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        st.write("Entraînement du modèle arbre de décision...")
        min_sample_leaf_values=st.slider("Min Sample Leaf", 1000, 10000, step=1000)
        dtr = DecisionTreeRegressor(criterion='squared_error', splitter='best', min_samples_leaf=min_sample_leaf_values)
        dtr.fit(X_train, y_train)
        st.write("Modèle arbre de décision entraîné et sauvegardé.")

        # Prédire la cible à partir du test set des variables explicatives
        y_pred_dtr = dtr.predict(X_test)

        # Évaluation du modèle
        # Calcul des scores R² sur l'entraînement et le test
        train_score_dtr = dtr.score(X_train, y_train)
        test_score_dtr = dtr.score(X_test, y_test)

        # Calcul de l'écart de performance
        ecart_performance_dtr = train_score_dtr - test_score_dtr

        mae_dtr = mean_absolute_error(y_test, y_pred_dtr)
        mse_dtr = mean_squared_error(y_test, y_pred_dtr)
        rmse_dtr = np.sqrt(mse_dtr)
        
        #Créer un dictionnaire avec les résultats
        performances = {
            "Métrique": ["Train Score (R²)", "Test Score (R²)", "Écart de Performance", 
                        "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
            "Valeur": [f"{train_score_dtr:.4f}", f"{test_score_dtr:.4f}", f"{ecart_performance_dtr:.4f}", 
                    f"{mae_dtr:.4f}", f"{mse_dtr:.4f}", f"{rmse_dtr:.4f}"]
        }

        # Créer un DataFrame pour structurer les données
        dtr_performance = pd.DataFrame(performances)

        # Afficher le tableau dans Streamlit
        st.write("Performances du modèle de Régression par arbre de décision :")
        st.table(dtr_performance)

    if regression=="RandomForest":

        if "last_selected_model" not in st.session_state:
            st.session_state["last_selected_model"] = None
        if regression != st.session_state["last_selected_model"]:
            st.cache_resource.clear()  # Nettoie le cache lorsque le modèle change
            st.session_state["last_selected_model"] = regression

        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        # Entraînement du modèle si le fichier n'existe pas
        st.write("Entraînement du modèle Random Forest Regressor...")
        n_estimators_value=st.slider("N estimators", 5, 25, step=5)
        max_depth_value=st.slider("Max depth", 5, 15, step=1)
        rf = RandomForestRegressor(n_estimators=n_estimators_value,max_depth=max_depth_value, random_state=42)
        rf.fit(X_train, y_train)
        st.write("Modèle Random Forest Regressor entraîné.")

        # Prédire la cible à partir du test set des variables explicatives
        y_pred_rf = rf.predict(X_test)

        # Évaluation du modèle
        # Calcul des scores R² sur l'entraînement et le test
        train_score_rf = rf.score(X_train, y_train)
        test_score_rf = rf.score(X_test, y_test)

        # Calcul de l'écart de performance
        ecart_performance_rf = train_score_rf - test_score_rf

        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mse_rf)
        
        #Créer un dictionnaire avec les résultats
        performances = {
            "Métrique": ["Train Score (R²)", "Test Score (R²)", "Écart de Performance", 
                        "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
            "Valeur": [f"{train_score_rf:.4f}", f"{test_score_rf:.4f}", f"{ecart_performance_rf:.4f}", 
                    f"{mae_rf:.4f}", f"{mse_rf:.4f}", f"{rmse_rf:.4f}"]
        }

        # Créer un DataFrame pour structurer les données
        rf_performance = pd.DataFrame(performances)

        # Afficher le tableau dans Streamlit
        st.write("Performances du modèle de Random Forest Regressor :")
        st.table(rf_performance)

    if regression=="GradientBoosting":

        if "last_selected_model" not in st.session_state:
            st.session_state["last_selected_model"] = None
        if regression != st.session_state["last_selected_model"]:
            st.cache_resource.clear()  # Nettoie le cache lorsque le modèle change
            st.session_state["last_selected_model"] = regression

        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        # Entraînement du modèle si le fichier n'existe pas
        st.write("Entraînement du modèle régression par Gradient Boosting...")
        n_estimators_value=st.slider("N estimators", 5, 25, step=5)
        max_depth_value=st.slider("Max depth", 5, 15, step=1)
        reg = GradientBoostingRegressor(n_estimators=n_estimators_value,max_depth=max_depth_value, random_state=42)
        reg.fit(X_train, y_train)
        st.write("Modèle de régression par Gradient Boosting entraîné et sauvegardé.")

        # Prédire la cible à partir du test set des variables explicatives
        y_pred_reg = reg.predict(X_test)

        # Évaluation du modèle
        # Calcul des scores R² sur l'entraînement et le test
        train_score_reg = reg.score(X_train, y_train)
        test_score_reg= reg.score(X_test, y_test)

        # Calcul de l'écart de performance
        ecart_performance_reg = train_score_reg - test_score_reg

        mae_reg = mean_absolute_error(y_test, y_pred_rf)
        mse_reg = mean_squared_error(y_test, y_pred_rf)
        rmse_reg = np.sqrt(mse_reg)
        
        #Créer un dictionnaire avec les résultats
        performances = {
            "Métrique": ["Train Score (R²)", "Test Score (R²)", "Écart de Performance", 
                        "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
            "Valeur": [f"{train_score_reg:.4f}", f"{test_score_reg:.4f}", f"{ecart_performance_reg:.4f}", 
                    f"{mae_reg:.4f}", f"{mse_reg:.4f}", f"{rmse_reg:.4f}"]
        }

        # Créer un DataFrame pour structurer les données
        reg_performance = pd.DataFrame(performances)

        # Afficher le tableau dans Streamlit
        st.write("Performances du modèle de régression par Gradient Boosting :")
        st.table(reg_performance)
    

if page==pages[4] :
    st.image("lfb.png")
    st.header("Modélisation par Classification 🛠️")
    
    st.subheader("Objectif")
    st.write("Prédire l'intervalle de temps de réponse de la Brigade des Pompiers de Londres")

    st.subheader("1. Classification de la variable cible")
    st.write(":pushpin: _Distribution des valeurs avant la classification_")
    st.image('distri_cible.png')

    st.write(":pushpin: _Répartition des classes après la classification_")
    df1 = pd.DataFrame([
        {"Dataset": "y_train", 
        "Très Lente\n(plus de 500 sec)": 32865, 
        "Lente\n(400-500 sec)": 69645, 
        "Modérée\n(300-400 sec)": 75821, 
        "Rapide\n(200-300 sec)": 207009, 
        "Très Rapide\n(0-200 sec)": 90178},
        {"Dataset": "y_test", 
        "Très Lente\n(plus de 500 sec)": 11011, 
        "Lente\n(400-500 sec)": 23374, 
        "Modérée\n(300-400 sec)": 25249, 
        "Rapide\n(200-300 sec)": 69026, 
        "Très Rapide\n(0-200 sec)": 29847}
        ])
    st.dataframe(df1, use_container_width=True, hide_index=True)

    st.subheader("2. Calcul du poids des classes ")
    df2 = pd.DataFrame(
        {"Très Lente": [2.8939049995435595],
        "Lente": [1.365530906741331],
        "Modérée": [1.254301578718297],
        "Rapide": [0.45941190962711764],
        "Très Rapide": [1.0546543349524253]}
    )
    st.dataframe(df2, use_container_width=True, hide_index=True)

    # Créer une "select box" permettant de choisir le modèle de classification
    st.subheader("3. Entraînement des modèles")
    choix = ['Random Forest Classifier', 'Decision Tree Classifier', 'Logistic Regression']
    option = st.selectbox('_Choisissez votre modèle_', choix)
    st.write('Le modèle choisi est :', option)

    # Afficher des options à choisir pour scruter la performance
    display = st.radio("_Choisissez l'indicateur de performance_", ('Accuracy', 'Confusion Matrix'))

    if display == 'Accuracy' and option == 'Random Forest Classifier':
        st.write('54,7%')
    elif display == 'Accuracy' and option == 'Decision Tree Classifier':
        st.write('47,6%')
    elif display == 'Accuracy' and option == 'Logistic Regression':
        st.write('40,2%')
    elif display == 'Confusion Matrix' and option == 'Random Forest Classifier':
        st.image('rf_clf_matrix.png')
    elif display == 'Confusion Matrix' and option == 'Decision Tree Classifier':
        st.image('tree_clf_matrix.png')
    elif display == 'Confusion Matrix' and option == 'Logistic Regression':
        st.image('logistic_reg_matrix.png')

if page==pages[5] :
    st.image("lfb.png")
    st.header("Conclusion 📌")

    # Section 1: Bilan
    st.subheader("1. Bilan")
    st.markdown("""
    Malgré les efforts fournis pour équilibrer les classes et ajuster les hyperparamètres des modèles, les résultats obtenus restent insuffisants pour offrir des recommandations qualitatives et précises quant au temps de réponse des pompiers de Londres. 
    Nous avons constaté que les variables explicatives disponibles ne capturent pas les facteurs clés influençant la variable cible.
    """)

    # Section 2: Problèmes rencontrés
    st.subheader("2. Problèmes rencontrés")
    st.markdown("""
    **Jeux de données :**  
    - Séparation initiale des jeux de données en plusieurs fichiers couvrant diverses périodes, nécessitant de nombreuses itérations pour le nettoyage et le traitement des valeurs nulles.

    **Problèmes liés à l'IT :**  
    - La volumétrie importante du jeu de données a limité nos options d'encodage, nécessitant des regroupements dans certaines catégories.

    **Prévisionnel :**  
    - Le temps de traitement et de nettoyage a été conséquent, réduisant la phase d'optimisation et de finalisation.
    """)

    st.subheader("3. Suite du projet")
    st.markdown("""
    Pour améliorer la modélisation du temps de réponse, nous suggérons :  
    - **Ajout de nouvelles variables :** Les conditions météorologiques et la distance entre la caserne et le lieu de l'incident pourraient affiner les prédictions.
    - **Données GPS :** Intégrer des données GPS des camions pour mesurer les distances tout en respectant l'anonymisation des lieux d'incidents.
    """)

    # Section 4: Bibliographie
    st.subheader("4. Bibliographie")

    bibliographie_choice = st.radio(
        "Quelle bibliographie souhaitez-vous consulter ?",
        ["Données", "Publications, articles et études consultées"]
    )

    # Contenu affiché en fonction du choix
    if bibliographie_choice == "Données":
        st.markdown("""
        - [London Fire Brigade Incident Records - Ville de Londres](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)
        - [Mobilisation Data - Ville de Londres](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)
        """)

    elif bibliographie_choice == "Publications, articles et études consultées":
        st.markdown("""
        - [Review and comparison of prediction algorithms for the estimated time of arrival using geospatial transportation data.](https://doi.org/10.1016/j.procs.2021.11.003)
        - [GPS is (finally) coming to the London Fire Department.](https://www.cbc.ca/news/canada/london/london-fire-department-gps-computerized-dispatch-system-field-tested-2021-1.5801119)
        - [Guide relatif aux opérations des services de sécurité incendie.](https://cdn-contenu.quebec.ca/cdn-contenu/adm/min/securitepublique/publications-adm/publications-secteurs/securite-incendie/services-securite-incendie/guides-reference-ssi/guide_operations_2024_2_01.pdf)
        - [Modelling residential fire incident response times: A spatial analytic approach.](https://doi.org/10.1016/j.apgeog.2017.03.004)
        - [Scalable Real-time Prediction and Analysis of San Francisco Fire Department Response Times.](https://doi.org/10.1109/SmartWorld-UIC-ATC-SCALCOM-IOP-SCI.2019.00154)
        - [Survey of ETA prediction methods in public transport networks.](http://arxiv.org/abs/1904.05037)
        - [Impact of weather conditions on macroscopic urban travel times.](https://doi.org/10.1016/j.jtrangeo.2012.11.003)
        """)

