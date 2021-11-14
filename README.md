# Data Vizualisation Projet - Analyse des valeurs foncières
##### Flavien Lallemand - M1-APP-BD 



Pour réaliser ce projet, j'ai utilisé les datasets fournis par le gouvernement, vous pourvez les retouver ici : https://www.data.gouv.fr/en/datasets/demandes-de-valeurs-foncieres/


### OBJECTIF 
L'objectif du projet est de contruire un application streamlit et de la publier sur streamlitshare. Cette application a pour objectif de visualiser différentes données concernant les veleurs foncières entre 2017 et 2020. 

Vous pouvez accéder à l'application streamlit ici :
https://share.streamlit.io/flavien-lallemand/real-estate-data-viz/main/projet.py 

### CONTRAINTES
Pour réaliser ce projet, les consignes étaient les suivantes : 
- 2 internal streamlit plots : st.line or st.bar_chart AND st.map
- 4 different external plots (histograms, Bar, Scatter or Pie charts) integrated with your application from external librairies like matplotlib, seaborn, plotly or Altair
- 2 checkbox that interacts with your dataset
- A slider that interacts with one or multiple plots
- Cache usage : At minimum a cache for data loading and pre-processing, you can use the st.cache
- A decorator that logs in a file the time execution interval in seconds (30 seconds, 2 seconds, 0.01 seconds, ...) and the timestamp of the call ()


### PRE-PROCESSING 
- Afin de travailler sur des datasets propres, plutôt légers et les plus consistants possible, j'ai concu un petit script permettant de générer un sous dataset contenant uniquement les informations intéressantes pour mon analyse, voici le script utilisé pour générer les 4 datasets présents dans le dossier Ressources.

```sh
import pandas as pd


# We keep only the columns that we will use in our data vizualisation project
df = pd.read_csv("full_2020.csv", usecols = ['id_mutation', 'type_local', 'valeur_fonciere', 'date_mutation','code_departement','longitude','latitude'])

#We drop rows where 3 or more values are NaN to have a consistant dataset ( > 20% of missing informations)
df.dropna(thresh=3)



#We computing Q1
q1=df["valeur_fonciere"].quantile(q=0.25)
#Computing Q3
q3=df["valeur_fonciere"].quantile(q=0.75)

#We compute the IQR
IQR=q3-q1

#The lower bound is set
inf = 20000
#The superior bound is calculated using Q3 and the interquartile range.
sup = q3 +1.5*IQR

#We keep the values within the lower and upper bounds
df= df[df["valeur_fonciere"]<sup]
df=df[df["valeur_fonciere"]>inf]


#We select a representative part of the dataset
df = df.sample(frac = 0.5)

#We create a new file
opts = dict(method='zip', archive_name='2020.csv')  
df.to_csv('2020.zip', index=False, compression=opts)
```

### EXECUTION

Pour éxéuter le porgramme, rien de plus simple : 
Cloner ce projet Github en utilisant la commande : 
```
git clone https://github.com/flavien-lallemand/real-estate-data-viz
```

Rendez-vous au dossier racine et éxécutez la commande suivante : 
```
streamlit run projet.py
```

### PROBLÈMES
Nous n'avons pas réussi à solutionner un problème que nous avons pourtant longuement creusé : L'affichage de la totalité du dataset globale sur la HeatMap. 
Nous nous sommes donc résolu à affiché seulement la moitité des lignes par le biais de la fonction sample afin de réduire la charge. La fonction sample prenant des lignes aléatoirement, nous nous retrouvons quand même avec un échantillon représentatif. 



