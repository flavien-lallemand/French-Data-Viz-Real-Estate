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