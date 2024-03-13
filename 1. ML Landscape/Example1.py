import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import csv

#Descargar y preparar los datos 

lifesat = pd.read_csv('/home/jonathanmerino/Documents/MachineLearning/lifesat.csv')
print("Visualizacion de los datos", lifesat, sep='\n')


x = lifesat[["GDP per capita (USD)"]].values #Se almacenan los datos de la columna en la variable x

y = lifesat[["Life satisfaction"]].values #Se almacenan los datos de la columna en la variable y

#Visualizacion de los datos 
lifesat.plot(kind='scatter', grid=True, x = 'GDP per capita (USD)', y ='Life satisfaction')
plt.axis([23_500,62_500,4,9])
plt.show()

#Seleccionar un modelo lineal
model = LinearRegression()

#Entrenar el modelo
model.fit(x,y)

#Hacer la prediccion para Cyprus
x_pred = [[37_655.2]]  #Cyprus GDP per capita 2020
print(model.predict(x_pred))


