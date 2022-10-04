# cardio

#Codigo funcionalidad pagina web python 


#Importar librerias a utilizar

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Definición de variable "COLUMNA"
COLUMNA=['age','sex','chest pain type','cholesterol','max heart rate','oldpeak','ST slope']
app=Flask(__name__)

#Ruta y definición de función "data_analist" para retornar al programa "data_analist" en html
@app.route('/', methods=["GET", "POST"])
def data_analist():
    #Método POST a utilizar para recolectar los datos del programa en HTML
    if request.method == "POST": 
        #Definición de la lista "data" y recolección de datos mediante el loop for y envío de la lista "data" a diferentes funciones
        data = []
        for val in COLUMNA:
            (data.append( float(request.form.get(val))))
            
        dataset_new(data)
        a = analisis(data)

        return render_template ('index.html', a = a) 
    return render_template ('index.html')

#Definición de función dataset_new la cual crea un nuevo dataseet con los datos ingresados
def dataset_new(data):
    df = pd.DataFrame([data], columns=COLUMNA)
    df.to_csv('lista.csv')
    data1=pd.read_csv('heart_statlog_cleveland_hungary_final.csv')
    data2=pd.read_csv('lista.csv')

    value=data1[COLUMNA]
    value2=data2[COLUMNA]

    datafin=[value,value2]

    join=pd.concat(datafin)
    join.to_csv('lista2.csv')
    
#-------------------------------------------------------------------------------------
#Definición de función analisis la cual 
def analisis(data):

    df= pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
    data_df= df[['age','sex','chest pain type','cholesterol','max heart rate','oldpeak','ST slope', 'target']]
    age = data_df["age"].values
    sex = data_df["sex"].values
    chest_pain_type = data_df["chest pain type"].values
    cholesterol = data_df["cholesterol"].values
    max_heart_rate = data_df["max heart rate"].values
    oldpeak = data_df["oldpeak"].values
    ST_slope = data_df["ST slope"].values
    target = data_df["target"].values

    X = np.array([age, sex, chest_pain_type, cholesterol, max_heart_rate, oldpeak, ST_slope]).T
    y = np.array(target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    from sklearn.ensemble import RandomForestRegressor

    bar = RandomForestRegressor (n_estimators = 800, max_depth = 10)
    bar.fit(X_train, y_train)
    y_pred = bar.predict([data])
    y_pred_porc = round(y_pred[0], 2) * 100
    message = ""
    if y_pred < 0.3: 
        message = f"Tu probabilidad de padecer una afeccion cardiaca por complicaciones de miocardio oscila el valor de {y_pred_porc}%. ¿Cómo podriamos prevenir una afección cardíaca?\n  Dejando de fumar y evitar bebidas alcohólicas, llevar una dieta equilibrada, rica en fruta, verduras, legumbres y cereales. Práctica habitual de deporte. Evitar a toda costa el estrés. Buen control de enfermedades como la hipertensión arterial, diabetes Mellitus."

    elif y_pred < 0.7:
        message = f"Tu probabilidad de padecer una afeccion cardiaca por complicaciones de miocardio oscila el valor de {y_pred_porc}%.\n Esta es una posibilidad media por lo cual te recomendamos:\n Dejar de fumar y evitar bebidas alcohólicas, llevar una dieta equilibrada, rica en fruta, verduras, legumbres y cereales. Práctica habitual de deporte. Evitar a toda costa el estrés. Buen control de enfermedades como la hipertensión arterial, diabetes Mellitus."

    else:
        message = f"Tu probabilidad de padecer una afeccion cardiaca por complicaciones de miocardio oscila el valor de {y_pred_porc}%.\n Ya que posee una alta probabilidad de sufrir una insuficiencia cardiaca le recomendamos buscar asistencia medica para verificar su caso."
    return message 
    
if __name__=='__main__':
    app.run(debug=True)
