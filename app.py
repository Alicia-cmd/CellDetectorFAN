from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Charger le modèle TensorFlow
model = tf.keras.models.load_model('D:/A3MSI/Projet/Projet/Classif_Model')

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():    
    # Téléchargez l'image à partir de la requête POST
    fichier = request.files['image']
    nom_fichier = fichier.filename
    
    # Chemin vers le répertoire où vous souhaitez enregistrer l'image
    destination = 'D:/A3MSI/Projet/CellDetectorMIB-main/images/'
    
    # Vérifiez si le répertoire existe et créez-le s'il n'existe pas
    if not os.path.exists(destination):
        os.makedirs(destination)

    # Déplacez le fichier vers le répertoire de destination
    fichier.save(os.path.join(destination, nom_fichier))
    
    #create a dataframe to run the predictions
    test_df = pd.DataFrame({'id':os.listdir(destination)})
    test_df.head()
    # prepare test data (in same way as train data)
    datagen_test = ImageDataGenerator(rescale=1./255.)
    
    test_generator = datagen_test.flow_from_dataframe(
    dataframe=test_df,
    directory=destination,
    x_col='id', 
    y_col=None,
    target_size=(64,64),         # original image = (96, 96) 
    batch_size=1,
    shuffle=False,
    class_mode=None)
    
    print("Prediction des images en cours....")
    # Check model
    predictions = model.predict(test_generator, verbose=1)
    
    #create submission dataframe
    predictions = np.transpose(predictions)[0]
    submission_df = pd.DataFrame()
    submission_df['id'] = test_df['id'].apply(lambda x: x.split('.')[0])
    submission_df['label'] = list(map(lambda x: 0 if x < 0.5 else 1, predictions))
    submission_df.head()
    
    #convert to csv to submit to competition
    print("Generation en cours ....")
    submission_df.to_csv('D:/A3MSI/Projet/CellDetectorMIB-main/resultat.csv', index=False)
    
if __name__ == '__main__':
    app.run(host='10.0.0.5',port=5000)
