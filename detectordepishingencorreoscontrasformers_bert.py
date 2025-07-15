# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

!pip install -U numpy datasets --no-cache-dir

#Librerias para procesamiento de datos
import pandas as pd
import numpy as np

#Librerias para limpieza de datos
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Librerias de evaluacion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

#librerias para convertir a formato de HuggingFace dataset
!pip install datasets
from datasets import Dataset

#librerias para tokenizar datos de bert
!pip install transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

#Solo ejecutar esta celda en caso de que se tengan datos con formato HTML
!pip install beautifulsoup4
from bs4 import BeautifulSoup

#Ruta del archivo del dataset
archivoCSV = '/content/drive/MyDrive/pishing/CEAS_08.csv'
dataFrame = pd.read_csv(archivoCSV)

#Comprobamos que se cargó correctamente
print("Dimenciones del dataset ", dataFrame.shape)
dataFrame.head()

#Funcion para limpar el texto
def limpiarTextoBert(texto):
  if pd.isnull(texto):
    return ""

  texto = str(texto)                              #convertir la entrada a string
  texto = re.sub(r"http\S+|www.\S", "", texto)    #Eliminar los URLs
  texto = re.sub(r"\S+@\S","", texto)             #Eliminar las direcciones de email
  texto = re.sub(r"[^a-zA-Z]"," ", texto)         #Eliminar los numeros
  texto = texto.lower()                           #Normalizar texto a minusculas

  return texto

#Aplicar limpieza de datos al dataFrame
dataFrame['text'] = dataFrame['body'].apply(limpiarTextoBert)
dataFrame = dataFrame[['text', 'label']]          #Nos quedamos solo con el texto y la etiqueta
dataFrame = dataFrame[dataFrame['text'].str.strip().astype(bool)] #Quitamos los espacios vacios

#Mostrar datos despues de limpiarlos
print("Dimenciones del dataset ", dataFrame.shape)
dataFrame.sample(5)

#Convertir el dataFrame a formato huggingFace
dataset = Dataset.from_pandas(dataFrame)
dataset = dataset.class_encode_column("label")

print(dataset)

#-------------TOKENIZACION---------------
#Cargar el tokenizer de BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Función para tokeniar el texto
def tokenizarLote(batch):
  return tokenizer(batch['text'],
                   padding="max_length",  #Rellenar hasta una longitud fija
                   truncation=True,       #Cortar si el texto supera la longitud
                   max_length=512         #Limite estandar de BERT
                   )

#Aplicar la tokenizacion sobre los datos del batch
datasetTokenizado = dataset.map(tokenizarLote, batched=True)

#Elimianr las columnas innesesarias
datasetTokenizado = datasetTokenizado.remove_columns(['text'])

datasetTokenizado = datasetTokenizado.rename_column("label", "labels")  #Renombrar la columnas por la esperada por el trainer

datasetTokenizado.set_format("torch")                                   #Establecer el formato de pytorch

#Dividir el dataset -
datasetSplit = datasetTokenizado.train_test_split(test_size=0.2, seed=42)   #Separar el dataset formato 80-20

#Separar cada particion
datasetTrain = datasetSplit["train"]  #Parte de entrenamiento
datasetTest = datasetSplit["test"]    #Parte de prueba

print("Tamaño del entrenamiento: ", len(datasetTrain))
print("Tamaño de la prueba: ", len(datasetTest))

#Definimos el modelo
modelo = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

#Definir los hyperparametros

argumentosEntrenamiento = TrainingArguments(
    output_dir = "/content/drive/MyDrive/pishing/resultados",              #Definir directorio de salida
    do_train = True,                          #Entrenar el modelo
    do_eval = True,                           #Evaluar el modelo
    eval_strategy = "epoch",                  #Evaluar el modelo cada epoca
    save_strategy = "epoch",                  #Guardar el modelo cada epoca
    logging_strategy = "epoch",               #Cargar el modelo cada epoca
    learning_rate = 2e-5,                     #Taza de ajuste de aprendizaje
    per_device_train_batch_size = 16,         #Batch de entrenamiento (tamaño)
    per_device_eval_batch_size = 16,          #Batch de evaluacion
    num_train_epochs = 3,                     #EPOCAS a entrenar
    weight_decay = 0.01,                      #Caida de los pesos
    load_best_model_at_end = True,            #Cargar el mejor modelo al terminar
    metric_for_best_model = "accuracy",       #metricas a usar
    report_to = "none",                        #Para no usar weights ni biases
    fp16=True                                 #Usar half precision
)

#funcion para evalucar

def funcionMetricas(eval_pred):
  logits, labels = eval_pred
  preds = np.argmax(logits, axis=-1)
  return {
      "accuracy": accuracy_score(labels, preds),
      "precision": precision_score(labels, preds),
      "recall": recall_score(labels, preds),
      "f1": f1_score(labels, preds)
  }

!pip install -U accelerate

!accelerate config

import torch
print("Dispositivo:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#Configurar trainer y entrenar

trainer = Trainer(
    model = modelo,                   #pasamos el modelo contruido
    args = argumentosEntrenamiento,   #pasamos los argumentos de entrenamiento
    train_dataset = datasetTrain,     #pasamos el dataset de pruebas
    eval_dataset = datasetTest,       #pasamos el dataset de prueba
    compute_metrics = funcionMetricas #aplicamos la funcion de metricas
)

trainer.train()           #Entrenar

#Obtener predicciones
predicciones = trainer.predict(datasetTest)

#probabilidades
logits = predicciones.predictions
y_predict = np.argmax(logits, axis=-1)

#Etiquetas reales
y_real = predicciones.label_ids

#Reporte de clasificacion
print(classification_report(y_real, y_predict, target_names=["Legitimo", "Pishing"]))

#Generar matriz de confusion
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_real, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimo", "Pishing"])

plt.figure(figsize=(6,4))
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusion")
plt.show()

#--------APLICAR EL MODELO----------------------------
#Hacemos una prueba de inferencia sobre un nuevo correo

def PredecirCorreo(texto):
    # Limpieza ligera

    texto = re.sub(r"http\S+|www.\S+", "", texto)
    texto = re.sub(r"\S+@\S+", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    # Tokenizar
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Asegurar que esté en el mismo dispositivo que el modelo
    device = next(modelo.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Obtener predicción
    with torch.no_grad():
        salida = modelo(**inputs)
        logits = salida.logits
        pred = torch.argmax(logits, dim=1).item()

    return ["Legítimo", "Phishing"][pred]

#Ejemplos de implementación
correo_1 = "Dear user, we detected unusual activity in your account. Please verify your login at our secure link."
correo_2 = "Hola equipo, les dejo el acta de la reunión pasada y el presupuesto. Saludos."

print("Correo 1:", PredecirCorreo(correo_1))
print("Correo 2:", PredecirCorreo(correo_2))