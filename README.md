# Detección de Correos Phishing usando BERT

Este proyecto implementa un pipeline completo para un modelo de **detección de correos electrónicos de phishing** utilizando **BERT** (Bidirectional Encoder Representations from Transformers), una arquitectura de lenguaje preentrenada de Google, combinando técnicas de NLP y ciberseguridad para la detección automatizada de amenazas.

---

## Descripción

El objetivo es clasificar correos electrónicos en dos clases:  
- `0`: legítimo  
- `1`: phishing

Se utiliza el modelo **bert-base-uncased** de Hugging Face con una capa de clasificación entrenada sobre un conjunto de correos electrónicos etiquetados.

---

## Características del proyecto

- Preprocesamiento de texto y limpieza (opcional si se entrena desde los embeddings crudos)
- Tokenización con `BertTokenizer`
- Entrenamiento usando `transformers` y `Trainer` o modelo personalizado con `torch`
- Evaluación del modelo con métricas clásicas: **accuracy**, **precision**, **recall**, **f1-score**
- Visualización de resultados: matriz de confusión, curva de aprendizaje, etc.

---

## Arquitectura

Modelo base: `bert-base-uncased`  

## Requisitos

- Python 3.10 (Recomendado)
  o
- Google Colab  Jupyter Notebook

```bash
pip install transformers datasets torch scikit-learn pandas matplotlib seaborn
```


## Advertencia
Este proyecto tiene fines educativos y de investigación.
No debe usarse en producción sin pruebas rigurosas ni en escenarios reales sin aprobación ética/legal.


