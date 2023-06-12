Project 4:
Predicting Strokes and Heart Disease

Collaborators:

Devorah Zweig: https://github.com/devorahzweig

Janki Shah: https://github.com/Jan0405

Ian Lloyd: https://github.com/eyejayell

Darya Naymon: https://github.com/DaryaNaymon

Crisaldry Brito: https://github.com/Crisaldry

Project Summary:

The project aims to initialize, train and optimize two machine learning models that can accurately predict the likelihood of a person contracting a stroke or heart disease. The models analyze factors such as age, BMI and gender combined with lifestyle choices such as smoking habits to determine whether or not the subject is prone to developing either of the diseases.

Project Files:

The project contains several different codes, each one contributing to the development of the final project. 

stroke_ros_RandomForestClassifier_final.ipynb:
This notebook contains the main code for the stroke dataset, which includes cleaning the data, scaling it, training the model and making predictions. 
After testing multiple models, it was determined that RandomForestClassifier yielded the most accurate results, and was the final model used to predict a stroke. 

PredictingDisease.ipynb:
This notebook contains one of the codes for analyzing the heart disease dataset. Within this code, the data was cleaned, analyzed, and trained several machine learning models. 

heart_disease_model_final.ipynb: 
This notebook contains the final code for the machine learning model that was trained for the heart disease dataset. After optimizing the model, the RandomForestClassifier yielded the highest results. 

app.py:
This python file contains the Flask file which sets up the different routes which each render a different use of the models. Each of these routes displays a different view of the interface, including the input of data from the user themselves. This input allows the models to predict the likelihood of the user contracting one of the aforementioned diseases. 

Webpage Link: https://jadecriada-test.bubbleapps.io/


Data Sources: 

In order to properly access the datasets, follow the links below to download them before running the codes and models. 

Stroke Dataset: https://www.kaggle.com/code/rishabh057/healthcare-dataset-stroke-data

Heart Disease Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease?resource=download
