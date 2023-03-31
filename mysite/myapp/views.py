# imports
import os
import json
from django import forms
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import joblib


# define API endpoint
@login_required
def index(request):
    context = {
        'myapp': []
    }
    """
    polls = models.Poll.objects.all()
    for poll in polls:
        item = {
            "title": poll.title,
            "id": poll.pk,
            "answers": [{
                "value": answer.value,
                "user_first_name": answer.user.first_name,
                "user_last_name": answer.user.last_name,
                "id": answer.pk,
            } for answer in poll.answers.all()]
        }
        context['polls'].append(item)
    """
    return render(request, 'myapp/index.html', context)

@login_required
@csrf_exempt
def predict_species(request):
    # Get the absolute path to the directory containing this file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define the relative path to the model's file
    model_file_path = os.path.join(BASE_DIR, '../models/iris_model.pkl')

    # Load the model
    model = joblib.load(model_file_path)

    # parse input data from request body
    sepal_length = float(request.POST['sepal_length'])
    sepal_width = float(request.POST['sepal_width'])
    petal_length = float(request.POST['petal_length'])
    petal_width = float(request.POST['petal_width'])

    # perform data preprocessing
    input_vector = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

    # generate predictions using pre-trained model
    predicted_species = model.predict(input_vector)[0]
    probability = model.predict_proba(input_vector)

    # return predictions as JSON response
    context = {
        'json_response': {'predicted_species': {
            'prediction': predicted_species,
            'probability': {
                'setosa': round(probability[0][0], 3),
                'versicolor': round(probability[0][1], 3),
                'virginica': round(probability[0][2], 3)
            }
        }}
    }
    return render(request, 'myapp/response.html', context)
