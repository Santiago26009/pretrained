# imports
import json
from django import forms
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import joblib

# load pre-trained machine learning model
#model = joblib.load('models/iris_model.pkl')

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
    model = joblib.load('/home/luis_wazoku/Documents/wazoku_dev_tests/test_python/mysite/models/iris_model.pkl')
    # parse input data from request body
    # input_data = json.loads(request.body)
    # sepal_length = input_data['sepal_length']
    # sepal_width = input_data['sepal_width']
    # petal_length = input_data['petal_length']
    # petal_width = input_data['petal_width']
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
    # return HttpResponse(predicted_species)
    # return JsonResponse(response_data)

