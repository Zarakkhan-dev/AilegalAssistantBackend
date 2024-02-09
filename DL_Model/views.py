from django.shortcuts import render
from DL_Model.Pytorch_Model.chat import Legal_Model
from django.http import JsonResponse

# Create your views here.
import json
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body);
        user_input = data.get('fields')
        user_input =json.dumps(user_input)
        response = Legal_Model(user_input);

     
        return JsonResponse({'response':response},status=202);
    else:
        return JsonResponse({'prediction' :"working",})