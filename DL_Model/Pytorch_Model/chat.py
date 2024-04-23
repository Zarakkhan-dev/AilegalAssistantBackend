import random
import json
import torch
from DL_Model.Pytorch_Model.Neutral import NeuralNet
from DL_Model.Pytorch_Model.nltk_utilis import bag_of_words, tokenize,stem
from django.conf import settings
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
json_file_path_1 = os.path.join(settings.STATIC_ROOT, 'myapp', 'sentimental_analysis.json')
json_file_path_2 = os.path.join(settings.STATIC_ROOT, 'myapp', 'ppc_Acts_info.json')
with open(json_file_path_1, 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

with open(json_file_path_2, 'r', encoding='utf-8') as json_data:
    intents_info = json.load(json_data)

FILE = os.path.join(settings.PTH,'data.pth')
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

FILE_info = os.path.join(settings.PTH,'data_info.pth')
data_info = torch.load(FILE_info)

input_size_info = data_info["input_size"]
hidden_size_info = data_info["hidden_size"]
output_size_info = data_info["output_size"]
all_words_info = data_info['all_words']
tags_info = data_info['tags']
model_state_info = data_info["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) #KNow ours parameter
model.eval() #evalution paramter

model_info = NeuralNet(input_size_info, hidden_size_info, output_size_info).to(device)
model_info.load_state_dict(model_state_info) #KNow ours parameter
model_info.eval() #evalution paramter


def Legal_Model(user_input):
    # sentence = "do you use credit cards?"
    sentence = user_input;
    sentence =  tokenize(sentence)
    sections_check = sentence
    ignore_words = ['?', '.', '!']
    check_Text = [stem(w) for w in sections_check if w not in ignore_words]
    if("summar"  in check_Text ):
        response =Summarization_Model(user_input)
        return response
    else:
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0]) #reshape to get only one rows of set
        X = torch.from_numpy(X).to(device)

        X_info = bag_of_words(sentence, all_words_info)
        X_info = X_info.reshape(1, X_info.shape[0]) #reshape to get only one rows of set
        X_info = torch.from_numpy(X_info).to(device)

        output = model(X)
        output_info = model_info(X_info)
        _, predicted = torch.max(output, dim=1)
        _, predicted_info = torch.max(output_info, dim=1)

        tag = tags[predicted.item()] #Get the actual Tags 
        tag_info = tags_info[predicted_info.item()] #Get the actual Tags 
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]#Probability for this tag is high enough and to do this 
        
        probs_info = torch.softmax(output_info, dim=1)
        prob_info = probs_info[0][predicted_info.item()]
        if prob.item() > 0.5:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    sentimental_analysis = intent['responses']
                    if(sentimental_analysis == "gaining information"):
                        if prob_info.item() > 0.5:
                            for intent_info in intents_info['intents']:
                                if tag_info == intent_info["tag"]:
                                    not_found = 0
                                    for i in range(299, 325):
                                            if str(i) in sections_check:
                                                response = random.choice(intent_info['responses'])
                                                return response
                                            elif int(i) >=324:
                                                not_found =1
                                                break
                                                
                                    if not_found == 1:
                                        response=  "I am designed to facilitate the user with the information of Law under act 299 to act 324 of Pakistan "
                                        return response        
                    else:
                        response = random.choice(intent['responses'])
                        return response
        else:
            response ="I do not understand..."
            return response


def Summarization_Model(paragraph):
    stopward = list(STOP_WORDS);

    # print(stopward);
    nlp = spacy.load("en_core_web_sm")
    document= nlp(paragraph);

    tokens =[token.text for token in document]

    #how many time word is present is caled frequency

    word_frequency ={}
    for word in document:
        if word.text.lower()  not in stopward and word.text.lower() not in punctuation:
            if word.text not in word_frequency.keys():
                word_frequency[word.text] =1
            else:
                word_frequency[word.text] +=1

    #find the max frequency 
    max_frequency = max(word_frequency.values());

    #find the normalize frequency max/corresponding word frequency

    for word in word_frequency.keys():
        word_frequency[word] =word_frequency[word]/max_frequency 

    sentence_token = [sentence  for sentence in document.sents];

    # print(sentence_token)
    sentence_scores={}

    for single_sentence in sentence_token:
        for word in single_sentence:
            if word.text in word_frequency.keys():
                if single_sentence not in sentence_scores.keys():
                    sentence_scores[single_sentence] = word_frequency[word.text];
                else:
                    sentence_scores[single_sentence] +=word_frequency[word.text];
    select_length = int(len(sentence_token) *0.1)
    summary =nlargest(select_length,sentence_scores,key=sentence_scores.get);
    final_summary = [word.text for word in  summary]
    summary =' '.join(final_summary);
    return summary
