# pipline out nlp Preprocessing 
# Tokennize 
# Lower +Stem  (basic keyword)
# Exclude punctuatin characters
# Bag of Words

import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
# nltk stand for natural language Tool kit
#This is python library to work with human language data 

# 1: Theory + NLP concepts (stemming , tokenization, bag of words)
# import nltk
# import numpy as np
# #Punkt this is a package with a pre-trained tokenizer 
# # nltk.download("punkt");

# #Stemmer will help to achieve a base word from a words 
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()
# def tokenize(sentence):
#     return nltk.word_tokenize(sentence)

# def stem(word):
#     return stemmer.stem(word.lower())

# #passing the array with all words
# def bag_of_words(tokenized_sentence,words):
#     """
#     sentence =["hello" ,"how","are","you"]
#     words =   ["hi" ,"hello" ,"I","you","bye","thank","cool"] / check keyword present in sentence if present then 
#     in bag we like hello present in place of hello place 1 if not present then place 0
#     bag =     [ 0 ,     1    , 0 ,  1  ,  0  ,   0   ,  0   ]
#     """
#     tokenized_sentence =[stem(w) for w in tokenized_sentence]
#     # we create and a back and initialize it with zero for each word 
    
#     bag = np.zeros(len(words),dtype=np.float32 );

#     #enumerate function have taken two parameter one is index and other is value 
#     for index, word in enumerate(words):
#         if word in tokenized_sentence:
#             bag[index] =1.0
#     return bag
        
# # a="How long does shipping taken ?"

# # print("Simple Sentence  :" +a);

# # a =tokenize(a);
# # print(a);

# # a = [stem(text) for text in a];

# # print(a)


