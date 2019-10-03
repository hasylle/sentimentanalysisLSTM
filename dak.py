''' section for initialiing things, will clean up eventually '''

import re
import string
# import the inflect library 
import inflect 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 

stemmer = PorterStemmer() 
p = inflect.engine() 
stop_words = set(stopwords.words("english")) 
# remove stopwords function 
def remove_stopwords(text): 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    filtered_text = " ".join(filtered_text)
    return filtered_text 

def text_lowercase(text): 
    return text.lower() 

# convert number into words 
def convert_number(text): 
    # split string into list of words 
    temp_str = text.split() 
    # initialise empty list 
    new_string = [] 

    for word in temp_str: 
        # if word is a digit, convert the digit 
        # to numbers and append into the new_string list 
        if word.isdigit(): 
            temp = p.number_to_words(word) 
            new_string.append(temp) 

        # append the word as it is 
        else: 
            new_string.append(word) 

    # join the words of new_string to form a string 
    temp_str = ' '.join(new_string) 
    return temp_str

# remove punctuation 
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator)

# remove whitespace from text 
def remove_whitespace(text): 
    return  " ".join(text.split()) 

lemmatizer = WordNetLemmatizer() 
# lemmatize string 
def lemmatize_word(text): 
    word_tokens = word_tokenize(text) 
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
    #lemmas = " ".join(lemmas)
    return lemmas 
# stem words in the list of tokenised words 
def stem_words(text): 
    word_tokens = word_tokenize(text) 
    stems = [stemmer.stem(word) for word in word_tokens] 
    stems = " ".join(stems)
    return stems 

def preprocess_text(text):
    text = text_lowercase(text)
    text = convert_number(text)
    text = remove_punctuation(text)
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    #text = stem_words(text)
    text = lemmatize_word(text)
    return text

''' section over '''

def init_review():
    import pandas as pd
    read_traindf = pd.read_csv('labeledTrainData.tsv', delimiter="\t")
    traindf = read_traindf.drop(['id'], axis=1)
    read_testdf = pd.read_csv('testData.tsv', delimiter="\t")
    testdf = read_testdf.drop(['id'], axis=1)
    read_u_traindf = pd.read_csv('unlabeledTrainData.tsv', delimiter="\t")
    read_u_traindf = read_u_traindf.drop(['col3'], axis=1)
    u_traindf = read_u_traindf.drop(['id'], axis=1)
    
    print("Training set:")
    print(traindf.head())
    print("Test set:")
    print(testdf.head())
    print("Unlabeled set:")
    print(u_traindf.head())

    temp_traindf = traindf.drop(['sentiment'], axis=1)
    word2vec_traindf = [temp_traindf, testdf, u_traindf]
    word2vec_traindf = pd.concat(word2vec_traindf)

    return traindf, read_testdf, word2vec_traindf

def preprocess_reviews(input_list):
    input_list['processed_review'] = input_list.review.apply(lambda x: preprocess_text(x))  
    print("Processed number of reviews: ",len(input_list))
    input_list.head(3) 
    return input_list

def init_word2vec(traindf):
    #Word2Vec MODEL
    from gensim.models import Word2Vec
    from gensim.test.utils import get_tmpfile
    embedding_vector_size = 300
    wordvec_model = Word2Vec(
        sentences = traindf['processed_review'],
        size = embedding_vector_size,
        min_count=3, window=10, workers=4)
    get_tmpfile("IMDB.model")
    wordvec_model.save("IMDB.model")
    return wordvec_model

# Code from https://www.kaggle.com/alexcherniuk/imdb-review-word2vec-bilstm-99-acc
def vectorize_data(data, vocab: dict) -> list:
    print('Vectorize sentences...', end='\r')
    keys = list(vocab.keys())
    filter_unknown = lambda word: vocab.get(word, None) is not None
    encode = lambda review: list(map(keys.index, filter(filter_unknown, review)))
    vectorized = list(map(encode, data))
    print('Vectorize sentences... (done)')
    return vectorized

# I used this guide for saving and loading the Keras
# model state

# I will fix this someday

def restore_state():
    from gensim.models import Word2Vec
    import pandas as pd
    import h5py
    from keras.models import model_from_json
    word2vec_model = Word2Vec.load("IMDB.model")
    traindf = pd.read_csv('processed_train.csv')
    testdf = pd.read_csv('processed_test.csv')
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")


    return traindf, testdf, word2vec_model, loaded_model
    
def save_state(traindf, testdf, word2vec_model, keras_model):
    import h5py
    traindf.save_csv('processed_train.csv')
    testdf.save_csv('processed_test.csv')
    word2vec_model.save("IMDB.model")
    # serialize model to JSON
    model_json = keras_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    keras_model.save_weights("model.h5")
    print("Saved model to disk")
