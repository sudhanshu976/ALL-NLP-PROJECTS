import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    stems = [stemmer.stem(word) for word in filtered_text]
    preprocessed_text = ' '.join(stems)
    return  preprocessed_text



cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("LANGUAGE DETECTOR")
message= st.text_input("ENTER THE MESSAGE")
 

if st.button("PREDICT"):
    # PREPROCESS 
    transformed_text = preprocess(message)

    # VECTORIZE
    vector_input = cv.transform([message])

    # PREDICTION
    result = model.predict(vector_input)[0]


    # DISPLAY
    if result==0:
       st.header("ARABIC")
    elif result==1:
       st.header("DANISH")
    elif result==2:
       st.header("DUTCH")
    elif result==3:
       st.header("ENGLISH")
    elif result==4:
       st.header("FRENCH")
    elif result==5:
       st.header("GERMAN")
    elif result==6:
       st.header("GREEK")
    elif result==7:
       st.header("HINDI")
    elif result==8:
       st.header("ITALIAN")
    elif result==9:
       st.header("KANNADA")
    elif result==10:
       st.header("MALYALAM")
    elif result==11:
       st.header("PORTUGESE")
    elif result==12:
       st.header("RUSSIAN")
    elif result==13:
       st.header("SPANISH")
    elif result==14:
       st.header("SWEDISH")
    elif result==15:
       st.header("TAMIL")
    else:
       st.header("TURKISH")