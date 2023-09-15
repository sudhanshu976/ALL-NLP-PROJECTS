import streamlit as st
import pickle
import string

from nltk.corpus import stopwords
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    # Lowering 
    text = text.lower()
    
    #Tokenizing
    text = nltk.word_tokenize(text)
    
    #Removing special characters
    new_text=[]
    for word in text:
        if word.isalnum():
            new_text.append(word)
    
    text = new_text[:]
    new_text.clear()
    
    #Removing stopwords and punctuation
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            new_text.append(word)
            
            
    text = new_text[:]
    new_text.clear()
    
    # Stemming
    for word in text :
        new_text.append(ps.stem(word))
        
    return " ".join(new_text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title("SMS SPAM CLASSIFIER")
sms = st.text_input("ENTER THE MESSAGE")

if st.button("PREDICT"):
    # PREPROCESS 
    transformed_sms = transform_text(sms)

    # VECTORIZE
    vector_input = tfidf.transform([transformed_sms])

    # PREDICTION
    result = model.predict(vector_input)[0]


    # DISPLAY

    if result==1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")