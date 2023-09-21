import streamlit as st
from transformers import AutoTokenizer , AutoModelForSequenceClassification
import torch


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')



st.title("SENTIMENT ANALYZER")


message= st.text_input("ENTER THE MESSAGE")

if st.button("PREDICT"):
    tokens  = tokenizer.encode(message , return_tensors='pt')
    output = model(tokens)
    result = int(torch.argmax(output.logits))+1


    if result==1:
       st.header("TOO MUCH NEGATIVE STATEMENT")
       st.header("RATING : ⭐ ")
    elif result==2:
       st.header("NEGATIVE STATEMENT")
       st.header("RATING : ⭐⭐")
    elif result==3:
       st.header("NEUTRAL STATEMENT")
       st.header("RATING : ⭐⭐⭐")
    elif result==4:
       st.header("POSITIVE STATEMENT")
       st.header("RATING : ⭐⭐⭐⭐ ")
    elif result==5:
       st.header("TOO MUCH POSITIVE STATEMENT")
       st.header("RATING : ⭐⭐⭐⭐⭐ ")

