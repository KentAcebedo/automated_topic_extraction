#!/usr/bin/env python
# coding: utf-8
#LDA
import nltk
from sklearn.datasets import fetch_20newsgroups
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import spacy
import pandas as pd
import PyPDF2
from PyPDF2 import PdfReader
import pdfplumber
from gensim import corpora, models
from gensim.parsing.preprocessing import preprocess_string
from gensim.models.coherencemodel import CoherenceModel

#For stremlit
import pathlib
import textwrap
import streamlit as st
import docx2txt

#Gemini API
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown



 
def lda_analysis(text):
    nltk.download("stopwords")
    nltk.download('wordnet')
    stemmer = SnowballStemmer("english")

        
    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result
    

    unseen_document = text
    preprocessed_text = preprocess_string(unseen_document)
    dictionary = corpora.Dictionary([preprocessed_text])
    corpus = [dictionary.doc2bow(preprocessed_text)]

    # LDA model
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=1)
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic ))
        print("\n")

    # unseen document
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        print(f"Score: {score}\n Topic: {lda_model.print_topic(index, 5)}")

    # highest-scoring topic and its representative sentence
    topic_distribution = lda_model[bow_vector]
    highest_score_topic, highest_score = max(topic_distribution, key=lambda x: x[1])
    nlp = spacy.load("en_core_web_sm")
    top_words = lda_model.print_topic(highest_score_topic, topn=5)
    top_words_list = [word.strip().split('*')[1].strip('\"') for word in top_words.split('+')]
    lemmatized_words = [token.lemma_ for token in nlp(" ".join(top_words_list))]

    representative_sentence = f"This document is about {' '.join(lemmatized_words)}."
    print(representative_sentence)
    
    #Metrics
    log_perplexity = lda_model.log_perplexity(corpus)
    #coherence_model = CoherenceModel(model=lda_model, texts=preprocessed_text)
    #coherence_score = coherence_model.get_coherence()
    perplexity = np.exp(log_perplexity)

    #Output
    print("\nPerplexity score: ", perplexity)
    #print("Coherence score: ", coherence_score)
    return representative_sentence

    
def api(text):
    #API
     genai.configure(api_key="AIzaSyCF7KnBkBduE47AjyZY-gPXB4mx0JWHwgg")
     model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
     response = model.generate_content(["Make this sentence a little longer and suggest at least five related studies with accurate links. (Don't create any title just create a paragraph)", text])
     return response.text


def main():
    
    st.markdown(
        """
        <style>
        /* Customize file uploader button */
        .stFileUploader > div:first-child {
            display: flex;
            justify-content: center;
        }

        div.stButton  {
           
            color: #00FFFF; /* Text color */
            border-color: #00FFFF; /* Remove border */
            display: flex;
            justify-content: center;
        }

        div.FileUploader file_uploader {
            color: #00FFFF; /* Text color */
        
        }
        
        </style>
        """, unsafe_allow_html=True,
    )

    
    st.markdown("<h1 style='text-align: center; color: #00FFFF'><strong>Topic Modeling<strong></h1>", unsafe_allow_html=True)
    #st.title(":orange[Topic] Modeling") 
    st.subheader("", divider='rainbow')
    st.markdown('''<h4 style = 'text-align: center;'>
                If you wish to extract the topic of your selected document</strong>, kindly proceed with uploading the document below.
                ''', unsafe_allow_html=True)
    pdf_text = ""

    docx_file = st.file_uploader("Upload Document :open_file_folder:")
    type = "docx"

    #Button
    if st.button("Process"):
        if docx_file is not None:
            file_details = {"filename":docx_file.name,
                            "filetype":docx_file.type,
                            "filesize":docx_file.size}
            #st.write(file_details)


            if docx_file.type == "text/plain":
                #Read as string (decode bytes to string)
                raw_text = str(docx_file.read(), "utf-8")

                #LDA Algorithm
                topic_result = lda_analysis(raw_text)

                #with API
                summary = api(topic_result)
                st.write(summary)
                #st.write("Document details: \n", raw_text)
            
            
            elif docx_file.type == "application/pdf":
                pdf_text = " "
                try:
                    with pdfplumber.open(docx_file) as pdf:
                        for page in pdf.pages:
                            pdf_text += page.extract_text()
                except:
                    st.write("None")
                
                #LDA Algorithm
                topic_result = lda_analysis(pdf_text)

                #with API
                summary = api(topic_result)
                st.write(summary)
                #st.write("Document details: \n", pdf_text)
            
            else:
                raw_text  = docx2txt.process(docx_file)

                #LDA Algorithm
                topic_result = lda_analysis(raw_text)

                #with API
                summary = api(raw_text)
                st.write(summary)
                #st.write("Document details: \n", raw_text)
    
    else:
        st.subheader("", divider='rainbow')
        
        
    

    if st.button("Clear"):
            docx_file = None
            st.empty()

                          


if __name__ == "__main__":
    main()


