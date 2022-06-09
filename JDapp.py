# Import Libraries
import nltk
import pandas
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import WordNetLemmatizer

from sklearn.metrics.pairwise import cosine_similarity

from rake_nltk import Rake

import streamlit as st
# from streamlit_folium import folium_static

df = pandas.read_csv('dataset_csv.csv', encoding='windows-1252')
stop_words = set(stopwords.words('english'))

def pre_process(text):

    # lowercase
    text=text.lower()

    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    ##Convert to list from string
    text = text.split()

    # remove stopwords
    text = [word for word in text if word not in stop_words]

    # lemmatize
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word) for word in text]

    return ' '.join(text)

Resp_preprocessed = df['Responsibilities'].apply(lambda x:pre_process(x))

def get_keywords_rake(idx, docs, n=10):
    # Uses stopwords for english from NLTK, and all puntuation characters by default
    r = Rake()

    # Extraction given the text.
    r.extract_keywords_from_text(docs[idx])

    # To get keyword phrases ranked highest to lowest.
    keywords = r.get_ranked_phrases()[0:n]

    return keywords

def print_results(idx,keywords, df):
    # now print the results
    for k in keywords:
        print(k)

skills_df = pd.DataFrame(df['Job Titles'])
df_new = pd.DataFrame()

for x in range(len(skills_df)):
    idx = x
    keywords_req = get_keywords_rake(idx, df['Requirements'], n=10)
    keywords_resp = get_keywords_rake(idx, df['Responsibilities'], n=10)
    # updated = pd.Series([skills_df.iloc[x], keywords])
    df_new = df_new.append({'Job Title': skills_df.iat[x,0], 'Responsibilities' : keywords_resp ,
                            'Requirements': keywords_req}, ignore_index=True)

pd.options.display.max_colwidth = 1500
#Add sidebar to the app
st.sidebar.markdown("### View Job Descriptions")
job_role = st.sidebar.selectbox('Select Role', (df['Job Titles']))
if st.sidebar.button('View'):
    resp_string = df['Responsibilities'].loc[df['Job Titles'] == job_role].to_string(index=False)
    new_set = resp_string.replace(r'\n', '')
    # new_set = [x.replace(r"\n", '') for x in resp_string]
    new_set_string = ''.join([str(item) for item in new_set])
    sentenceSplit = filter(None, new_set_string.split("."))
    for s in sentenceSplit :
        st.sidebar.caption(s.strip() + ".")
#Add title and subtitle to the main interface of the app
st.title("Job Description Engine")
st.markdown("Input the following in order to suggest suitable job ")
degree = st.text_input('Qualification')
degree = degree.lower()
skills = st.multiselect('Skills', ['Coding Ability', 'Project Management', 'Agile Testing', 'DevOps', 'Troubleshooting',
                                   'Software Development', 'Automation Testing'])

skills = ' '.join([str(item) for item in skills])
skills = skills.lower()

exp = st.slider('Experience in years', 0, 10)

# df_new['Keywords'] = df_new['Responsibilities'] + df_new['Requirements']


def create_dataframe(matrix, tokens):

    doc_names = [f'jd_{i+1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return(df)


if st.button('Confirm'):
    # if input
    # for x in range(len(text_keywords)):

    if 'business' in degree:
        degree_set = 2
    elif 'computer science' or 'information technology' or 'software engineering' or 'computer engineering' or \
            'information systems' in degree:
        degree_set = 1


    df_new['Experience'] = df['Experience_Years']
    df_new['Degree'] = df['Degree']
    df_new['Keywords'] = df['Keywords']

    def check_empty(exp_years):
        if df_new[(df_new['Experience'] == exp_years)].empty:
            exp_years = exp_years - 1
            return check_empty(exp_years)
        else:
            return exp_years

    df_filtered = df_new[(df_new['Experience'] == check_empty(exp)) & (df_new['Degree'] == degree_set)]
    df_filtered.reset_index(inplace=True)

    keywords_preprocessed = df_filtered['Keywords'].apply(lambda x: pre_process(x))

    # print(df_filtered)
    # text_keywords = df_filtered['Keywords'].map(' '.join)

    if skills == '':
        jobs_list = df_filtered['Job Title'].tolist()
        st.subheader("Job Prediction", anchor=None)
        for i in jobs_list:
            st.markdown("- " + i)

    else:
        highest_cosine = 0
        for x in range(len(df_filtered)):
            data = [keywords_preprocessed.iloc[x], skills]
            print(data)
            count_vectorizer = CountVectorizer()
            vector_matrix = count_vectorizer.fit_transform(data)
            # cosine_similarity_matrix = cosine_similarity(vector_matrix)
            locals()['cosine_similarity_matrix_' + str(x)] = cosine_similarity(vector_matrix)
            print(locals()['cosine_similarity_matrix_' + str(x)][1, 0])
            if len(df_filtered) == 1:
                highest_cosine = locals()['cosine_similarity_matrix_' + str(x)][1, 0]
                jd = df_filtered['Job Title'][x]
            else:
                if locals()['cosine_similarity_matrix_' + str(x)][1, 0] > highest_cosine:
                    highest_cosine = locals()['cosine_similarity_matrix_' + str(x)][1, 0]
                    jd = df_filtered['Job Title'][x]

        if highest_cosine == 0:
            jobs_list = df_filtered['Job Title'].tolist()
            st.subheader("Job Prediction", anchor=None)
            for i in jobs_list:
                st.markdown("- " + i)

        else:
            st.subheader("Job Prediction", anchor=None)
            st.markdown("- " + jd)
            skills = ''