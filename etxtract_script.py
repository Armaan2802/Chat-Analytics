import pandas as pd
from datetime import datetime
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import string
import gensim
from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
# nltk.download('wordnet')
from nltk.corpus import stopwords
from colorama import Fore, Back


#Data Cleaning

df = pd.read_csv('data3.csv')  #Data3 is the responses of the chatbot from MongoDB in a csv file
del df['_id']
del df['updatedAt']

df['Time'] = pd.to_datetime(
    df['createdAt'], format='%Y-%m-%dT%H:%M:%S.%fZ', utc=True)

# df['seconds1'] = (df['iso_timestamp1'] -
#                  pd.to_datetime(0, utc=True)).dt.total_seconds()

df['createdAt'] = df['Time']
del df['createdAt']

values_to_drop = ['welcome', 'trackorder', 'thankyou', 'cs', 'health_concerns', 'thank',
                  'programmes', 'healthrange', 'ingredients', 'clear', 'enquiry', 'offerings', 'information','ingredient']
df = df[~df['output'].isin(values_to_drop)]
df = df[~df['input'].isin(values_to_drop)]
# df = df[~df['input'].str.contains("hi","hello")]
df["input"] = df["input"].str.lower()
df["output"] = df["output"].str.lower()

hourly_counts = df.groupby(pd.Grouper(
    key='Time', freq='H')).size().reset_index(name='count')

hourly_counts = hourly_counts.sort_values(by='count', ascending=False)

df_data=df
doc_full = df_data['input']



#Beginning of extraction 

#Text Extraction Begins

bl = []
docs = df_data['output']
for el in docs:
    a = str(el).lower()
    bl.append(a)

docs=bl

sd = df_data[df_data['input'].str.contains(
    'ayurved', case=False)]['userID'].unique()

su = df_data['userID'].unique()

stop = list(stopwords.words('english'))
stop.extend(
    'ayurvedic utm_source utm_medium clear information new type start doctor consult https utm_campaign inhousebot whatsapp chatbot ingredients kapiva conversation nplease'.split())

# instantiate CountVectorizer()
cv = CountVectorizer(stop_words=stop)

# this steps generates word counts for the words in your docs
word_count_vector = cv.fit_transform(docs)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_,
                      index=cv.get_feature_names_out(), columns=["idf_weights"])

# sort ascending
df_fin = df_idf.sort_values(by=['idf_weights']).head(10)

al = (tfidf_transformer.idf_)
bl = (cv.get_feature_names_out())

df = pd.DataFrame(list(zip(bl, al)),
                  columns=['Name', 'Value'])

df_sort = df.sort_values(by=['Value']).head(15)




#Topic Modelling


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


doc_clean = [clean(doc).split() for doc in doc_full]
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=6, id2word=dictionary, passes=100)


#All Print Statements
print("\n\n\n")
# print(df_data, "\n")
print(Fore.WHITE, "\n", Back.GREEN, "Total number of users are:",
      Back.RESET, Fore.WHITE, len(su))
print(Fore.CYAN, su, Fore.WHITE, "\n\n")
print(Fore.GREEN, "\n\n",Back.GREEN,"Users that asked questions related to ayurveda are:",Back.RESET,
      Fore.WHITE, len(sd))
print(Fore.CYAN, sd, "\n\n", Fore.WHITE)
print(Fore.WHITE,Back.GREEN, "Hourly Counts:        ", Fore.WHITE,Back.RESET)
print(hourly_counts.head(15), "\n")
print(Back.GREEN, "Word Frequecy:        ", Back.RESET)
print(df_sort, "\n\n")
print(Back.GREEN,"Topics of the input field are:", Back.RESET,"\n")
print(ldamodel.print_topics(num_topics=6, num_words=2), "\n\n")
