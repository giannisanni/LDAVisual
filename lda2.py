import zipfile
import pandas as pd
import os
import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from pprint import pprint
import numpy as np
import tqdm
import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis
import spacy
def load_data(file_path):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall("temp")
    return pd.read_csv("temp/NIPS Papers/papers.csv")

def preprocess_data(data):
    data = data.drop(columns=['id', 'title', 'abstract', 'event_type', 'pdf_name', 'year'], axis=1)
    data = data.sample(100)
    data['paper_text_processed'] = data['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))
    data['paper_text_processed'] = data['paper_text_processed'].map(lambda x: x.lower())
    return data

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
# Load Spacy's 'en' model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def process_words(texts, stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
def build_lda_model(corpus, id2word, num_topics=10):
    return gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, chunksize=100, passes=10, per_word_topics=True)


def compute_coherence_values(corpus, dictionary, texts, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=k, random_state=100,
                                           chunksize=100, passes=10, alpha=a, eta=b)
    coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    return coherence_model_lda.get_coherence()

def tune_hyperparameters(corpus, id2word, texts, min_topics=2, max_topics=11, step_size=1, alpha_list=None, beta_list=None):
    if alpha_list is None:
        alpha_list = list(np.arange(0.01, 1, 0.3))
        alpha_list.append('symmetric')
        alpha_list.append('asymmetric')

    if beta_list is None:
        beta_list = list(np.arange(0.01, 1, 0.3))
        beta_list.append('symmetric')

    topics_range = range(min_topics, max_topics, step_size)
    model_results = {'Topics': [], 'Alpha': [], 'Beta': [], 'Coherence': []}

    for k in topics_range:
        for a in alpha_list:
            for b in beta_list:
                cv = compute_coherence_values(corpus=corpus, dictionary=id2word, texts=texts, k=k, a=a, b=b)
                model_results['Topics'].append(k)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)

    return pd.DataFrame(model_results)


def visualize_topics(lda_model, corpus, id2word, num_topics):
    # Visualize the topics
    pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))

    # Check if the pyLDAvis file already exists
    if not os.path.exists(LDAvis_data_filepath):
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    else:
        # Load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)

    # Save HTML representation of the visualization
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')

    return LDAvis_prepared


# Main execution
if __name__ == "__main__":
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Load and preprocess the data
    papers = load_data("./data/NIPS Papers.zip")
    papers_preprocessed = preprocess_data(papers)

    # Further process the papers
    data_words = list(sent_to_words(papers_preprocessed['paper_text_processed'].values.tolist()))
    data_ready = process_words(data_words, stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary and Corpus
    id2word = corpora.Dictionary(data_ready)
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    lda_model = build_lda_model(corpus, id2word, num_topics=10)

    # Tune Hyperparameters (optional)
    # tune_hyperparameters(corpus, id2word)

    # Visualize the topics
    visualize_topics(lda_model, corpus, id2word, num_topics=10)