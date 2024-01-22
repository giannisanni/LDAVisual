import pandas as pd
import os
import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import numpy as np
import tqdm
import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis
import spacy
from spacy.lang.nl import Dutch

def load_data(file_path):
    return pd.read_excel(file_path, usecols=['Titel', 'Onderwerp'])

def preprocess_data(data):
    # Combine 'titel' and 'onderwerp' into one text column for processing
    data['combined_text'] = data['Titel'].astype(str) + " " + data['Onderwerp'].astype(str)
    # Process the combined text
    data['text_processed'] = data['combined_text'].map(lambda x: re.sub('[,\\.!?]', '', x))
    data['text_processed'] = data['text_processed'].map(lambda x: x.lower())
    return data

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# Load Spacy's Dutch model
nlp = spacy.load("nl_core_news_lg")

# Define Dutch stopwords
dutch_stop_words = set([
    "de", "en", "van", "ik", "te", "dat", "die", "in", "een", "hij",
    "het", "niet", "zijn", "is", "was", "op", "aan", "met", "als", "voor",
    "had", "er", "maar", "om", "hem", "dan", "zou", "of", "wat", "mijn",
    "men", "dit", "zo", "door", "over", "ze", "zich", "bij", "ook", "tot",
    "je", "mij", "uit", "der", "daar", "haar", "naar", "heb", "hoe", "heeft",
    "hebben", "deze", "u", "want", "nog", "zal", "me", "zij", "nu", "ge",
    "geen", "omdat", "iets", "worden", "toch", "al", "waren", "veel", "meer",
    "doen", "toen", "moet", "ben", "zonder", "kan", "hun", "dus", "alles",
    "onder", "ja", "eens", "hier", "wie", "werd", "altijd", "doch", "wordt",
    "wezen", "kunnen", "ons", "zelf", "tegen", "na", "reeds", "wil", "kon",
    "niets", "uw", "iemand", "geweest", "andere", "verzoek", "besluit", "open" , "wet", "basis"
])

def process_words(texts, stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
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

def build_lda_model(corpus, id2word, num_topics=20):
    return gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, chunksize=100, passes=10, per_word_topics=True)

def compute_coherence_values(corpus, dictionary, texts, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=k, random_state=100,
                                           chunksize=100, passes=10, alpha=0.01, eta=0.9)
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
    # Prepare LDA visualization
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

    # Define the file path for saving the HTML file
    html_file_path = 'C:/Users/gsanr/PycharmProjects/LDA_topic_modeling/ldavis_prepared_{}ALLdata.html'.format(num_topics)

    # Save as an HTML file
    pyLDAvis.save_html(LDAvis_prepared, html_file_path)

    return html_file_path  # Return the path to the saved HTML file


if __name__ == "__main__":
    # Load and preprocess the data
    file_path = 'C:/Users/gsanr/PycharmProjects/LDA_topic_modeling/original_merged_data.xlsx'

    papers = load_data(file_path)
    papers_preprocessed = preprocess_data(papers)

    # Further process the papers
    data_words = list(sent_to_words(papers_preprocessed['text_processed'].values.tolist()))
    data_ready = process_words(data_words, dutch_stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Check if data_ready is not empty
    print("Sample processed data:", data_ready[:1])

    # Create Dictionary and Corpus
    id2word = corpora.Dictionary(data_ready)
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Check the contents of id2word and corpus
    print("Number of unique tokens:", len(id2word))
    print("Number of documents:", len(corpus))

    # Ensure that the corpus is not empty before building the LDA model
    if len(corpus) > 0 and len(id2word) > 0:
        lda_model = build_lda_model(corpus, id2word, num_topics=20)
    else:
        print("Error: The corpus or id2word dictionary is empty.")

    # Create Dictionary and Corpus
    id2word = corpora.Dictionary(data_ready)
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    lda_model = build_lda_model(corpus, id2word, num_topics=20)

    # Tune Hyperparameters (optional)
    # model_tuning_results = tune_hyperparameters(corpus, id2word, data_ready)

    # Visualize the topics and get the path to the saved HTML file
    html_visualization_path = visualize_topics(lda_model, corpus, id2word, num_topics=20)
    print(f"Topic visualization saved to: {html_visualization_path}")
