import spacy
import glob
import codecs
import os

from gensim.models import Phrases
from gensim.models.word2vec import LineSentence


# load english
nlp = spacy.load('en')


# file paths
directory = 'orderModels/'
episode_directory = 'data/'
all_txt_filepath = os.path.join(episode_directory,'all_text.txt')
unigram_sentences_filepath = os.path.join(directory,'unigram_sentences_all.txt')
bigram_sentences_filepath = os.path.join(directory,'bigram_sentences_all.txt')
trigram_sentences_filepath = os.path.join(directory,'trigram_sentences_all.txt')
triJOIN_sentences_filepath = os.path.join(directory,'trigram_sentences_alljoined.txt')



# combine all episodes 
with open(all_txt_filepath, 'w') as outfile:
    for f in glob.glob(episode_directory + 's*'):
        with open(f) as infile:
            for line in infile:
                outfile.write(line)
            outfile.write('\n')



def punct_or_space(token):
    return token.is_punct or token.is_space

def line_review(filename):

    with codecs.open(filename, encoding='utf-8') as f:
        for episode in f:
            yield episode.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):

    for parsed_episode in nlp.pipe(line_review(filename),
                                  batch_size=1000, n_threads=4):
        
        for sent in parsed_episode.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_or_space(token)
                             if token not in spacy.en.STOP_WORDS])


# save unigrams
print 'Saving Unigrams....'
with codecs.open(unigram_sentences_filepath, 'w', encoding='utf-8') as f:
        for sentence in lemmatized_sentence_corpus(all_txt_filepath):
            f.write(sentence + '\n')
  
# load unigrams            
unigram_sentences = LineSentence(unigram_sentences_filepath)


# create bigrams
bigram_model = Phrases(unigram_sentences)

# save bigrams
print 'Saving Bigrams....'
with codecs.open(bigram_sentences_filepath, 'w', encoding='utf-8') as f:   
        for unigram_sentence in unigram_sentences:   
            bigram_sentence = u' '.join(bigram_model[unigram_sentence]) 
            f.write(bigram_sentence + '\n')
     
# load bigrams            
bigram_sentences = LineSentence(bigram_sentences_filepath)

trigram_model = Phrases(bigram_sentences)

# save trigrams
print 'Saving Trigrams....'
with codecs.open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:   
        for bigram_sentence in bigram_sentences:
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])
            f.write(trigram_sentence + '\n')


# save trigrams to blob
print 'Saving Trigrams....'
with codecs.open(triJOIN_sentences_filepath, 'w', encoding='utf-8') as f:
        for parsed_review in nlp.pipe(line_review(all_txt_filepath),
                                      batch_size=1000, n_threads=4):
            # lemmatize the text, removing punctuation and whitespace
            unigrams = [token.lemma_ for token in parsed_review
                              if not punct_or_space(token)]
            # apply the first-order and second-order phrase models
            bigrams = bigram_model[unigrams]
            trigrams = trigram_model[bigrams]
            # remove any remaining stopwords
            trigram = [term for term in trigrams
                              if term not in spacy.en.STOP_WORDS]
            # write the transformed review as a line in the new file
            trigram = u' '.join(trigram)
            f.write(trigram + '\n')












