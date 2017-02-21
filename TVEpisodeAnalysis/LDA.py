import codecs
import itertools
import spacy
import warnings
import random
warnings.filterwarnings("ignore")

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

import pyLDAvis
import pyLDAvis.gensim

import colorama
from colorama import Fore, Back, Style

from config import *

colorama.init()
nlp = spacy.load('en')



def punct_or_space(token):
    return token.is_punct or token.is_space

def line_review(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        for episode in f:
            yield episode.replace('\\n', '\n')
    
def get_sample(episode_number):
    return list(itertools.islice(line_review(all_txt_filepath),
                          episode_number, episode_number+1))[0]
        
def lda_description(text, bigram_model, trigram_model, trigram_dictionary, topn=25):
    
    parsed_episode = nlp(text)    
    unigram_episode = [token.lemma_ for token in parsed_episode
                      if not punct_or_space(token)]
    
    bigram_episode = bigram_model[unigram_episode]
    trigram_episode = trigram_model[bigram_episode]
    trigram_episode = [term for term in trigram_episode
                      if not term in spacy.en.STOP_WORDS]
    
    bow = trigram_dictionary.doc2bow(trigram_episode)
    episode_lda = lda[bow]
    episode_lda = sorted(episode_lda, key=lambda (topic_number, freq): -freq)
    keywords = []
    for term in lda.show_topic(episode_lda[0][0], topn=topn):
        keywords.append(term[0])
        
    return keywords


def print_sample_episode(sample_number, bigram_model, trigram_model, trigram_dictionary):
	new_text = []
	episode = get_sample(sample_number)
	keywords = lda_description(episode, bigram_model, trigram_model, trigram_dictionary)

	for word in episode.split():
	    
	    if word.lower() in keywords:
	        word = Fore.RED + word + Style.RESET_ALL
	    # need to get all variations of the word. Lemma, cap, etc... Maybe similar words with spacy? 
	    new_text.append(word)
	    
	print ' '.join(new_text)


def main(sample_episde=False, sample=random.randint(0, 100), visualize=False):

	trigram_sentences = LineSentence(triJOIN_sentences_filepath)
	trigram_dictionary = Dictionary(trigram_sentences)
	trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)
	trigram_dictionary.compactify()


	trigram_bow_corpus = [trigram_dictionary.doc2bow(episode) for episode in trigram_sentences]


	lda = LdaMulticore(corpus=trigram_bow_corpus, num_topics=18, id2word=trigram_dictionary, workers=3, passes=10)
	lda.save(lda_model_filepath)

	unigram_sentences = LineSentence(unigram_sentences_filepath)
	bigram_model = Phrases(unigram_sentences)
	bigram_sentences = LineSentence(bigram_sentences_filepath)
	trigram_model = Phrases(bigram_sentences)

	if sample_episde:
		print_sample_episode(sample, bigram_model, trigram_model, trigram_dictionary)
# new_text = []
# episode = get_sample(6)
# keywords = lda_description(episode)

# for word in episode.split():
    
#     if word.lower() in keywords:
#         word = Fore.RED + word + Style.RESET_ALL
#     # need to get all variations of the word. Lemma, cap, etc... Maybe similar words with spacy? 
#     new_text.append(word)
    
# print ' '.join(new_text)

	if visualize:
		LDAvis_prepared = pyLDAvis.gensim.prepare(lda, trigram_bow_corpus, trigram_dictionary)
		pyLDAvis.show(LDAvis_prepared)



# if __name__ == '__main__':
# 	main(sample_episde=True, visualize=True)



class LDA(object):
	def __init__(self):
		pass

	def fit(self, num_topics, passes=10, save_model=False):

		# if model:
		# self.lda = LdaMulticore.load(lda_model_filepath)
		# else: 
		self.lda = LdaMulticore(corpus=self.bow, num_topics=num_topics, 
							id2word=self.trigram_dictionary, workers=3, passes=passes)
		
		if save_model:
			self.lda.save(lda_model_filepath)

	def visualize(self):
		LDAvis_prepared = pyLDAvis.gensim.prepare(self.lda, self.bow, self.trigram_dictionary)
		pyLDAvis.show(LDAvis_prepared)

	def lda_description(self, text, topn=25):
    
		parsed_episode = nlp(text)    
		unigram_episode = [token.lemma_ for token in parsed_episode
		                  if not punct_or_space(token)]

		unigram_sentences = LineSentence(unigram_sentences_filepath)
		bigram_model = Phrases(unigram_sentences)
		bigram_sentences = LineSentence(bigram_sentences_filepath)
		trigram_model = Phrases(bigram_sentences)                  

		bigram_episode = bigram_model[unigram_episode]
		trigram_episode = trigram_model[bigram_episode]
		trigram_episode = [term for term in trigram_episode
		                  if not term in spacy.en.STOP_WORDS]

		bow = self.trigram_dictionary.doc2bow(trigram_episode)
		episode_lda = self.lda[bow]
		episode_lda = sorted(episode_lda, key=lambda (topic_number, freq): -freq)
		keywords = []
		for term in self.lda.show_topic(episode_lda[0][0], topn=topn):
		    keywords.append(term[0])
		    
		return keywords


	def evaluate(self, episode=get_sample(random.choice(range(100))), topn=25, random_sample=False):
		output = []
		if random_sample:
			episode = get_sample(6)
		else:
			episode = episode

		keywords = self.lda_description(episode, topn)

		for word in episode.split():
		    if word.lower() in keywords:
		        word = Fore.RED + word + Style.RESET_ALL
		    # need to get all variations of the word. Lemma, cap, etc... Maybe similar words with spacy? 
		    output.append(word)
		    
		print ' '.join(output)



	def create_dictionary(self, filename, no_below=10, no_above=0.4):
		trigram_sentences = LineSentence(filename)
		self.trigram_dictionary = Dictionary(trigram_sentences)
		self.trigram_dictionary.filter_extremes(no_below=no_below, no_above=no_below)
		self.trigram_dictionary.compactify()
		self.bow = self.create_bow(self.trigram_dictionary, trigram_sentences)
		return self.trigram_dictionary, trigram_sentences, self.bow

	def create_bow(self, dictionary, sentences):
		return [dictionary.doc2bow(episode) for episode in sentences]


if __name__ == '__main__':

	model = LDA()
	trigram_dictionary, trigram_sentences, bow = model.create_dictionary(triJOIN_sentences_filepath)
	# model.fit(num_topics = 18, corpus = bow, id2word = trigram_dictionary)
	model.fit(num_topics = 18)
	model.evaluate()
	model.visualize()











