import os


directory = 'orderModels/'
episode_directory = 'data/'
all_txt_filepath = os.path.join(episode_directory,'all_text.txt')
unigram_sentences_filepath = os.path.join(directory,'unigram_sentences_all.txt')
bigram_sentences_filepath = os.path.join(directory,'bigram_sentences_all.txt')
trigram_sentences_filepath = os.path.join(directory,'trigram_sentences_all.txt')
triJOIN_sentences_filepath = os.path.join(directory,'trigram_sentences_alljoined.txt')

trigram_dictionary_filepath = os.path.join(directory,'trigram_dict_all.dict')
trigram_bow_filepath = os.path.join(directory,'trigram_bow_corpus_all.mm')
lda_model_filepath = os.path.join(directory, 'lda_model_all')