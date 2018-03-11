 #!/usr/bin/env python -W ignore::DeprecationWarning
import pandas as pd, numpy as np
import nltk
import sklearn_crfsuite
import eli5
import matplotlib.pyplot as plt
import pycrfsuite

from itertools import chain
import re

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import nltk.tag.stanford as st
from itertools import groupby

tagger = st.StanfordNERTagger('../../stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
               '../../stanford-ner/stanford-ner.jar') 

def get_continuous_chunks(tag,tagged_sent):
    continuous_chunk = []
    current_chunk = []

    for token, tag in tagged_sent:
    	if tag == tag:
        # if tag == "PERSON":
            current_chunk.append((token, tag))
        else:
            if current_chunk: # if the current chunk is not empty
                continuous_chunk.append(current_chunk)
                current_chunk = []
    # Flush the final current_chunk into the continuous_chunk, if any.
    if current_chunk:
        continuous_chunk.append(current_chunk)
    return continuous_chunk


def entity_list(train_file, label, tag, save_folder):
	train = pd.read_csv(train_file)
	# test = pd.read_csv('../input/test.csv')
	# subm = pd.read_csv('../input/sample_submission.csv')
	 
	selected = train.loc[train[label]==1]
	select_comments = selected["comment_text"]
	comments = select_comments.as_matrix()
	# r=tagger.tag('John Eid is studying at Stanford University in NY'.split())
	# print(r)
	names = []
	count = 0
	for comment in comments:
		count+=1
		# if count<200:
		# 	continue
		r=tagger.tag(comment.decode('utf-8').strip().split())
		c = get_continuous_chunks(tag,r)
		c2 = [" ".join([token for token, tag in ne]) for ne in c]
		names = names+c2	
		if count%100 ==0:
			print(names)
			namelist = names
			names = []
			filename = save_folder+'entity_'+str(count)+'.txt'
			with open(filename, 'w') as file:
				for item in namelist:
					item = item.encode('utf-8').strip()
		  			file.write("%s\n" % item)

if __name__ == '__main__':
	train_file = '../input/train.csv'
	label = "identity_hate"
	save_folder = "locations/"
	tag = "PERSON"
	entity_list(train_file, label,tag, save_folder)






# chunked = nltk.ne_chunk(comments[0], binary=True)

# train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
# test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

# def word2features(sent, i):
#     word = sent[i][0]
#     postag = sent[i][1]

#     features = {
#         'bias': 1.0,
#         'word.lower()': word.lower(),
#         'word[-3:]': word[-3:],
#         'word.isupper()': word.isupper(),
#         'word.istitle()': word.istitle(),
#         'word.isdigit()': word.isdigit(),
#         'postag': postag,
#         'postag[:2]': postag[:2],
#     }
#     if i > 0:
#         word1 = sent[i-1][0]
#         postag1 = sent[i-1][1]
#         features.update({
#             '-1:word.lower()': word1.lower(),
#             '-1:word.istitle()': word1.istitle(),
#             '-1:word.isupper()': word1.isupper(),
#             '-1:postag': postag1,
#             '-1:postag[:2]': postag1[:2],
#         })
#     else:
#         features['BOS'] = True

#     if i < len(sent)-1:
#         word1 = sent[i+1][0]
#         print(word1, sent[i+1])
#         postag1 = sent[i+1][1]
#         features.update({
#             '+1:word.lower()': word1.lower(),
#             '+1:word.istitle()': word1.istitle(),
#             '+1:word.isupper()': word1.isupper(),
#             '+1:postag': postag1,
#             '+1:postag[:2]': postag1[:2],
#         })
#     else:
#         features['EOS'] = True

#     return features


# def sent2features(sent):
#     return [word2features(sent, i) for i in range(len(sent))]

# def sent2labels(sent):
#     return [label for token, postag, label in sent]

# def sent2tokens(sent):
#     return [token for token, postag, label in sent]

# X_train = [sent2features(s) for s in train_sents]
# y_train = [sent2labels(s) for s in train_sents]

# X_test = [sent2features(s) for s in test_sents]
# y_test = [sent2labels(s) for s in test_sents]


# trainer = pycrfsuite.Trainer(verbose=False)

# for xseq, yseq in zip(X_train, y_train):
#     trainer.append(xseq, yseq)

# trainer.set_params({
#     'c1': 1.0,   # coefficient for L1 penalty
#     'c2': 1e-3,  # coefficient for L2 penalty
#     'max_iterations': 50,  # stop earlier

#     # include transitions that are possible, but not observed
#     'feature.possible_transitions': True
# })

# trainer.train('conll2002-esp.crfsuite')

# tagger = pycrfsuite.Tagger()
# tagger.open('conll2002-esp.crfsuite')

#example_sent = test_sents[0]
# print(' '.join(sent2tokens(example_sent)))
#print(example_sent)
# sentence = "I am John from America"
# # sentence = "La Coruna , 23 may ( EFECOM )" #comments[0]
# sent1 = nltk.word_tokenize(sentence)
# sent2 = nltk.pos_tag(sent1)
# sent3 = nltk.ne_chunk(sent2, binary=True)
# sent4=[]
# for c in sent3:
#   if hasattr(c, 'node'):
#     sent4.append(' '.join(i[0] for i in c.leaves()))
# print(sent2,sent3, sent4)

# print("Predicted:", ' '.join(tagger.tag(sent2features(sent3))))
# print("Correct:  ", ' '.join(sent2labels(example_sent)))

# sentences =[ line.decode('utf-8').strip()  for line in comments[:10]]
# tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
# tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
# chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)


# def extract_entity_names(t):
#     entity_names = []

#     if hasattr(t, 'label') and t.label:
#         if t.label() == 'NE':
#             entity_names.append(' '.join([child[0] for child in t]))
#         else:
#             for child in t:
#                 entity_names.extend(extract_entity_names(child))

#     return entity_names

# entity_names = []
# for tree in chunked_sentences:
#     # Print results per sentence
#     # print extract_entity_names(tree)

#     entity_names.extend(extract_entity_names(tree))

# # Print all entity names
# #print entity_names

# # Print unique entity names
# print set(entity_names)





