import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import nltk
from nltk.corpus import conll2000
import pickle

#unique_list = list(OrderedDict(izip(tokens, repeat(None))))
#print tokens[len(tokens)-1]
#data = cv.fit_transform(tokens).toarray()

# lines = f.readline().strip('\r')
wordList = []
tagList = []
LabelList = []

wordDict = dict()
tagDict = dict()
#f_v1 = open("vectors1.txt", "rU")  # opens file with name of "test.txt"

'''
for line in f_v1:
    tmp = line.strip('\n')
    words = tmp.split(' ')
    wordList.append(words[0])
    wordDict[words[0]] = words[1:]
f_v1.close()

f_v2 = open("vectors2.txt", "rU")  # opens file with name of "test.txt"
for line in f_v2:
    tmp = line.strip('\n')
    words = tmp.split(' ')
    tagList.append(words[0])
    tagDict[words[0]] = words[1:]
f_v2.close()

print len(wordDict)
train_c1 = open("Data/Column1", "rU")

batch_size = len(tagDict[wordList[0]])
print batch_size
'''
def getWordList(filename):
    wordList = []
    f = open(filename, "rU")
    words = f.read().split('\n')
    return words

# def getDict(filename):
#     d = dict()
#     f = open(filename, "rU")
#     for line in f:
#         tmp = line.strip('\n')
#         words = tmp.split(' ')
#         wordList.append(words[0])
#         d[words[0]] = words[1:]
#     f.close()
#     return d



def getXTrian(offset, batch_size, wordlist, worddict, tagdict):
    w = len(worddict[wordlist[0]])
    h = batch_size
    x_word_mat = np.zeros(w * h).reshape(h, w)
    x_tag_mat = np.zeros(w * h).reshape(h, w)

    for i in range(offset, batch_size + offset):
        word = wordlist[i]
        x_word_mat[i] = worddict[word]
        #x_tag_mat[i] = tagDict[word]

    return x_word_mat, x_tag_mat

def trimDataSet(filename):
    f = open(filename, "rU")
    words = []
    tags = []
    count = 0
    for line in f:
        count = count + 1
        if(len(line) < 4):
            print line
            continue
        line = line.strip('\n')
        terms = line.split('\t')
        words.append(terms[0])
        tags.append(terms[1])
    print count
    return words, tags


#to get ylabel in index form
#import PreProcess as pp
#yT_fn = "Data/Train/Column3"
#yTrain = pp.getLabel(pp.initVocab(), yT_fn)
def getLabel(vocab, filename):
    cv = CountVectorizer(vocabulary=vocab)
    c3 = open(filename, "rU")
    line = c3.readline().strip(" ")
    tokens = line.split(" ")
    le = preprocessing.LabelEncoder()
    le.fit(vocab)
    data = le.transform(tokens)
    # unique_list = list(OrderedDict(izip(tokens, repeat(None))))
    # print tokens[len(tokens)-1]
    # data = cv.fit_transform(tokens).toarray()
    return data

def getLabelMat(vocab, filename, w):
    cv = CountVectorizer(vocabulary=vocab)
    c3 = open(filename, "rU")
    line = c3.readline().strip(" ")
    tokens = line.split(" ")
    lb = preprocessing.LabelBinarizer()
    lb.fit(vocab)
    data = lb.transform(tokens)
    d_w = data.shape[1]
    d_h = data.shape[0]
    padding = np.zeros((d_h, w-d_w))

    output = np.hstack((data, padding))
    # unique_list = list(OrderedDict(izip(tokens, repeat(None))))
    # print tokens[len(tokens)-1]
    # data = cv.fit_transform(tokens).toarray()
    c3.close()
    return output
def initVocab():
    vocab = []
    for i in range(3):
        if i == 0:
            a = "I"
        if i == 1:
            a = "B"
        if i == 2:
            a = "O"
        for j in range(11):
            if j == 0:
                b = "ADJP"
            if j == 1:
                b = "ADP"
            if j == 2:
                b = "CONJP"
            if j == 3:
                b = "INTJ"
            if j == 4:
                b = "LST"
            if j == 5:
                b = "NP"
            if j == 6:
                b = "PP"
            if j == 7:
                b = "PRT"
            if j == 8:
                b = "SBAR"
            if j == 9:
                b = "UCP"
            if j == 10:
                b = "VP"
            vocab.append(a + "-" + b)
    #print vocab
    vocab.append("I")
    vocab.append("O")
    vocab.append("B")
    vocab.append("")
    vocab.append("None")
    return vocab


class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        return chunktags


def getnltkChunkLabel3(filename):
    f = open(filename, "rU")
    tagged_words = []
    for line in f.readlines():
        (word, pos, chunk) = line.strip('\n').split("\t")
        tagged_words.append((word,pos))

    train_sents = conll2000.chunked_sents('train.txt')

    chunker = BigramChunker(train_sents)
    predicts = []
    sentence = []
    for (w, t) in tagged_words:
        sentence.append((w, t))
        if w == ".":
            for tag in chunker.parse(sentence):
                predicts.append(str(tag))
            sentence = []

    thefile = open("nltk_chunk_tag_"+filename, "wb")
    for item in predicts:
        thefile.write("%s " % item)
    thefile.close()

def indexToLabel(filename, vocab):

    output = pickle.load(open(filename, 'rb'))
    return output


