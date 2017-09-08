import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import nltk
from nltk.corpus import conll2000

def initVocab():
    '''
    initVocab() initializes a list containing all the BIO tags
    '''
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
    return vocab

def initVocabWithNone():
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

def BIO2Idx():
    '''
    BIO2Idx() returns dictionary containing
    a BIO tag list and index array 

    @return
            yLabelDictionary
                {
                    'tag': [|tags|],
                    'index': [|index|]
                }
    '''
    vocab = initVocabWithNone()
    # sort the vocab in alphabetical order
    vocab = sorted(vocab)
    yLabelDictionary = {}
    index = [i for i in range(1, len(vocab) + 1)]
    yLabelDictionary['tag'] = vocab
    yLabelDictionary['index'] = index
    return yLabelDictionary

def BIO2vec():
    iobvec = getLabelMat(initVocabWithNone(), \
        "nltk_chunk_tag_train10601_re.txt", 45)
    print iobvec
    return iobvec

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

def words2img(winSize = 5, IOBtag = False):
    '''
    words2img() read word vector and pos vector, concatenate them horizontally,
    and produce word_img
    @return word_img : shape = (157283, 90 * winSize)
    '''
    #path_word = raw_input("Please input the path of words vectors: ")
    #path_pos = raw_input("Please input the path of pos-tag vectors: ")
    path_word = "channel1.txt"
    path_pos = "channel2.txt"
    path_iob = "nltk_chunk_tag_train10601_re.txt"
    wordsvec = np.genfromtxt(path_word, delimiter=' ')
    posvec = np.genfromtxt(path_pos, delimiter=' ')
    source_vec = np.hstack((wordsvec, posvec))
    if IOBtag == True:
        iobvec = BIO2vec()
        print iobvec.shape
        source_vec = np.hstack((source_vec, iobvec))
    word_img = []
    length = source_vec.shape[1]
    for i in xrange(source_vec.shape[0]):
        # for each word vector, generate 
        # per_img which is centered at current
        # word vector with (winSize / 2) word vectors before it
        # and winSize - (winSize / 2) - 1 word vectors
        # after it
        # for example, window size if 5 (winSize = 5)
        # the length of per_img is (45    +    45)    *    5 = 4   50
        #                           |           |
        #                        wordvec     posvec
        per_img = []
        if i < int(winSize) / 2:
            temp = int(winSize) / 2
            for k in range(winSize):
                if k < temp:
                    per_img = np.hstack((per_img, np.zeros(length)))
                else:
                    per_img = np.hstack((per_img, source_vec[i + k - temp]))
        elif source_vec.shape[0] - i <= int(winSize) / 2:
            temp = int(winSize) / 2
            for k in range(winSize):
                if i + k < source_vec.shape[0]:
                    per_img = np.hstack((per_img, source_vec[i + k]))
                else:
                    per_img = np.hstack((per_img, np.zeros(length)))
        else:
            for k in range(i - int(winSize) / 2, i + 1 + int(winSize) / 2):
                per_img = np.hstack((per_img, source_vec[k]))
        word_img.append(per_img)
    return np.array(word_img)

def getLabelIndex(yLabelDictionary, path = "yLabel.txt"):
    '''
    getLabelIndex() takes yLabelDictionary,
    and read BIO tag file
    convert a BIO label to its corresponding
    index
    @return yLabelIndex : shape = (157283, ) 
    '''
    #path = raw_input("Please input the BIO tag file: ")
    # path = "/home/lui/CMU/hw9/testProcessing/BIO_tag_train.txt"
    file = open(path, 'r')
    yLabelIndex = []
    while (True):
        line = file.readline()
        if (line == ""):
            break
        tag = line[:-1]
        if (tag in yLabelDictionary['tag']):
            yLabelIndex.append(
                yLabelDictionary['index'][yLabelDictionary['tag'].index(tag)])
    file.close()
    yLabelIndex = np.array(yLabelIndex)
    return yLabelIndex

def load_data(predict, iobtag = False):
    '''
    This function has the same responsibility as that of function 
    "cnn_lenet.load_mnist(fullset)"

    load_data() takes word_img and yLabelDictionary,
    calls getLabelIndex(),
    produce xTrain, yTrain, xTest, yTest.
    @return 
        xTrain : shape = (winSize * 90, trainNum)
        yTrain : shape = (trainNum, )
        xTest : shape = (winSize * 90, word_img.shape[0] - trainNum)
        yTest : shape = (word_img.shape[0] - trainNum, )
    '''

    word_img = words2img(winSize=7, IOBtag = iobtag)
    yLabelDictionary = BIO2Idx()

    yLabelIndex = getLabelIndex(yLabelDictionary)
    print "shape: word_img, yLabelIndex"
    print word_img.shape, yLabelIndex.shape
    # shuffle
    word_img_indices = range(word_img.shape[0])
    np.random.shuffle(word_img_indices)
    word_img.take(word_img_indices, axis = 0, out = word_img)
    yLabelIndex.take(word_img_indices, axis = 0, out = yLabelIndex)

    # partition, test and train
    trainNum = int(math.floor(word_img.shape[0] / 5 * 4))
    if predict:
        trainNum = word_img.shape[0]
    xTrain = word_img[: trainNum]
    yTrain = yLabelIndex[: trainNum]
    xTest = word_img[trainNum:]
    yTest = yLabelIndex[trainNum:]

    # transpose, columns are sample, rows are featrues
    xTrain = xTrain.T
    xTest = xTest.T
    print "shape: xTrain, yTrain, xTest, yTest"
    print "shape: ", xTrain.shape, yTrain.shape, xTest.shape, yTest.shape
    return (xTrain, yTrain, xTest, yTest)


BIO2vec()