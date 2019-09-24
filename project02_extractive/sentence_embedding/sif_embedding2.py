import sys, numpy
sys.path.append('../src')
import data_io, params, SIF_embedding

# input
wordfile = '../data/glove.840B.300d.txt' # word vector file, can be downloaded from GloVe website; it's quite large but you can truncate it and use only say the top 50000 word vectors to save time
weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
sentences = ['this is an example sentence', 'this is another sentence that is slightly longer', 'the quick brown fox jumps over the lazy dog']

# load word vectors
(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# load sentences
x, m = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
w = data_io.seq2weight(x, m, weight4ind) # get word weights

# set parameters
params = params.params()
params.rmpc = rmpc
# get SIF embedding
embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i

#cosin
emb1 = embedding[0,:]
emb2 = embedding[1,:]
inn = (emb1 * emb2).sum()
emb1norm = numpy.sqrt((emb1 * emb1).sum())
emb2norm = numpy.sqrt((emb2 * emb2).sum())
score = inn / emb1norm / emb2norm  #inn / (emb1norm * emb2norm )




