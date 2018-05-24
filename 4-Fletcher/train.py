FN = 'train'
FN0 = 'vocabulary-embedding'
FN1 = 'train'

import os

os.environ['KERAS_BACKEND'] = 'theano'

os.environ["MKL_THREADING_LAYER"] = "GNU"
import keras
keras.__version__

maxlend = 100 #number of words from transcript
maxlenh = 25 #number of words from title
maxlen = maxlend + maxlenh  #total length of input

rnn_size = 512
rnn_layers = 3
batch_norm = False

activation_rnn_size = 40 if maxlend else 0

#Training Parameters
seed = 42
p_W, p_U, p_dense, p_emb, weight_decay = 0,0,0,0,0
optimizer = 'Adam'
LR = 1e-4
batch_size = 64
n_flips = 10

nb_train_samples = 5000
nb_val_samples = 500



import pickle

with open('data/%s.pkl'%FN0, 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape

with open('data/%s.data.pkl'%FN0, 'rb') as fp:
    X, Y = pickle.load(fp)

nb_unknown_words = 30

print ('number of examples',len(X),len(Y))
print ('dimension of embedding space for words',embedding_size)
print ('vocabulary size', vocab_size, 'the last %d words can be used as place holders for unknown/oov words'%nb_unknown_words)
print ('total number of different words',len(idx2word), len(word2idx))
print ('number of words outside vocabulary which we can substitue using glove similarity', len(glove_idx2idx))
print ('number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',len(idx2word)-vocab_size-len(glove_idx2idx))


for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i

oov0 = vocab_size-nb_unknown_words


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)
len(X_train), len(Y_train), len(X_test), len(Y_test)


del X
del Y

empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'

import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random,sys


def prt(label, x):
    print(label+':',)
    for w in x:
        print(idx2word[w],)
    print()


from keras.models import Sequential
from keras.layers import Merge, SpatialDropout1D
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2

random.seed(seed)
np.random.seed(seed)

regularizer = l2(weight_decay) if weight_decay else None

#Initialize the model with sequential
model = Sequential()

#add an embedding layer
model.add(Embedding(vocab_size,embedding_size,
                   input_length = maxlen, 
                   activity_regularizer = regularizer, weights = [embedding],
                   mask_zero = True, name = 'embedding_1'))

#add the dropout layer (replaces the dropout paramater in embedding layer)
#model.add(SpatialDropout1D(rate = p_emb,name='dropout_0'))


#add #rnn_layers (start with 3) of LSTMs
for i in range(rnn_layers):
    lstm = LSTM(rnn_size, return_sequences=True, #batch_norm = batch_norm,
                kernel_regularizer = regularizer, recurrent_regularizer = regularizer,
                bias_regularizer = regularizer, dropout = p_W, recurrent_dropout = p_U,
                name = 'lstm_%d'%(i+1)
               )
    
    model.add(lstm)
    #add a dropout layer for each LSTM
    model.add(Dropout(rate = p_dense, name = 'dropout_%d'%(i+1)))
                
from keras.layers.core import Lambda
import keras.backend as K

def simple_context(X, mask, n = activation_rnn_size, 
                   maxlend = maxlend, maxlenh = maxlenh):
    desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
    head_activations, head_words = head[:,:,:n],head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n],desc[:,:,n:]
    
    #activation for every head word and desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    
    #don't use description words that are masked out
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:,:maxlend],dtype='float32'),1)
    
    #for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights, (-1,maxlenh,maxlend))
    
    #for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights,desc_words,axes=(2,1))
    return K.concatenate((desc_avg_word,head_words))


class SimpleContext(Lambda):
    def __init__(self,**kwargs):
        super(SimpleContext, self).__init__(simple_context,**kwargs)
        self.supports_masking = True
    
    def compute_mask(self,input, input_mask=None):
        return input_mask[:,maxlend:]
    
    def compute_output_shape(self,input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples,maxlenh,n)

if activation_rnn_size:
    model.add(SimpleContext(name='simplecontext_1'))
    
model.add(TimeDistributed(Dense(vocab_size,
                               kernel_regularizer = regularizer, bias_regularizer = regularizer,
                               name = 'timedistributed_1')))
model.add(Activation('softmax',name='activation_1'))


from keras.optimizers import Adam, RMSprop
model.compile(loss='categorical_crossentropy',optimizer=optimizer)

K.set_value(model.optimizer.lr,np.float32(LR))

def str_shape(x):
    return 'x'.join(map(str,x.shape))
    
def inspect_model(model):
    for i,l in enumerate(model.layers):
        print (i, 'cls=%s name=%s'%(type(l).__name__, l.name))
        weights = l.get_weights()
        for weight in weights:
            print (str_shape(weight),)
        print()


inspect_model(model)

if FN1 and os.path.exists('data/%s.hdf5'%FN1):
    model.load_weights('data/%s.hdf5'%FN1)


def lpadd(x,maxlend=maxlend,eos=eos):
    """left pad a description to maxlend and then add eos.
    The eos is the input to predicting the first word in the headline"""
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]

samples = [lpadd([3]*26)]
data = sequence.pad_sequences(samples,maxlen=maxlen, value=empty,padding='post',truncating='post')

np.all(data[:,maxlend]==eos)

data.shape,list(map(len,samples))

probs = model.predict(data,verbose=1,batch_size=1)
probs.shape

def flip_headline(x, n_flips=None, model=None, debug=False):
    """given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline)
    with words predicted by the model
    """
    if n_flips is None or model is None or n_flips <= 0:
        return x
    
    batch_size = len(x)
    assert np.all(x[:,maxlend] == eos)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        # pick locations we want to flip
        # 0...maxlend-1 are descriptions and should be fixed
        # maxlend is eos and should be fixed
        flips = sorted(random.sample(range(maxlend+1,maxlen), n_flips))
        if debug and b < debug:
            print (b,)
        for input_idx in flips:
            if x[b,input_idx] == empty or x[b,input_idx] == eos:
                continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
            label_idx = input_idx - (maxlend+1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty:  # replace accidental empty with oov
                w = oov0
            if debug and b < debug:
                print ('%s => %s'%(idx2word[x_out[b,input_idx]],idx2word[w]),)
            x_out[b,input_idx] = w
        if debug and b < debug:
            print()
    return x_out



def conv_seq_labels(xds, xhs, n_flips=None, model=None, debug=False):
    """description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
    batch_size = len(xhs)
    assert len(xds) == batch_size
    x = [vocab_fold(lpadd(xd)+xh) for xd,xh in zip(xds,xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
    x = flip_headline(x, n_flips=n_flips, model=model, debug=debug)
    
    y = np.zeros((batch_size, maxlenh, vocab_size))
    for i, xh in enumerate(xhs):
        xh = vocab_fold(xh) + [eos] + [empty]*maxlenh  # output does have a eos at end
        xh = xh[:maxlenh]
        y[i,:,:] = np_utils.to_categorical(xh, vocab_size)
        
    return x, y

def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, n_flips=None, model=None, debug=False, seed=seed):
    """yield batches. for training use nb_batches=None
    for validation generate deterministic results repeating every nb_batches
    
    while training it is good idea to flip once in a while the values of the headlines from the
    value taken from Xh to value generated by the model.
    """
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, sys.maxsize)
        random.seed(c+123456789+seed)
        for b in range(batch_size):
            t = random.randint(0,len(Xd)-1)

            xd = Xd[t]
            s = random.randint(min(maxlend,len(xd)), max(maxlend,len(xd)))
            xds.append(xd[:s])
            
            xh = Xh[t]
            s = random.randint(min(maxlenh,len(xh)), max(maxlenh,len(xh)))
            xhs.append(xh[:s])

        # undo the seeding before we yield inorder not to affect the caller
        c+= 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, n_flips=n_flips, model=model, debug=debug)


r = next(gen(X_train, Y_train, batch_size=batch_size))
r[0].shape, r[1].shape, len(r)

def test_gen(gen, n=5):
    Xtr,Ytr = next(gen)
    for i in range(n):
        assert Xtr[i,maxlend] == eos
        x = Xtr[i,:maxlend]
        y = Xtr[i,maxlend:]
        yy = Ytr[i,:]
        yy = np.where(yy)[1]
        prt('L',yy)
        prt('H',y)
        if maxlend:
            prt('D',x)


valgen = gen(X_test, Y_test,nb_batches=3, batch_size=batch_size)

history = {}

traingen = gen(X_train, Y_train, batch_size=batch_size, n_flips=n_flips, model=model)
valgen = gen(X_test, Y_test, nb_batches=nb_val_samples//batch_size, batch_size=batch_size)

r = next(traingen)
r[0].shape, r[1].shape, len(r)


for iteration in range(100):
    print ('Iteration', iteration)
    h = model.fit_generator(traingen, steps_per_epoch=nb_train_samples//batch_size,
                        epochs=1, validation_data=valgen, validation_steps=nb_val_samples//batch_size,
                            use_multiprocessing=True
                           )
    for k,v in h.history.iteritems():
        history[k] = history.get(k,[]) + v
    with open('data/%s.history.pkl'%FN,'wb') as fp:
        pickle.dump(history,fp,-1)
    model.save_weights('data/%s.hdf5'%FN, overwrite=True)
    gensamples(batch_size=batch_size)


