import numpy as np
import pandas as pd
import lasagne
import theano
import theano.tensor as T
from keras.datasets import mnist
import time, shutil, os
import scipy
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
from read_write_model import read_model, write_model
from time import mktime, gmtime
import random
from PIL import Image

theano.config.floatX = 'float32'



#settings
do_train_model = True #False
batch_size = 100
latent_size = 20
nhidden = 512
lr = 0.001
num_epochs = 20 #50
model_filename_read = "mnist_ae"
model_filename_write = "mnist_classifier"
#model_filename = "mnist_ae_adv_trained"
#model_filename = ""
nonlin = lasagne.nonlinearities.rectify

np.random.seed(1234) # reproducibility


#SYMBOLIC VARS
sym_x = T.matrix()
sym_lr = T.scalar('lr')
sym_z = T.matrix()
sym_y = T.matrix()
sym_target = T.vector()

### LOAD DATA
print ("Using MNIST dataset")

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = (train_x.reshape((-1, 784))/255.0).astype(np.float32)
test_x = (test_x.reshape((-1, 784))/255.0).astype(np.float32)


train_y_one_hot = np.zeros((len(train_y),10))
for i in range(0,len(train_y)):
    train_y_one_hot[i,train_y[i]] = 1

test_y_one_hot = np.zeros((len(test_y),10))
for i in range(0,len(test_y)):
    test_y_one_hot[i,test_y[i]] = 1

train_y_one_hot = train_y_one_hot.astype(np.float32)
test_y_one_hot = test_y_one_hot.astype(np.float32)

train_x[train_x > 0.5] = 1.0
train_x[train_x <= 0.5] = 0.0

#setup shared variables
sh_x_train = theano.shared(train_x, borrow=True)
sh_x_test = theano.shared(test_x, borrow=True)
sh_y_train = theano.shared(train_y_one_hot, borrow=True)
sh_y_test = theano.shared(test_y_one_hot, borrow=True)
sh_y_target_train = theano.shared(train_y, borrow = True)
sh_y_target_test = theano.shared(test_y, borrow = True)

nfeatures=train_x.shape[1]
n_train_batches = int(train_x.shape[0] / batch_size)
n_test_batches = int(test_x.shape[0] / batch_size)
print("n_train_batches: ", n_train_batches)
print("n_test_batches: ", n_test_batches)


### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, nfeatures))

l_noise = lasagne.layers.BiasLayer(l_in, b = np.zeros(nfeatures, dtype = np.float32), name = "NOISE")
l_noise.params[l_noise.b].remove("trainable")

l_enc_h1 = lasagne.layers.DenseLayer(l_noise, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE1')

l_enc_h1.params[l_enc_h1.W].remove("trainable")
l_enc_h1.params[l_enc_h1.b].remove("trainable")

l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE2')

l_enc_h1.params[l_enc_h1.W].remove("trainable")
l_enc_h1.params[l_enc_h1.b].remove("trainable")

l_z = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='Z')
l_z.params[l_z.W].remove("trainable")
l_z.params[l_z.b].remove("trainable")

l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE2')
l_dec_h1 = lasagne.layers.DenseLayer(l_dec_h1, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE1')
l_dec_h1.params[l_dec_h1.W].remove("trainable")
l_dec_h1.params[l_dec_h1.b].remove("trainable")

l_dec_x = lasagne.layers.DenseLayer(l_dec_h1, num_units=nfeatures, nonlinearity=lasagne.nonlinearities.sigmoid, name='DEC_X_MU')
l_dec_x.params[l_dec_x.W].remove("trainable")
l_dec_x.params[l_dec_x.b].remove("trainable")

dec_x = lasagne.layers.get_output(l_dec_x, sym_x, deterministic=False)

# In[8]:
l_cls_h1 = lasagne.layers.DenseLayer(l_z, num_units=100, nonlinearity=nonlin, name='CLS_DENSE1')
l_cls_h2 = lasagne.layers.DenseLayer(l_cls_h1, num_units=100, nonlinearity=nonlin, name='CLS_DENSE2')
l_cls_output = lasagne.layers.DenseLayer(l_cls_h2, num_units=10, nonlinearity=lasagne.nonlinearities.softmax,  name='CLS_OUTPUT')
# In[9]:

cls_out = lasagne.layers.get_output(l_cls_output, sym_x, deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(cls_out, sym_y.reshape((batch_size, -1)))
loss = lasagne.objectives.aggregate(loss, mode="mean")

#loss = lasagne.objectives.binary_crossentropy(cls_out, sym_y)
'''
def zero_one_loss(sym_target):
    loss = T.sum(T.neq(np.argmax(l_cls_output), sym_target))
    return loss
'''
#acc = T.scalar()
#acc = np.sum(np.argmax(l_cls_output) == sym_target.reshape((batch_size, -1)))/batch_size


params = lasagne.layers.get_all_params(l_cls_output, trainable=True)

for p in params:
    print (p, p.get_value().shape)

### Take gradient of Negative LogLikelihood
grads = T.grad(loss, params)

# Add gradclipping to reduce the effects of exploding gradients.
# This speeds up convergence
clip_grad = 1
max_norm = 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g,-clip_grad, clip_grad) for g in mgrads]


# In[10]:


#Setup the theano functions
print('shape of train_x: ', np.shape(train_x))
print('shape of train_y: ', np.shape(train_y))
print('shape of train_y_one_hot: ', np.shape(train_y_one_hot))
sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

train_model = theano.function(inputs = [sym_batch_index, sym_lr], outputs = [loss, cls_out], updates=updates,
                             givens={sym_x: sh_x_train[batch_slice], sym_y: sh_y_train[batch_slice]},)

#test_model = theano.function([sym_batch_index], loss, acc,
 #                            givens={sym_x: sh_x_test[batch_slice],sym_y: sh_y_test[batch_slice]},)

test_model = theano.function(inputs = [sym_batch_index], outputs = [loss, cls_out],
                             givens={sym_x: sh_x_test[batch_slice], sym_y: sh_y_test[batch_slice]},)

#plot_results = theano.function([sym_batch_index], dec_x,
 #                              givens={sym_x: sh_x_test[batch_slice]},)

def train_epoch(lr):
    costs = []
    predictions= []
    for i in range(n_train_batches):
        cost_batch, output_batch = train_model(i, lr)
        #print("shape of cost_batch: ", np.shape(cost_batch))
        #print("shape of output_batch: ", np.shape(output_batch))
        #cost_batch = train_model(i, lr)
        prediction_batch = list(np.argmax(output_batch, axis = 1))
        #print("shape of prediction_batch: ", np.shape(prediction_batch))
        costs += [cost_batch]
        predictions+= prediction_batch
        #acc += [acc_batch]
    #print("shape of train predictions: ", np.shape(predictions))
    return (np.mean(costs),predictions)
    #return (np.mean(costs))


def test_epoch():
    costs = []
    predictions= []

    for i in range(n_test_batches):
        cost_batch, output_batch = test_model(i)
        prediction_batch = list(np.argmax(output_batch, axis = 1))
        #cost_batch = test_model(i)
        costs += [cost_batch]
        predictions+= prediction_batch

    return (np.mean(costs), predictions)
    #return (np.mean(costs))


# In[11]:
#get_output = theano.function([sym_batch_index], net_output)

read_model(l_dec_x, model_filename_read)

if do_train_model:
    # Training Loop
    for epoch in range(num_epochs):
        start = time.time()
        
        #shuffle train data, train model and test model
        s1 = np.arange(train_x.shape[0])
        np.random.shuffle(s1)
        sh_x_train.set_value(train_x[s1])
        sh_y_train.set_value(train_y_one_hot[s1])
        #sh_y_target_train.set_value(train_y[s])
        train_cost, predictions_train = train_epoch(lr)
        train_acc = np.sum(predictions_train==train_y[s1])/train_x.shape[0]
        
        #print("sh_y_target_train shape : ", np.shape(train_y[s1]))
        
        s2 = np.arange(test_x.shape[0])
        np.random.shuffle(s2)
        sh_x_test.set_value(test_x[s2])
        sh_y_test.set_value(test_y_one_hot[s2])
        
        test_cost, predictions_test = test_epoch()
        test_acc = np.sum(predictions_test==test_y[s2])/test_x.shape[0]
        
        t = time.time() - start
        
        line =  "*Epoch: %i\tTime: %0.2f\tLL Train: %0.3f\tLL test: %0.3f\tLL Train accuracy: %0.3f\tLL test accuracy: %0.3f\t" % ( epoch, t, train_cost, test_cost, train_acc, test_acc)
        #line =  "*Epoch: %i\tTime: %0.2f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, train_cost, test_cost)

        print (line)
    
    print ("Write model data")
    write_model(l_cls_output, model_filename_write)
else:
    read_model(l_cls_output, model_filename_read)



