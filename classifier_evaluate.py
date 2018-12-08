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
import pickle

theano.config.floatX = 'float32'



#settings
do_train_model = False #False

latent_size = 20
nhidden = 512
lr = 0.001
num_epochs = 20 #50
model_filename_read = "mnist_ae"
classifier_filename_read = "mnist_classifier"
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

#load adversarial examples
'''
adv_train_x = []
orig_train_x = []
adv_img_num_train = 5000
for img_num in range(0,adv_img_num_train):
    fadv = os.path.join('dataset/train/adversarial_images',"img_"+str(img_num)+".png")
    forig = os.path.join('dataset/train/original_images',"img_"+str(img_num)+".png")
    adv_train = Image.open(fadv)
    orig_train = Image.open(forig)
    adv_train.load()
    orig_train.load()
    adv_train_x.append(np.asarray(adv_train, dtype="float32"))
    orig_train_x.append(np.asarray(orig_train, dtype="float32"))
#train_y = train_y.astype(np.float32)
#test_y = test_y.astype(np.float32)
file_path_adv = 'dataset/train/adversarial_images/'
file_path_orig = 'dataset/train/original_images/'
f1 = file_path_adv + "true_label.p"
f2 = file_path_adv + "target_label.p"
with open(f1, 'rb') as f:
    adv_train_y_orig = pickle.load(f)
f.close()
with open(f2, 'rb') as f:
    adv_train_y_target = pickle.load(f)
f.close()

#convert the lists into np arrays

adv_train_x = np.asarray(adv_train_x)
orig_train_x = np.asarray(orig_train_x)
adv_train_y_orig = np.asarray(adv_train_y_orig)
adv_train_y_target = np.asarray(adv_train_y_target)
adv_train_x = adv_train_x[:,:,:,0]
orig_train_x = orig_train_x[:,:,:,0]


adv_train_x = (adv_train_x.reshape((-1,784))/255.0).astype(np.float32)
orig_train_x = (orig_train_x.reshape((-1,784))/255.0).astype(np.float32)

adv_train_y_one_hot_orig = np.zeros((len(adv_train_y_orig),10))
for i in range(0,len(adv_train_y_orig)):
    adv_train_y_one_hot_orig[i,adv_train_y_orig[i]] = 1

adv_train_y_one_hot_target = np.zeros((len(adv_train_y_target),10))
for i in range(0,len(adv_train_y_target)):
    adv_train_y_one_hot_target[i,adv_train_y_target[i]] = 1

adv_train_y_one_hot_target = adv_train_y_one_hot_target.astype(np.float32)
adv_train_y_one_hot_orig = adv_train_y_one_hot_orig.astype(np.float32)


adv_train_x[adv_train_x > 0.5] = 1.0
adv_train_x[adv_train_x <= 0.5] = 0.0
orig_train_x[orig_train_x > 0.5] = 1.0
orig_train_x[orig_train_x <= 0.5] = 0.0

'''
file_path_adv = 'dataset/test/adversarial_images/'
file_path_orig = 'dataset/test/original_images/'
adv_test_x = []
orig_test_x = []

adv_img_num_test = 500
for img_num in range(0,adv_img_num_test):
    fadv = os.path.join('dataset/test/adversarial_images','img_'+str(img_num)+'.png')
    forig = os.path.join('dataset/test/original_images','img_'+str(img_num)+'.png')
    adv_test = Image.open(fadv)
    orig_test = Image.open(forig)
    adv_test.load()
    orig_test.load()
    adv_test_x.append(np.asarray(adv_test, dtype="float32"))
    orig_test_x.append(np.asarray(orig_test, dtype="float32"))
#train_y = train_y.astype(np.float32)
#test_y = test_y.astype(np.float32)
f1 = file_path_adv + "true_label.p"
f2 = file_path_adv + "target_label.p"
with open(f1, 'rb') as f:
    adv_test_y_orig = pickle.load(f)
f.close()
with open(f2, 'rb') as f:
    adv_test_y_target = pickle.load(f)
f.close()

adv_test_x = np.asarray(adv_test_x)
orig_test_x = np.asarray(orig_test_x)
adv_test_y_orig = np.asarray(adv_test_y_orig)
adv_test_y_target = np.asarray(adv_test_y_target)
adv_test_x = adv_test_x[:,:,:,0]
orig_test_x = orig_test_x[:,:,:,0]
adv_test_x = (adv_test_x.reshape((-1,784))/255.0).astype(np.float32)
orig_test_x = (orig_test_x.reshape((-1,784))/255.0).astype(np.float32)

#compare the accuracy of the classifier on (orig_train_x, adv_train_y_orig) and (adv_train_x, adv_train_y_adv)
#also check what are the predictions of the classifier. are they the target labels?


adv_test_y_one_hot_orig = np.zeros((len(adv_test_y_orig),10))
for i in range(0,len(adv_test_y_orig)):
    adv_test_y_one_hot_orig[i,adv_test_y_orig[i]] = 1

adv_test_y_one_hot_target = np.zeros((len(adv_test_y_target),10))
for i in range(0,len(adv_test_y_target)):
    adv_test_y_one_hot_target[i,adv_test_y_target[i]] = 1

adv_test_y_one_hot_target = adv_test_y_one_hot_target.astype(np.float32)
adv_test_y_one_hot_orig = adv_test_y_one_hot_orig.astype(np.float32)

adv_test_x[adv_test_x > 0.5] = 1.0
adv_test_x[adv_test_x <= 0.5] = 0.0
orig_test_x[orig_test_x > 0.5] = 1.0
orig_test_x[orig_test_x <= 0.5] = 0.0

print("===========================")
print("shape of adv_test_x: ", np.shape(adv_test_x))
print("shape of adv_test_y_target: ", np.shape(adv_test_y_target))
print("shape of adv_test_y_one_hot_target: ", np.shape(adv_test_y_one_hot_target))
print("===========================")
#setup shared variables
#sh_x_train = theano.shared(adv_train_x, borrow=True)
sh_x_test = theano.shared(adv_test_x, borrow=True)
#sh_y_train = theano.shared(adv_train_y_one_hot_orig, borrow=True)
sh_y_test = theano.shared(adv_test_y_one_hot_orig, borrow=True)

nfeatures=adv_test_x.shape[1]
n_test_batches = 1
batch_size = adv_test_x.shape[0]
print("n_test_batches: ", n_test_batches)
print("batch_size: ", batch_size)


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
#print('shape of train_x: ', np.shape(train_x))
#print('shape of train_y: ', np.shape(train_y))
#print('shape of train_y_one_hot: ', np.shape(train_y_one_hot))

sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

test_model = theano.function(inputs = [sym_batch_index], outputs = [loss, cls_out],
                             givens={sym_x: sh_x_test[batch_slice], sym_y: sh_y_test[batch_slice]},)

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
   

read_model(l_dec_x, model_filename_read)

read_model(l_cls_output, classifier_filename_read)

sh_y_test.set_value(adv_test_y_one_hot_orig)
start = time.time()
test_cost, predictions_test = test_epoch()
test_acc = np.sum(predictions_test==adv_test_y_orig)/adv_test_x.shape[0]
        
t = time.time() - start
line =  "Time: %0.2f\tLL test: %0.3f\tLL test accuracy: %0.3f\t" % ( t, test_cost, test_acc)
        #line =  "*Epoch: %i\tTime: %0.2f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, train_cost, test_cost)
print(line)




