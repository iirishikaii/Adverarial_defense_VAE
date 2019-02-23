import numpy as np
import pandas as pd
import lasagne
import theano
import theano.tensor as T
#from past import autotranslate
#autotranslate(['parmesan'])
#import parmesan
#from parmesan.distributions import log_bernoulli, kl_normal2_stdnormal
#from parmesan.layers import SimpleSampleLayer
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
import sys

theano.config.floatX = 'float32'


# In[5]:

def now():
    return mktime(gmtime())

#settings
do_train_model = False #False
batch_size = 100
latent_size = 100
nhidden = 512
lr = 0.001
num_epochs = 30 #50

#model_filename = "svhn_conv_ae"
#model_filename = "svhn_ae_adv_trained"
model_filename = sys.argv[1]
nplots = 5
nonlin = lasagne.nonlinearities.rectify

np.random.seed(1234) # reproducibility

#SYMBOLIC VARS
sym_x = T.tensor4()
sym_x_desired = T.tensor4()
sym_lr = T.scalar('lr')

### LOAD DATA
print ("Using SVHN dataset")

svhn_train = loadmat('train_32x32.mat')
svhn_test = loadmat('test_32x32.mat')

train_x = np.rollaxis(svhn_train['X'], 3).transpose(0,3,1,2).astype(theano.config.floatX)
test_x = np.rollaxis(svhn_test['X'], 3).transpose(0,3,1,2).astype(theano.config.floatX)

train_y = svhn_train['y'].flatten() - 1
test_y = svhn_test['y'].flatten() - 1

svhn_mean = 115.11177966923525
svhn_std = 50.819267906232888
train_x = (train_x - svhn_mean)/svhn_std
test_x = (test_x - svhn_mean)/svhn_std

n_train_batches = train_x.shape[0] / batch_size
n_test_batches = test_x.shape[0] / batch_size

#setup shared variables
sh_x_train = theano.shared(train_x, borrow=True)
sh_x_train_desired = theano.shared(train_x, borrow = True)
sh_x_test = theano.shared(test_x, borrow=True)
sh_x_test_desired = theano.shared(test_x, borrow = True)


print(len(train_x))


# In[5]:


### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, 3, 32, 32))
l_noise = lasagne.layers.BiasLayer(l_in, b = np.zeros((3,32,32), dtype = np.float32), shared_axes = 0, name = "NOISE")
l_noise.params[l_noise.b].remove("trainable")
l_enc_h1 = lasagne.layers.Conv2DLayer(l_noise, num_filters = 32, filter_size = 4, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'ENC_CONV1')
l_enc_h1 = lasagne.layers.Conv2DLayer(l_enc_h1, num_filters = 64, filter_size = 4, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'ENC_CONV2')
l_enc_h1 = lasagne.layers.Conv2DLayer(l_enc_h1, num_filters = 128, filter_size = 4, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'ENC_CONV3')
l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=512, nonlinearity=lasagne.nonlinearities.elu, name='ENC_DENSE2')

l_z = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='Z')


# In[6]:


### GENERATIVE MODEL p(x|z)
l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=512, nonlinearity=lasagne.nonlinearities.elu, name='DEC_DENSE1')
l_dec_h1 = lasagne.layers.ReshapeLayer(l_dec_h1, (batch_size, -1, 4, 4))
l_dec_h1 = lasagne.layers.TransposedConv2DLayer(l_dec_h1, num_filters = 128, crop="same",filter_size = 5, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'DEC_CONV1')
l_dec_h1 = lasagne.layers.TransposedConv2DLayer(l_dec_h1, num_filters = 64, crop="same",filter_size = 5, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'DEC_CONV2')
l_dec_h1 = lasagne.layers.TransposedConv2DLayer(l_dec_h1, num_filters = 32, filter_size = 5, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'DEC_CONV3')
l_dec_x = lasagne.layers.TransposedConv2DLayer(l_dec_h1, num_filters = 3,filter_size = 4, nonlinearity = lasagne.nonlinearities.identity, name = 'DEC_MU')
l_dec_x = lasagne.layers.ReshapeLayer(l_dec_x, (batch_size, -1))

# Get outputs from model
dec_x = lasagne.layers.get_output(l_dec_x, sym_x, deterministic=False)


# In[7]:


loss = lasagne.objectives.squared_error(dec_x, sym_x_desired.reshape((batch_size, -1)))
loss = lasagne.objectives.aggregate(loss, mode="mean")

params = lasagne.layers.get_all_params(l_dec_x, trainable=True)
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


# In[8]:


#Setup the theano functions
sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

train_model = theano.function([sym_batch_index, sym_lr], loss, updates=updates,
                                  givens={sym_x: sh_x_train[batch_slice],sym_x_desired: sh_x_train[batch_slice]},)

test_model = theano.function([sym_batch_index], loss,
                                  givens={sym_x: sh_x_test[batch_slice], sym_x_desired : sh_x_test[batch_slice]},)

plot_results = theano.function([sym_batch_index], dec_x,
                                  givens={sym_x: sh_x_test[batch_slice]},)

def train_epoch(lr):
    costs = []
    for i in range(int(n_train_batches)):
        cost_batch = train_model(i, lr)
        costs += [cost_batch]
    return np.mean(costs)


def test_epoch():
    costs = []
    for i in range(int(n_test_batches)):
        cost_batch = test_model(i)
        costs += [cost_batch]
    return np.mean(costs)

read_model(l_dec_x, model_filename)

l_z, reconstruction = lasagne.layers.get_output([l_z, l_dec_x], inputs = sym_x, deterministic=True)
# Adversarial confusion cost function
    
# Mean squared reconstruction difference
# KL divergence between latent variables
adv_z =  T.vector()
adv_confusion = lasagne.objectives.squared_error(adv_z, l_z).sum()

# Adversarial regularization
C = T.scalar()
adv_reg = C*lasagne.regularization.l2(l_noise.b)
# Total adversarial loss
adv_loss = adv_confusion + adv_reg
adv_grad = T.grad(adv_loss, l_noise.b)

# Function used to optimize the adversarial noise
adv_function = theano.function([sym_x, adv_z, C], [adv_loss, adv_grad])

# Helper to plot reconstructions    
adv_plot = theano.function([sym_x], reconstruction)

# Function to get latent variables of the target
adv_l_z = theano.function([sym_x], l_z)

def svhn_input(img):
    return np.tile(img, (batch_size, 1, 1, 1)).reshape(batch_size, 3, 32, 32)

def svhn_dist(img1, img2):
    img1_pixels = img1.flatten()*svhn_std + svhn_mean
    img2_pixels = img2.flatten()*svhn_std + svhn_mean
    return np.linalg.norm(img1_pixels - img2_pixels)

def adv_test(orig_img = 0, target_img = 1, C = 200.0, plot = True, iteration = 1):
    # Set the adversarial noise to zero
    l_noise.b.set_value(np.zeros((3,32,32)).astype(np.float32))
    
    # Get latent variables of the target
    adv_target_z = adv_l_z(svhn_input(test_x[target_img]))
    adv_target_z = adv_target_z[0]

    original_reconstruction = adv_plot(svhn_input(test_x[orig_img]))[0]
    target_reconstruction = adv_plot(svhn_input(test_x[target_img]))[0]

    orig_recon_dist = svhn_dist(original_reconstruction, test_x[orig_img])
    target_recon_dist = svhn_dist(target_reconstruction, test_x[target_img])
    orig_target_recon_dist = svhn_dist(original_reconstruction, test_x[target_img])
    target_orig_recon_dist = svhn_dist(target_reconstruction, test_x[orig_img])

    
    # Initialize the adversarial noise for the optimization procedure
    l_noise.b.set_value(np.random.uniform(-1e-8, 1e-8, size=(3,32,32)).astype(np.float32))
    
    # Optimization function for L-BFGS-B
    def fmin_func(x):
        l_noise.b.set_value(x.reshape(3, 32, 32).astype(np.float32))
        f, g = adv_function(svhn_input(test_x[orig_img]), adv_target_z, C)
        return float(f), g.flatten().astype(np.float64)
        
    # Noise bounds (pixels cannot exceed 0-1)
    bounds = list(zip(-svhn_mean/svhn_std-test_x[orig_img].flatten(), (255.0-svhn_mean)/svhn_std-test_x[orig_img].flatten()))
    
    # L-BFGS-B optimization to find adversarial noise
    x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value().flatten(), bounds = bounds, fprime = None, factr = 10, m = 25)
    
    adv_img = adv_plot(svhn_input(test_x[orig_img]))[0]
    
    orig_dist = svhn_dist(adv_img, test_x[orig_img])
    adv_dist = svhn_dist(adv_img, test_x[target_img])
    recon_dist = svhn_dist(adv_img, test_x[orig_img]+x.reshape(3, 32, 32))


    if plot:
        
        fig = plt.figure(figsize=(10,10))
        
        img = test_x[orig_img]
        i = 1
        title = "Original Image"
        img = img.copy().reshape(3, 32, 32).transpose(1,2,0)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        
        
        img = original_reconstruction
        i = 2
        title = "Original Reconstruction"
        img = img.copy().reshape(3, 32, 32).transpose(1,2,0)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        
        img = x
        i = 3
        title = "Adversarial noise"
        img = img.copy().reshape(3, 32, 32).transpose(1,2,0)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        

        img = test_x[target_img]
        i = 4
        title = "Target image"
        img = img.copy().reshape(3, 32, 32).transpose(1,2,0)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")


        img = test_x[orig_img].flatten()+x
        i = 5
        title = "Adversarial image"
        img = img.copy().reshape(3, 32, 32).transpose(1,2,0)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        

        img = adv_img
        i = 6
        title = "Adversarial reconstruction"
        img = img.copy().reshape(3, 32, 32).transpose(1,2,0)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

        #output_dir = '/Users/rishikaagarwal/Desktop/cs597/adv_vae-master/results/' + model_filename +'/'
        output_dir = 'results/compare_attacks/' + model_filename +'/'
        fig.savefig(os.path.join(output_dir, (str(iteration)+ '.png')))
        plt.close(fig)
 


    orig_target_dist = np.linalg.norm(test_x[orig_img] - test_x[target_img])

    returns = (np.linalg.norm(x),
               orig_dist, 
               adv_dist, 
               orig_recon_dist, 
               target_recon_dist, 
               recon_dist,
               orig_target_dist,
               orig_target_recon_dist,
               target_orig_recon_dist)
    
    return returns


# In[15]:


def orig_adv_dist(orig_img = None, target_img = None, plot = False, bestC = None, iteration = 1):
    if orig_img is None:
        orig_img = np.random.randint(0, len(test_x))
    if target_img is None:
        target_label = test_y[orig_img]
        while target_label == test_y[orig_img]:
            target_img = np.random.randint(0, len(test_x))
            target_label = test_y[target_img]
    
    noise_dist = []
    orig_dist=[]
    adv_dist=[]
    target_recon_dist=[]
    recon_dist=[]
    orig_target_dist=[]
    orig_target_recon_dist=[]
    target_orig_recon_dist=[]
    
    C = np.logspace(-5, 20, 50, base = 2, dtype = np.float32)
    
    for c in C:
        noise, od, ad, ore, tre, recd, otd, otrd, tord = adv_test(orig_img, target_img, C=c, plot = False, iteration = iteration)
        noise_dist.append(noise)
        orig_dist.append(od)
        adv_dist.append(ad)
        target_recon_dist.append(tre)
        recon_dist.append(recd)
        orig_target_dist.append(otd)
        orig_target_recon_dist.append(otrd)
        target_orig_recon_dist.append(tord)

    noise_dist = np.array(noise_dist)
    orig_dist = np.array(orig_dist)
    adv_dist = np.array(adv_dist)
    target_recon_dist = np.array(target_recon_dist)
    recon_dist = np.array(recon_dist)
    orig_target_dist = np.array(orig_target_dist)
    orig_target_recon_dist = np.array(orig_target_recon_dist)
    target_orig_recon_dist = np.array(target_orig_recon_dist)
    
    if bestC is None:
        bestC = C[np.argmax(adv_dist - orig_dist >= 0)-1]

    print (orig_img, target_img, bestC)

    best_noise, best_orig_dist, best_adv_dist, orig_reconstruction_dist, target_reconstruction_dist, _, _, _, _ = adv_test(orig_img, target_img, C=bestC, plot = plot, iteration = iteration)

    plt.ioff()
    if plot:
        fig = plt.figure()
        plt.axhline(y=target_reconstruction_dist, linewidth = 2, color = 'cyan', label = "Dist(Target reconstruction - Target)")
        plt.axvline(x=orig_reconstruction_dist, linewidth = 2, color='DarkOrange', label = "Dist(Original reconstruction - Original)")
        plt.scatter(orig_dist, adv_dist)
        plt.scatter([best_orig_dist], [best_adv_dist], color = "red")
        plt.xlabel("Dist(reconstructed Adversarial image - Original image)")
        plt.ylabel("Dist(reconstructed Adversarial image - Target image)")
        plt.legend()
        plt.plot()
        #output_dir = '/Users/rishikaagarwal/Desktop/cs597/adv_vae-master/results/' + model_filename + '/'
        output_dir = 'results/compare_attacks/' + model_filename + '/'
        #os.path.join(output_dir, {}/exp_'+ str(iteration)+ '.png')
        fig.savefig(os.path.join(output_dir, ('exp_'+ str(iteration)+ 'graph1.png')))
        plt.close(fig)
        
        fig = plt.figure()
        plt.axhline(y=target_reconstruction_dist, linewidth = 2, color = 'cyan', label = "Dist(Target reconstruction - Target)")
        plt.axvline(x=np.linalg.norm(test_x[orig_img]-test_x[target_img]), linewidth = 2, color='DarkOrange', label = "Dist(Original - Target)")
        plt.scatter(noise_dist, adv_dist)
        plt.scatter([best_noise], [best_adv_dist], color = "red")
        plt.ylabel("Dist(reconstructed Adversarial image - Target image)")
        plt.xlabel("Dist(noise)")
        plt.legend()
        fig.savefig(os.path.join(output_dir, ('exp_'+ str(iteration)+ 'graph2.png')))
        plt.plot()
        plt.close(fig)
    
        bestCind = np.where(C==bestC)
        print('orig_img : ',orig_img)
        print('target_img : ', target_img)
        print('orig_reconstruction_dist : ', orig_reconstruction_dist)
        print('target_reconstruction_dist : ',target_reconstruction_dist)
        print('original_target_dist : ', orig_target_dist[bestCind])
        print('orig_target_recon_dist : ', orig_target_recon_dist[bestCind])
        print('target_orig_recon_dist : ', target_orig_recon_dist[bestCind])

        print()
        print('bestC : ', bestC)
        print('adv_adv_recon_dist : ', recon_dist[bestCind])
        print('best noise_dist :  ', noise_dist[bestCind])
        print('best orig_dist :  ', orig_dist[bestCind])
        print('best adv_dist : ', adv_dist[bestCind])
        print()
        
    df = pd.DataFrame({'orig_img': orig_img,
                       'target_img': target_img,
                       'bestC': bestC,
                       'orig_reconstruction_dist': orig_reconstruction_dist,
                       'target_reconstruction_dist': target_reconstruction_dist,
                       'noise_dist': noise_dist,
                       'orig_dist': orig_dist,
                       'adv_dist': adv_dist,
                       'target_recon_dist': target_recon_dist,
                       'recon_dist': recon_dist,
                       'orig_target_dist': orig_target_dist,
                       'orig_target_recon_dist': orig_target_recon_dist,
                       'target_orig_recon_dist': target_orig_recon_dist,
                       'C': C})
    
    return df


# In[17]:


n = 15

#or_im = random.sample(range(1,len(test_x)), n)
#targ_im = random.sample(range(1,len(test_x)), n)

or_im =  [18419, 11631, 15937, 2169, 2999, 24821, 12772, 3888, 108, 22673, 23116, 7532, 24561, 10773, 9047]
targ_im = [7093, 12750, 19994, 3682, 16498, 7761, 14888, 3582, 24656, 22511, 5337, 22140, 14854, 10762, 10657]

print("original image list: ", or_im)
print("target image list: ", targ_im)

for i in range(n):
    while(test_y[or_im[i]]==test_y[targ_im[i]]):
        targ_im[i] = targ_im[i]+1

for i in range(n):
    start_time = time.time()
    df = orig_adv_dist(orig_img = or_im[i], target_img = targ_im[i], plot = True, iteration = i)
    print ("Iter", i, "Time", time.time() - start_time, "sec")
    print("############################################################")




