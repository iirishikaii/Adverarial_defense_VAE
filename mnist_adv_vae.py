
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:25:13 2016

@author: tabacof
"""

# Implements a variational autoencoder as described in Kingma et al. 2013
# "Auto-Encoding Variational Bayes"
#from __future__ import print_function
import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import numpy as np
import pandas as pd
import lasagne
from parmesan.distributions import log_bernoulli, kl_normal2_stdnormal
from parmesan.layers import SimpleSampleLayer
from keras.datasets import mnist
import time, shutil, os
import scipy
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
from read_write_model import read_model, write_model
from sklearn.metrics import auc
from time import mktime, gmtime
from PIL import Image
import pickle

# In[2]:
num_test_attacks = 20
num_adv_train = 500
num_adv_test = 50

def now():
    return mktime(gmtime())

#settings
do_train_model = False
batch_size = 100
latent_size = 20
nhidden = 512
lr = 0.001
num_epochs = 50
model_filename = "mnist_vae"
nonlin = lasagne.nonlinearities.rectify

np.random.seed(1234) # reproducibility


# In[3]:


#SYMBOLIC VARS
sym_x = T.matrix()
sym_x_desired = T.matrix()
sym_lr = T.scalar('lr')

### LOAD DATA
print ("Using MNIST dataset")

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = (train_x.reshape((-1, 784))/255.0).astype(np.float32)
test_x = (test_x.reshape((-1, 784))/255.0).astype(np.float32)

train_x[train_x > 0.5] = 1.0
train_x[train_x <= 0.5] = 0.0

test_x[test_x > 0.5] = 1.0
test_x[test_x <= 0.5] = 0.0

#setup shared variables
sh_x_train = theano.shared(train_x, borrow=True)
sh_x_test = theano.shared(test_x, borrow=True)
sh_x_desired_train = theano.shared(train_x, borrow=True)
sh_x_desired_test = theano.shared(test_x, borrow=True)

nfeatures=train_x.shape[1]
n_train_batches = int(train_x.shape[0] / batch_size)
n_test_batches = int(test_x.shape[0] / batch_size)


# In[4]:


### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, nfeatures))
l_noise = lasagne.layers.BiasLayer(l_in, b = np.zeros(nfeatures, dtype = np.float32), name = "NOISE")
l_noise.params[l_noise.b].remove("trainable")
l_enc_h1 = lasagne.layers.DenseLayer(l_noise, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE1')
l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE2')

l_mu = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_MU')
l_log_var = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_LOG_VAR')

#sample the latent variables using mu(x) and log(sigma^2(x))
l_z = SimpleSampleLayer(mean=l_mu, log_var=l_log_var)


# In[5]:


### GENERATIVE MODEL p(x|z)
l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE2')
l_dec_h1 = lasagne.layers.DenseLayer(l_dec_h1, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE1')
l_dec_x_mu = lasagne.layers.DenseLayer(l_dec_h1, num_units=nfeatures, nonlinearity=lasagne.nonlinearities.sigmoid, name='DEC_X_MU')

# Get outputs from model
# with noise
z_train, z_mu_train, z_log_var_train, x_mu_train = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=False
)

# without noise
z_eval, z_mu_eval, z_log_var_eval, x_mu_eval = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=True
)


# In[6]:


#Calculate the loglikelihood(x) = E_q[ log p(x|z) + log p(z) - log q(z|x)]
def ELBO(z, z_mu, z_log_var, x_mu, x):
    """
    Latent z       : gaussian with standard normal prior
    decoder output : bernoulli

    When the output is bernoulli then the output from the decoder
    should be sigmoid. The sizes of the inputs are
    z: (batch_size, num_latent)
    z_mu: (batch_size, num_latent)
    z_log_var: (batch_size, num_latent)
    x_mu: (batch_size, num_features)
    x: (batch_size, num_features)
    """
    kl_term = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis=1)
    log_px_given_z = log_bernoulli(x, x_mu, eps=1e-6).sum(axis=1)
    LL = T.mean(-kl_term + log_px_given_z)

    return LL

# TRAINING LogLikelihood
LL_train = ELBO(z_train, z_mu_train, z_log_var_train, x_mu_train, sym_x_desired)
#LL_train = ELBO(z_train, z_mu_train, z_log_var_train, x_mu_train, sym_x)


# EVAL LogLikelihood
LL_eval = ELBO(z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, sym_x_desired)
#LL_eval = ELBO(z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, sym_x)
params = lasagne.layers.get_all_params(l_dec_x_mu, trainable=True)
for p in params:
    print (p, p.get_value().shape)

### Take gradient of Negative LogLikelihood
grads = T.grad(-LL_train, params)

# Add gradclipping to reduce the effects of exploding gradients.
# This speeds up convergence
clip_grad = 1
max_norm = 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g,-clip_grad, clip_grad) for g in mgrads]


# In[7]:


#Setup the theano functions
sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

train_model = theano.function([sym_batch_index, sym_lr], LL_train, updates=updates,
                                  givens={sym_x: sh_x_train[batch_slice],sym_x_desired: sh_x_desired_train[batch_slice]},)

test_model = theano.function([sym_batch_index], LL_eval,
                                  givens={sym_x: sh_x_test[batch_slice],sym_x_desired: sh_x_desired_test[batch_slice]},)

#train_model = theano.function([sym_batch_index, sym_lr], LL_train, updates=updates,
 #                                 givens={sym_x: sh_x_train[batch_slice]},)

#test_model = theano.function([sym_batch_index], LL_eval,
 #                                 givens={sym_x: sh_x_test[batch_slice]},)



plot_results = theano.function([sym_batch_index], x_mu_eval,
                                  givens={sym_x: sh_x_test[batch_slice]},)

def train_epoch(lr):
    costs = []
    for i in range(n_train_batches):
        cost_batch = train_model(i, lr)
        costs += [cost_batch]
    return np.mean(costs)


def test_epoch():
    costs = []
    for i in range(n_test_batches):
        cost_batch = test_model(i)
        costs += [cost_batch]
    return np.mean(costs)


# In[8]:


def metrics_auc(points, limits):
    (noise_dist, adv_dist) = points
    (ex_orig_target_dist, ex_orig_target_recon_dist,
     target_reconstruction_dist) = limits

    max_noise = max(noise_dist)
    min_dist = min(adv_dist)

    noise_dist += (max_noise,)
    adv_dist += (min_dist,)

    noise_dist += (ex_orig_target_dist,)
    adv_dist += (min_dist,)

    return auc(noise_dist, adv_dist)


# In[9]:


if do_train_model:
    # Training Loop
    for epoch in range(num_epochs):
        start = time.time()

        #shuffle train data, train model and test model
        
        sh_x_train.set_value(train_x)
        sh_x_desired_train.set_value(train_x)
        sh_x_test.set_value(test_x)
        sh_x_desired_test.set_value(test_x)
        train_cost = train_epoch(lr)
        test_cost = test_epoch()

        t = time.time() - start

        line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
        print (line)
    
    print ("Write model data")
    write_model([l_dec_x_mu], model_filename)
else:
    read_model([l_dec_x_mu], model_filename)
    


# In[10]:


def show_mnist(img, i, title=""): # expects flattened image of shape (3072,) 
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")
    
def mnist_input(img):
    return np.tile(img, (batch_size, 1, 1, 1)).reshape(batch_size, 784)

def mnist_dist(imgs, img):
    assert imgs.shape == (batch_size, 784)
    diff = np.linalg.norm(imgs - img, axis = 1)
    return np.mean(diff), np.std(diff)


# In[11]:


def kld(mean1, log_var1, mean2, log_var2):
    mean_term = (T.exp(0.5*log_var1) + (mean1-mean2)**2.0)/T.exp(0.5*log_var2)
    return mean_term + log_var2 - log_var1 - 0.5

# Autoencoder outputs
mean, log_var, reconstruction = lasagne.layers.get_output(
    [l_mu, l_log_var, l_dec_x_mu], inputs = sym_x, deterministic=True)
    
# Adversarial confusion cost function
    
# Mean squared reconstruction difference
# KL divergence between latent variables
adv_mean =  T.vector()
adv_log_var = T.vector()
adv_confusion = kld(mean, log_var, adv_mean, adv_log_var).sum()

# Adversarial regularization
C = T.scalar()
adv_reg = C*lasagne.regularization.l2(l_noise.b)
# Total adversarial loss
adv_loss = adv_confusion + adv_reg
adv_grad = T.grad(adv_loss, l_noise.b)

# Function used to optimize the adversarial noise
adv_function = theano.function([sym_x, adv_mean, adv_log_var, C], [adv_loss, adv_grad])

# Helper to plot reconstructions    
adv_plot = theano.function([sym_x], reconstruction)

# Function to get latent variables of the target
adv_mean_log_var = theano.function([sym_x], [mean, log_var])


# In[12]:


def adv_test(orig_img, target_img, C, plot = True):
    # Set the adversarial noise to zero
    l_noise.b.set_value(np.zeros((784,)).astype(np.float32))
    
    # Get latent variables of the target
    adv_mean_values, adv_log_var_values = adv_mean_log_var(mnist_input(test_x[target_img]))
    adv_mean_values = adv_mean_values[0]
    adv_log_var_values = adv_log_var_values[0]

    original_reconstructions = adv_plot(mnist_input(test_x[orig_img]))
    target_reconstructions = adv_plot(mnist_input(test_x[target_img]))

    orig_recon_dist, orig_recon_dist_std = mnist_dist(original_reconstructions, test_x[orig_img])
    target_recon_dist, target_recon_dist_std = mnist_dist(target_reconstructions, test_x[target_img])
    orig_target_recon_dist, orig_target_recon_dist_std = mnist_dist(original_reconstructions, test_x[target_img])
    target_orig_recon_dist, target_orig_recon_dist_std = mnist_dist(target_reconstructions, test_x[orig_img])

    # Plot original reconstruction    
    if plot:
        plt.figure()
        show_mnist(test_x[orig_img], 1, "Original")
        show_mnist(original_reconstructions[0], 2, "Original rec.")

    # Initialize the adversarial noise for the optimization procedure
    l_noise.b.set_value(np.random.uniform(-1e-8, 1e-8, size=(784,)).astype(np.float32))
    
    # Optimization function for L-BFGS-B
    def fmin_func(x):
        l_noise.b.set_value(x.astype(np.float32))
        f, g = adv_function(mnist_input(test_x[orig_img]), adv_mean_values, adv_log_var_values, C)
        return float(f), g.flatten().astype(np.float64)
        
    # Noise bounds (pixels cannot exceed 0-1)
    bounds = list(zip(-test_x[orig_img], 1-test_x[orig_img]))
    
    # L-BFGS-B optimization to find adversarial noise
    x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value().flatten(), bounds = bounds, fprime = None, factr = 10, m = 25)
    
    adv_imgs = adv_plot(mnist_input(test_x[orig_img]))
    
    orig_dist, orig_dist_std = mnist_dist(adv_imgs, test_x[orig_img])
    adv_dist, adv_dist_std = mnist_dist(adv_imgs, test_x[target_img])
    recon_dist, recon_dist_std = mnist_dist(adv_imgs, test_x[orig_img]+x)
    
    # Plotting results
    if plot:
        fig = plt.figure(figsize=(10,10))
        
        img = test_x[orig_img]
        i = 1
        title = "Original Image"
        img = img.copy().reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")
        #show_mnist(test_x[orig_img], 1, "Original image")
        
        img = original_reconstructions[0]
        i = 2
        title = "Original Reconstruction"
        img = img.copy().reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")
        
        img = x
        i = 3
        title = "Adversarial noise"
        img = img.copy().reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")
        
        #show_mnist(x, 3, "Adversarial noise")

        img = test_x[target_img]
        i = 4
        title = "Target image"
        img = img.copy().reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")

        #show_mnist(test_x[target_img], 4, "Target image")

        #img = test_x[orig_img].flatten()+x
        img = test_x[orig_img].flatten() + x
        i = 5
        title = "Adversarial image"
        img = img.copy().reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")
        #show_mnist((test_x[orig_img].flatten()+x), 5, "Adversarial image")
        #show_mnist(test_x_app[target_img], 4, "Target image")
        #show_mnist((test_x_app[orig_img].flatten()+x), 5, "Adversarial image")

        img = adv_imgs[0]
        i = 6
        title = "Adversarial reconstruction"
        img = img.copy().reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(3, 2, i)
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")

        #output_dir = '/Users/rishikaagarwal/Desktop/cs597/adv_vae-master/results/' + model_filename +'/'
        output_dir = 'results/' + model_filename +'/'
        fig.savefig(os.path.join(output_dir, ('after_attack_' + str(now())+ '.png')))
        plt.close(fig)
            
    orig_target_dist = np.linalg.norm(test_x[orig_img] - test_x[target_img])
    
    returns = (np.linalg.norm(x),
               orig_dist, 
               orig_dist_std, 
               adv_dist, 
               adv_dist_std, 
               orig_recon_dist, 
               orig_recon_dist_std, 
               target_recon_dist, 
               target_recon_dist_std,
               recon_dist,
               recon_dist_std,
               orig_target_dist,
               orig_target_recon_dist,
               orig_target_recon_dist_std,
               target_orig_recon_dist,
               target_orig_recon_dist_std)
    
    return returns


# In[13]:


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
    orig_dist_std=[]
    adv_dist=[]
    adv_dist_std=[]
    target_recon_dist=[]
    target_recon_dist_std=[]
    recon_dist=[]
    recon_dist_std=[]
    orig_target_dist=[]
    orig_target_recon_dist=[]
    orig_target_recon_dist_std=[]
    target_orig_recon_dist=[]
    target_orig_recon_dist_std=[]
    
    C = np.logspace(-20, 20, 100, base = 2, dtype = np.float32)
    
    for c in C:
        noise, od, ods, ad, ads, ore, ores, tre, tres, recd, recs, otd, otrd, otrds, tord, tords = adv_test(orig_img, target_img, C=c, plot = False)
        noise_dist.append(noise)
        orig_dist.append(od)
        orig_dist_std.append(ods)
        adv_dist.append(ad)
        adv_dist_std.append(ads)
        target_recon_dist.append(tre)
        target_recon_dist_std.append(tres)
        recon_dist.append(recd)
        recon_dist_std.append(recs)
        orig_target_dist.append(otd)
        orig_target_recon_dist.append(otrd)
        orig_target_recon_dist_std.append(otrds)
        target_orig_recon_dist.append(tord)
        target_orig_recon_dist_std.append(tords)

    noise_dist = np.array(noise_dist)
    orig_dist = np.array(orig_dist)
    orig_dist_std = np.array(orig_dist_std)
    adv_dist = np.array(adv_dist)
    adv_dist_std = np.array(adv_dist_std)
    target_recon_dist = np.array(target_recon_dist)
    target_recon_dist_std = np.array(target_recon_dist_std)
    recon_dist = np.array(recon_dist)
    recon_dist_std = np.array(recon_dist_std)
    orig_target_dist = np.array(orig_target_dist)
    orig_target_recon_dist = np.array(orig_target_recon_dist)
    orig_target_recon_dist_std = np.array(orig_target_recon_dist_std)
    target_orig_recon_dist = np.array(target_orig_recon_dist)
    target_orig_recon_dist_std = np.array(target_orig_recon_dist_std)
    
    if bestC is None:
        bestC = C[np.argmax(adv_dist - orig_dist >= 0)-1]

    print (orig_img, target_img, bestC)

    ex_noise, _, _, ex_adv_dist, _, orig_reconstruction_dist, _, target_reconstruction_dist, _, _, _, ex_orig_target_dist, ex_orig_target_recon_dist, _, _, _ = adv_test(orig_img, target_img, C=bestC, plot = plot)
    
    plt.ioff()
    output_dir = 'results/' + model_filename + '/'
    if plot:
        fig = plt.figure()
        plt.axvline(x=ex_orig_target_dist, linewidth = 2, color='cyan', label = "Original - Target")
        plt.axhline(y=ex_orig_target_recon_dist, linewidth = 2, color = 'DarkOrange', label = "Original rec. - Target")
        plt.axhline(y=target_reconstruction_dist, linewidth = 2, color = 'red', label = "Target rec. - Target")
        plt.scatter(noise_dist, adv_dist)
        plt.scatter([ex_noise], [ex_adv_dist], color = "red")
        plt.ylabel("Dist(reconstructed Adversarial image - Target image)")
        plt.xlabel("Dist(noise)")
        plt.legend()
        fig.savefig(os.path.join(output_dir, ('exp_'+ str(iteration)+ 'graph1.png')))
        plt.plot()
        plt.close(fig)
        
        
        xs = noise_dist/ orig_target_dist
        zs = (
            (adv_dist- target_recon_dist) /
            (orig_target_recon_dist - target_recon_dist)
        )
        idx = np.argsort(xs)
        xs = xs[idx]
        zs = zs[idx]
        (xs, zs) = zip(*[(x, z) for (x, z)
                               in zip(xs, zs) if x >= 0 and x <= 1])
        points = (xs, zs)
        limits = (1.0, 1.0, 0)
        AUDDC = metrics_auc(points, limits)

        print("AUDDC: ", AUDDC)

        
    df = pd.DataFrame({'orig_img': orig_img,
                       'target_img': target_img,
                       'bestC': bestC,
                       'best_noise' : ex_noise,
                       'orig_reconstruction_dist': orig_reconstruction_dist,
                       'target_reconstruction_dist': target_reconstruction_dist,
                       'noise_dist': noise_dist,
                       'orig_dist': orig_dist,
                       'orig_dist_std': orig_dist_std,
                       'adv_dist': adv_dist,
                       'adv_dist_std': adv_dist_std,
                       'target_recon_dist': target_recon_dist,
                       'target_recon_dist_std': target_recon_dist_std,
                       'recon_dist': recon_dist,
                       'recon_dist_std': recon_dist_std,
                       'orig_target_dist': orig_target_dist,
                       'orig_target_recon_dist': orig_target_recon_dist,
                       'orig_target_recon_dist_std': orig_target_recon_dist_std,
                       'target_orig_recon_dist': target_orig_recon_dist,
                       'target_orig_recon_dist_std': target_orig_recon_dist_std,
                       'C': C,
                       'AUDDC': AUDDC})
    
    return df


# In[67]:


n = 5
auddc_list = []
time_taken = []
bn = []

for i in range(n):
    start_time = time.time()
    df = orig_adv_dist(plot = True, iteration = i)
    end_time = time.time()
    AUDDC = df.at[0,'AUDDC']
    best_noise = df.at[0,'best_noise']
    auddc_list.append(AUDDC)
    bn.append(best_noise)
    time_taken.append(end_time-start_time)
    print ("Iter", i, "Time", end_time - start_time, "sec")
    #df.to_csv("results/" + model_filename + "/exp_" + str(i) + ".csv")
    
print("Average AUDDC: ", sum(auddc_list)/len(auddc_list))
print("Average time taken for attack: ", sum(time_taken)/len(time_taken))
print("Average noise added: ", sum(bn)/len(bn))

# In[14]:


#df = orig_adv_dist(9791, 3405, plot = True, bestC = 50)


# In[15]:


def gen_adv_ex(orig_img, target_img, C):
    
   
    # Set the adversarial noise to zero
    l_noise.b.set_value(np.zeros((784,)).astype(np.float32))
    
    # Get latent variables of the target
    adv_mean_values, adv_log_var_values = adv_mean_log_var(mnist_input(train_x[target_img]))
    adv_mean_values = adv_mean_values[0]
    adv_log_var_values = adv_log_var_values[0]
    
    # Initialize the adversarial noise for the optimization procedure
    l_noise.b.set_value(np.random.uniform(-1e-8, 1e-8, size=(784,)).astype(np.float32))
    
    # Optimization function for L-BFGS-B
    def fmin_func(x):
        l_noise.b.set_value(x.astype(np.float32))
        f, g = adv_function(mnist_input(train_x[orig_img]), adv_mean_values, adv_log_var_values, C)
        return float(f), g.flatten().astype(np.float64)
        
    # Noise bounds (pixels cannot exceed 0-1)
    bounds = list(zip(-train_x[orig_img], 1-train_x[orig_img]))
    
    # L-BFGS-B optimization to find adversarial noise
    
    x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value().flatten(), bounds = bounds, fprime = None, factr = 10, m = 25)
    
    adv_imgs = adv_plot(mnist_input(train_x[orig_img]))
    
    
    orig_dist, orig_dist_std = mnist_dist(adv_imgs, train_x[orig_img])
    adv_dist, adv_dist_std = mnist_dist(adv_imgs, train_x[target_img])
    recon_dist, recon_dist_std = mnist_dist(adv_imgs, train_x[orig_img]+x)
    
    
    
    returns = (np.linalg.norm(x),
                x,
               orig_dist,
               adv_dist)
               
    return returns


# In[16]:


def gen_adv_ex_set(N, train_set):
    #N = 400
    or_ex_x = []
    adv_ex_x = []
    adv_ex_y_target = []
    adv_ex_y_true = []
    
    if(train_set==True):
        file_path_adv = 'dataset/mnist_vae/train/adversarial_images/'
        file_path_orig = 'dataset/mnist_vae/train/original_images/'
    else:
        file_path_adv = 'dataset/mnist_vae/test/adversarial_images/'
        file_path_orig = 'dataset/mnist_vae/test/original_images/'

    img_num = 0
    
    for i in range(N):
        if(i%10==0):
            print("generating ",i,"th adversarial example")


        orig_img = np.random.randint(0, len(train_x))
        
        target_label = train_y[orig_img]
        while target_label == train_y[orig_img]:
            target_img = np.random.randint(0, len(train_x))
            target_label = train_y[target_img]
        
        noise_dist = []
        orig_dist = []
        adv_dist = []
        
        C = np.logspace(-5, 20, 25, base = 2, dtype = np.float32)
        for c in C:
            noise, noise_matrix, od, ad = gen_adv_ex(orig_img, target_img, C = c)
            noise_dist.append(noise)
            orig_dist.append(od)
            adv_dist.append(ad)

        noise_dist = np.array(noise_dist)
        orig_dist = np.array(orig_dist)
        adv_dist = np.array(adv_dist)


        bestC = C[np.argmax(adv_dist - orig_dist >= 0)-1]

        best_noise, best_noise_matrix, best_orig_dist, best_adv_dist= gen_adv_ex(orig_img, target_img, C=bestC)
        or_ex_x.append(train_x[orig_img])
        adv_ex_x.append((train_x[orig_img] + best_noise_matrix))
        adv_ex_y_target.append(train_y[target_img])
        adv_ex_y_true.append(train_y[orig_img])
        
        #print("orig_im.shape: ", np.shape(train_x[orig_img]))
        #print("best noise.shape: ", np.shape(best_noise))

        #save adversarial images
        
        
        adv_im = train_x[orig_img] + best_noise_matrix
        adv_im = np.clip(adv_im, 0, 1)
        adv_im = (adv_im *255.0).astype('uint8')
        orig_im = train_x[orig_img]
        orig_im = np.clip(orig_im, 0, 1)
        orig_im = (orig_im * 255.0).astype('uint8')
        #print('adv image: ', np.reshape(adv_im,(28,28)))
        adv_im = Image.fromarray(np.reshape(adv_im, (28,28)))
        orig_im = Image.fromarray(np.reshape(orig_im, (28,28)))
        
        #adv_im = adv_im.convert('RGB')
        #orig_im = orig_im.convert('RGB')
        
        adv_im.save(os.path.join(file_path_adv,('img_'+str(img_num)+'.png')))
        orig_im.save(os.path.join(file_path_orig,('img_'+str(img_num)+'.png')))

        img_num+=1
        '''
        '''
        adv_im = train_x[orig_img] + best_noise_matrix
        adv_im = np.reshape(adv_im, (28,28))
        adv_im = np.clip(adv_im, 0, 1)
        matplotlib.pyplot.imsave(os.path.join(file_path_adv,('img_'+str(img_num)+'.png')), adv_im, cmap = 'Greys_r')
        orig_im = train_x[orig_img]
        orig_im = np.clip(orig_im, 0, 1)
        orig_im = np.reshape(orig_im, (28,28))
        matplotlib.pyplot.imsave(os.path.join(file_path_orig,('img_'+str(img_num)+'.png')), orig_im, cmap = 'Greys_r')
        img_num+=1

    f1 = file_path_adv + "true_label.p"
    f2 = file_path_adv + "target_label.p"

    with open(f1, 'wb') as f:
        pickle.dump(adv_ex_y_true,f)
    f.close()
    with open(f2, 'wb') as f:
        pickle.dump(adv_ex_y_target,f)
    f.close()
    
    return (or_ex_x, adv_ex_x, adv_ex_y_true)


# In[ ]:


def append_adv_ex():
    N = num_adv_train
    o_x, a_x, a_y = gen_adv_ex_set(N, train_set = True)
    M = 60000-N
    print(np.shape(a_x))
    print(np.shape(train_x))
    train_x_desired_app = np.concatenate((train_x[0:M], o_x), axis = 0)
    train_x_app = np.concatenate((train_x[0:M], a_x), axis = 0)
    train_y_app = np.concatenate((train_y[0:M], a_y), axis = 0)
    print(np.shape(train_x_app))
    print(np.shape(train_y_app))
    
    return (train_x_desired_app, train_x_app, train_y_app)


# In[ ]:


def append_adv_test_ex():
    N = num_adv_test
    o_x, a_x, a_y = gen_adv_ex_set(N, train_set = False)
    M = 4000-N
    print(np.shape(a_x))
    #print(np.shape(train_x))
    test_x_desired_app = np.concatenate((test_x[0:M], o_x), axis = 0)
    test_x_app = np.concatenate((test_x[0:M], a_x), axis = 0)
    test_y_app = np.concatenate((test_y[0:M], a_y), axis = 0)
    print(np.shape(test_x_app))
    print(np.shape(test_y_app))
    
    return (test_x_desired_app, test_x_app, test_y_app)

train_x_desired_app, train_x_app, train_y_app = append_adv_ex()
test_x_desired_app, test_x_app, test_y_app = append_adv_test_ex()

# In[59]:
train_x_app = train_x_app.astype(np.float32)
train_x_desired_app = train_x_desired_app.astype(np.float32)
test_x_app = test_x_app.astype(np.float32)
test_x_desired_app = test_x_desired_app.astype(np.float32)
#test_x_app = test_x_app.astype(np.float32)




# In[ ]:


#train on train_x_app and train_y_app
#settings
do_train_model = True #False
batch_size = 100
latent_size = 20
nhidden = 512
lr = 0.001
num_epochs = 50
model_filename = "mnist_vae_adv_trained"
nonlin = lasagne.nonlinearities.rectify

np.random.seed(1234) # reproducibility


# In[ ]:


### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, nfeatures))
l_noise = lasagne.layers.BiasLayer(l_in, b = np.zeros(nfeatures, dtype = np.float32), name = "NOISE")
l_noise.params[l_noise.b].remove("trainable")
l_enc_h1 = lasagne.layers.DenseLayer(l_noise, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE1')
l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE2')

l_mu = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_MU')
l_log_var = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_LOG_VAR')

#sample the latent variables using mu(x) and log(sigma^2(x))
l_z = SimpleSampleLayer(mean=l_mu, log_var=l_log_var)


# In[ ]:


### GENERATIVE MODEL p(x|z)
l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE2')
l_dec_h1 = lasagne.layers.DenseLayer(l_dec_h1, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE1')
l_dec_x_mu = lasagne.layers.DenseLayer(l_dec_h1, num_units=nfeatures, nonlinearity=lasagne.nonlinearities.sigmoid, name='DEC_X_MU')

# Get outputs from model
# with noise
z_train, z_mu_train, z_log_var_train, x_mu_train = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=False
)

# without noise
#edit: changed deterministic = True to deterministic = False
z_eval, z_mu_eval, z_log_var_eval, x_mu_eval = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=False
)


# In[ ]:


# TRAINING LogLikelihood
LL_train = ELBO(z_train, z_mu_train, z_log_var_train, x_mu_train, sym_x_desired)

# EVAL LogLikelihood
LL_eval = ELBO(z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, sym_x_desired)

params = lasagne.layers.get_all_params(l_dec_x_mu, trainable=True)
for p in params:
    print (p, p.get_value().shape)

### Take gradient of Negative LogLikelihood
grads = T.grad(-LL_train, params)

# Add gradclipping to reduce the effects of exploding gradients.
# This speeds up convergence
clip_grad = 1
max_norm = 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g,-clip_grad, clip_grad) for g in mgrads]


# In[ ]:


#Setup the theano functions
sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

train_model = theano.function([sym_batch_index, sym_lr], LL_train, updates=updates,
                                  givens={sym_x: sh_x_train[batch_slice],sym_x_desired: sh_x_desired_train[batch_slice]},)

test_model = theano.function([sym_batch_index], LL_eval,
                                  givens={sym_x: sh_x_test[batch_slice],sym_x_desired: sh_x_desired_test[batch_slice]},)

plot_results = theano.function([sym_batch_index], x_mu_eval,
                                  givens={sym_x: sh_x_test[batch_slice]},)

def train_epoch(lr):
    costs = []
    for i in range(n_train_batches):
        cost_batch = train_model(i, lr)
        costs += [cost_batch]
    return np.mean(costs)


def test_epoch():
    costs = []
    for i in range(n_test_batches):
        cost_batch = test_model(i)
        costs += [cost_batch]
    return np.mean(costs)


# In[ ]:


mean, log_var, reconstruction = lasagne.layers.get_output(
    [l_mu, l_log_var, l_dec_x_mu], inputs = sym_x, deterministic=True)
    
# Adversarial confusion cost function
    
# Mean squared reconstruction difference
# KL divergence between latent variables
adv_mean =  T.vector()
adv_log_var = T.vector()
adv_confusion = kld(mean, log_var, adv_mean, adv_log_var).sum()

# Adversarial regularization
C = T.scalar()
adv_reg = C*lasagne.regularization.l2(l_noise.b)
# Total adversarial loss
adv_loss = adv_confusion + adv_reg
adv_grad = T.grad(adv_loss, l_noise.b)

# Function used to optimize the adversarial noise
adv_function = theano.function([sym_x, adv_mean, adv_log_var, C], [adv_loss, adv_grad])

# Helper to plot reconstructions    
adv_plot = theano.function([sym_x], reconstruction)

# Function to get latent variables of the target
adv_mean_log_var = theano.function([sym_x], [mean, log_var])


# In[ ]:


sh_x_train = theano.shared(train_x_app, borrow=True)
sh_x_test = theano.shared(test_x_app, borrow=True)
sh_x_desired_train = theano.shared(train_x_desired_app, borrow=True)
sh_x_desired_test = theano.shared(test_x_desired_app, borrow=True)

nfeatures=train_x_app.shape[1]
n_train_batches = int(train_x_app.shape[0] / batch_size)
n_test_batches = int(test_x_app.shape[0] / batch_size)


# In[ ]:


if do_train_model:
    # Training Loop
    print("Training Adversarial Model")
    for epoch in range(num_epochs):
        start = time.time()

        #shuffle train data, train model and test model
        #s = np.arange(train_x_app.shape[0])
        #np.random.shuffle(s)
        #train_x_app = train_x_app[s]
        #train_x_desired_app = train_x_desired_app[s]
        sh_x_train.set_value(train_x_app)
        sh_x_desired_train.set_value(train_x_desired_app)
        
        #s = np.arange(test_x_app.shape[0])
        #np.random.shuffle(s)
        #test_x_app = test_x_app[s]
        #test_x_desired_app = test_x_desired_app[s]
        sh_x_test.set_value(test_x_app)
        sh_x_desired_test.set_value(test_x_desired_app)
            
        train_cost = train_epoch(lr)
        test_cost = test_epoch()

        t = time.time() - start

        line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
        print (line)
    
    print ("Write model data")
    write_model(l_dec_x_mu, model_filename)
else:
    read_model(l_dec_x_mu, model_filename)
    
read_model(l_dec_x_mu, model_filename)


# In[ ]:


print("############################################################")
print("############################################################")
# In[73]:
print()
print("After Adversarial Training")
print("############################################################")
print("############################################################")
print()


# In[ ]:


n = num_test_attacks
auddc_list = []
time_taken = []
bn = []
for i in range(n):
    start_time = time.time()
    df = orig_adv_dist(plot = True, iteration=i)
    end_time = time.time()
    AUDDC = df.at[0,'AUDDC']
    best_noise = df.at[0,'best_noise']
    auddc_list.append(AUDDC)
    time_taken.append(end_time-start_time)
    bn.append(best_noise)
    print ("Iter", i, "Time", time.time() - start_time, "sec")
    print("############################################################")
    #print(df.values)
    #f = "results/" + model_filename + "/exp_" + str(i) + ".txt"
    #np.savetxt(f, df.values, fmt = "%d")
    #df.to_csv("results/" + model_filename + "/exp_" + str(i) + ".csv", decimal=',', sep=' ', float_format='%.3f')
print("Average AUDDC: ", sum(auddc_list)/len(auddc_list))
print("Average time taken for attack: ", sum(time_taken)/len(time_taken))
print("Average noise added: ", sum(bn)/len(bn))
