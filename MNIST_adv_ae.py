#python3 MNIST_adv_ae.py num_test_attacks num_adv_train num_adv_test bin/mean/none

import numpy as np
import pandas as pd
import lasagne
import theano
import theano.tensor as T
from theano import printing
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
from PIL import Image
import pickle
from sklearn.metrics import auc
from skimage.filters.rank import mean
from skimage.morphology import disk
import sys

theano.config.floatX = 'float32'
num_test_attacks = int(sys.argv[1])
num_adv_train = int(sys.argv[2])
num_adv_test = int(sys.argv[3])

bin_for_adv = True
mean_for_adv = True

def now():
    return mktime(gmtime())

#settings
if(sys.argv[4]=='mean_filter' or sys.argv[4]=='bin'):
    do_train_model = True
else:
    do_train_model = False
batch_size = 100
latent_size = 20
nhidden = 512
lr = 0.001
num_epochs = 25 #50
model_filename = "mnist_ae_"+sys.argv[4]
nonlin = lasagne.nonlinearities.rectify

np.random.seed(1234) # reproducibility

#SYMBOLIC VARS
sym_x = T.matrix()
sym_x_desired = T.matrix()
sym_lr = T.scalar('lr')

### LOAD DATA
print ("Using MNIST dataset")

(train_x, train_y), (test_x, test_y) = mnist.load_data()

test_x = (test_x.reshape((-1, 784))/255.0).astype(np.float32)
print("train_x shape: ", np.shape(train_x))
print("test_x_shape: ", np.shape(test_x))
ltrain = np.shape(train_x)[0]
ltest = np.shape(test_x)[0]
#binariztion 
if(sys.argv[4]=='bin'):
    train_x = (train_x.reshape((-1, 784))/255.0).astype(np.float32)
    train_x[train_x > 0.5] = 1.0
    train_x[train_x <= 0.5] = 0.0

    #test_x[test_x > 0.5] = 1.0
    #test_x[test_x <= 0.5] = 0.0

if(sys.argv[4]=='mean_filter'):
    radius = 2
    train_x_list = [mean(train_x[i], disk(radius)) for i in range(0, ltrain)]
    train_x = np.asarray(train_x_list)
    train_x = (train_x.reshape((-1, 784))/255.0).astype(np.float32)
    #test_x_list = [mean(test_x[i], disk(radius)) for i in range(0, ltest)]
    #test_x = np.asarray(test_x_list)

#setup shared variables
sh_x_train = theano.shared(train_x, borrow=True)
sh_x_test = theano.shared(test_x, borrow=True)
sh_x_desired_train = theano.shared(train_x, borrow=True)
sh_x_desired_test = theano.shared(test_x, borrow=True)

nfeatures=train_x.shape[1]
n_train_batches = int(train_x.shape[0] / batch_size)
n_test_batches = int(test_x.shape[0] / batch_size)

### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, nfeatures))
l_noise = lasagne.layers.BiasLayer(l_in, b = np.zeros(nfeatures, dtype = np.float32), name = "NOISE")
l_noise.params[l_noise.b].remove("trainable")
l_enc_h1 = lasagne.layers.DenseLayer(l_noise, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE1')
l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE2')

l_z = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='Z')

### GENERATIVE MODEL p(x|z)
l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE2')
l_dec_h1 = lasagne.layers.DenseLayer(l_dec_h1, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE1')
l_dec_x = lasagne.layers.DenseLayer(l_dec_h1, num_units=nfeatures, nonlinearity=lasagne.nonlinearities.sigmoid, name='DEC_X_MU')

dec_x = lasagne.layers.get_output(l_dec_x, sym_x, deterministic=False)

#loss = lasagne.objectives.squared_error(dec_x, sym_x.reshape((batch_size, -1)))
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

#Setup the theano functions
sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

#train_model = theano.function([sym_batch_index, sym_lr], loss, updates=updates,
#                                 givens={sym_x: sh_x_train[batch_slice]},)

train_model = theano.function([sym_batch_index, sym_lr], loss, updates=updates,
                              givens={sym_x: sh_x_train[batch_slice] , sym_x_desired: sh_x_desired_train[batch_slice]},)

test_model = theano.function([sym_batch_index], loss,
                             givens={sym_x: sh_x_test[batch_slice], sym_x_desired: sh_x_desired_test[batch_slice]},)

plot_results = theano.function([sym_batch_index], dec_x,
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


# In[11]:


if do_train_model:
    # Training Loop
    for epoch in range(num_epochs):
        start = time.time()

        #shuffle train data, train model and test model
        np.random.shuffle(train_x)
        sh_x_train.set_value(train_x)
        sh_x_desired_train.set_value(train_x)
                                        
        train_cost = train_epoch(lr)
        test_cost = test_epoch()

        t = time.time() - start

        line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
        print (line)
    
    print ("Write model data")
    write_model(l_dec_x, model_filename)
else:
    read_model(l_dec_x, model_filename)

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


def show_mnist(img, i, title=""): # expects flattened image of shape (3072,) 
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(3, 2, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")
    
def mnist_input(img):
    return np.tile(img, (batch_size, 1, 1, 1)).reshape(batch_size, 784)

def mnist_dist(img1, img2):
    return np.linalg.norm(img1 - img2)


# In[14]:


l_z, reconstruction = lasagne.layers.get_output([l_z, l_dec_x], inputs = sym_x, deterministic=True)
# Adversarial confusion cost function
#print("shape of l_z, reconstruction: ", T.shape(l_z), T.shape(reconstruction))

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


# In[15]:


def adv_test(orig_img = 0, target_img = 1, C = 200.0, plot = True):
    # Set the adversarial noise to zero
    l_noise.b.set_value(np.zeros((784,)).astype(np.float32))
    
    # Get latent variables of the target
    adv_target_z = adv_l_z(mnist_input(test_x[target_img]))
    adv_target_z = adv_target_z[0] #doubt : why taking the 0th element only????
    
    #print("shape of adv_target_z: ", np.shape(adv_target_z))
    #print("shape of adv_target_z[0]: ", np.shape(adv_target_z[0]))

    original_reconstruction = adv_plot(mnist_input(test_x[orig_img]))[0]
    target_reconstruction = adv_plot(mnist_input(test_x[target_img]))[0]
    
    orig_recon_dist = mnist_dist(original_reconstruction, test_x[orig_img])
    target_recon_dist = mnist_dist(target_reconstruction, test_x[target_img])
    orig_target_recon_dist = mnist_dist(original_reconstruction, test_x[target_img])
    target_orig_recon_dist = mnist_dist(target_reconstruction, test_x[orig_img])
    
    # Initialize the adversarial noise for the optimization procedure
    l_noise.b.set_value(np.random.uniform(-1e-8, 1e-8, size=(784,)).astype(np.float32))
    
    # Optimization function for L-BFGS-B
    def fmin_func(x):
        l_noise.b.set_value(x.astype(np.float32))
        f, g = adv_function(mnist_input(test_x[orig_img]), adv_target_z, C)
        #f and g are adversarial loss and gradient, respectively
        return float(f), g.flatten().astype(np.float64)
        
    # Noise bounds (pixels cannot exceed 0-1)
    bounds = list(zip(-test_x[orig_img], 1-test_x[orig_img]))
    #bounds = list(zip(-test_x_app[orig_img], 1-test_x_app[orig_img]))
    
    # L-BFGS-B optimization to find adversarial noise
    x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value().flatten(), bounds = bounds, fprime = None, factr = 10, m = 25)
    
    adv_img = adv_plot(mnist_input(test_x[orig_img]))[0]
    #doubt: how is the optimal noise added to the original image!?
    #adv_img is the image reconstructed by the AE from the adversarial image?
    
    orig_dist = mnist_dist(adv_img, test_x[orig_img])
    adv_dist = mnist_dist(adv_img, test_x[target_img])
    recon_dist = mnist_dist(adv_img, test_x[orig_img]+x)
    #orig_dist should be small
    #adv_dist should be big
    #for good attacker
    
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
        
        img = original_reconstruction
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
        img = test_x[orig_img] + x
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

        img = adv_img
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
        #show_mnist(adv_img, 6, "Adversarial reconstruction")
    
        #plt.show()
    
    orig_target_dist = np.linalg.norm(test_x[orig_img] - test_x[target_img])
    #orig_target_dist = np.linalg.norm(test_x_app[orig_img] - test_x_app[target_img])
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


# In[46]:


def orig_adv_dist(orig_img = None, target_img = None, plot = False, bestC = None, iteration = 1):
    if orig_img is None:
        orig_img = np.random.randint(0, len(test_x))
        #orig_img = np.random.randint(0, len(test_x_app))
    if target_img is None:
        target_label = test_y[orig_img]
        #target_label = test_y_app[orig_img]
        while target_label == test_y[orig_img]:
        #while target_label == test_y_app[orig_img]:
            target_img = np.random.randint(0, len(test_x))
            target_label = test_y[target_img]
            #target_img = np.random.randint(0, len(test_x_app))
            #target_label = test_y_app[target_img]
    
    noise_dist = []
    orig_dist=[]
    adv_dist=[]
    target_recon_dist=[]
    recon_dist=[]
    orig_target_dist=[]
    orig_target_recon_dist=[]
    target_orig_recon_dist=[]

    
    C = np.logspace(-20, 20, 100, base = 2, dtype = np.float32)
    
    for c in C:
        noise, od, ad, ore, tre, recd, otd, otrd, tord = adv_test(orig_img, target_img, C=c, plot = False)
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

    #print (orig_img, target_img, bestC)

    best_noise, best_orig_dist, best_adv_dist, orig_reconstruction_dist, target_reconstruction_dist, _, _, _, _ = adv_test(orig_img, target_img, C=bestC, plot = True)

    plt.ioff()
    if plot:
        fig = plt.figure()
        plt.axhline(y=target_reconstruction_dist, linewidth = 2, color = 'cyan', label = "Dist(Target reconstruction - Target)")
        plt.axvline(x=orig_reconstruction_dist, linewidth = 2, color='DarkOrange', label = "Dist(Original reconstruction - Original)")
        plt.scatter(orig_dist, adv_dist)
        plt.scatter([best_orig_dist], [best_adv_dist], color = "red")
        plt.xlabel("Dist(recon Adversarial image, Original image)")
        plt.ylabel("Dist(recon Adversarial image, Target image)")
        plt.legend()
        plt.plot()
        #output_dir = '/Users/rishikaagarwal/Desktop/cs597/adv_vae-master/results/' + model_filename + '/'
        output_dir = 'results/' + model_filename + '/'
        #os.path.join(output_dir, {}/exp_'+ str(iteration)+ '.png')
        fig.savefig(os.path.join(output_dir, ('exp_'+ str(iteration)+ 'graph1.png')))
        plt.close(fig)
        
        fig = plt.figure()
        plt.axhline(y=target_reconstruction_dist, linewidth = 2, color = 'cyan', label = "Dist(Target reconstruction - Target)")
        plt.axvline(x=np.linalg.norm(test_x[orig_img]-test_x[target_img]), linewidth = 2, color='DarkOrange', label = "Dist(Original - Target)")
        plt.scatter(noise_dist, adv_dist)
        plt.scatter([best_noise], [best_adv_dist], color = "red")
        plt.ylabel("Dist(reconAdversarial image, Target image)")
        plt.xlabel("noise")
        plt.legend()
        fig.savefig(os.path.join(output_dir, ('exp_'+ str(iteration)+ 'graph2.png')))
        plt.plot()
        plt.close(fig)

        #compute AUDDC

       
    

    #orig_img : index of original image
    #target_img : index of target image
    #bestC: 
    #orig_reconstruction_dist : distance b/w original image and its recons
    #target_rec_dist :  distance b/w original image and its recons
    #noise dist : 
    #orig_dist : 
    #adv_dist :

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

    bestCind = np.where(C==bestC)
    
    print('orig_img : ',orig_img)
    print('target_img : ', target_img)
    #print('orig_reconstruction_dist : ', orig_reconstruction_dist)
    #print('target_reconstruction_dist : ',target_reconstruction_dist)
    #print('original_target_dist : ', orig_target_dist[bestCind])
    #print('orig_target_recon_dist : ', orig_target_recon_dist[bestCind])
    #print('target_orig_recon_dist : ', target_orig_recon_dist[bestCind])

    print()
    print('bestC : ', bestC)
    print('adv_adv_recon_dist : ', recon_dist[bestCind])
    print('best noise_dist :  ', noise_dist[bestCind])
    print('best orig_dist :  ', orig_dist[bestCind])
    print('best adv_dist : ', adv_dist[bestCind])
    print('AUDDC: ', AUDDC)
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
                       'C': C,
                       'AUDDC':AUDDC,
                       'best_noise': best_noise
                        })
    
    return df


n = num_test_attacks
auddc_list = []
time_taken = []
bn = []

for i in range(n):
    start_time = time.time()
    df = orig_adv_dist(plot = True, iteration = i)
    end_time = time.time()
    AUDDC = df.at[0,'AUDDC']
    best_noise = df.at[0,'best_noise']
    bn.append(best_noise)
    auddc_list.append(AUDDC)
    time_taken.append(end_time-start_time)

    print ("Iter", i, "Time", end_time-start_time, "sec")
    print("############################################################")
    #print(df.values)
    #f = "results/" + model_filename + "/exp_" + str(i) + ".txt"
    #np.savetxt(f, df.values, fmt = "%d")
    #df.to_csv("results/" + model_filename + "/exp_" + str(i) + ".csv", decimal=',', sep=' ', float_format='%.3f')

print("Average AUDDC: ", sum(auddc_list)/len(auddc_list))
print("Average time taken for attack: ", sum(time_taken)/len(time_taken))
print("Average noise added: ", sum(bn)/len(bn))
print("=====================================================")
def gen_adv_ex(orig_img, target_img, C):
    
    # Set the adversarial noise to zero
    l_noise.b.set_value(np.zeros((784,)).astype(np.float32))
    
    # Get latent variables of the target
    adv_target_z = adv_l_z(mnist_input(train_x[target_img]))
    adv_target_z = adv_target_z[0]

    # Initialize the adversarial noise for the optimization procedure
    l_noise.b.set_value(np.random.uniform(-1e-8, 1e-8, size=(784,)).astype(np.float32))
    
    # Optimization function for L-BFGS-B
    def fmin_func(x):
        l_noise.b.set_value(x.astype(np.float32))
        f, g = adv_function(mnist_input(train_x[orig_img]), adv_target_z, C)
        return float(f), g.flatten().astype(np.float64)
        
    # Noise bounds (pixels cannot exceed 0-1)
    bounds = list(zip(-train_x[orig_img], 1-train_x[orig_img]))
    
    # L-BFGS-B optimization to find adversarial noise
    
    x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value().flatten(), bounds = bounds, fprime = None, factr = 10, m = 25)
    
    adv_img = adv_plot(mnist_input(train_x[orig_img]))[0]
    
    orig_dist = mnist_dist(adv_img, train_x[orig_img])
    adv_dist = mnist_dist(adv_img, train_x[target_img])
    recon_dist = mnist_dist(adv_img, train_x[orig_img]+x)
    
    returns = (np.linalg.norm(x),
                x,
               orig_dist,
               adv_dist)
               
    return returns

def gen_adv_ex_set(N, train_set):
    #N = 400
    or_ex_x = []
    adv_ex_x = []
    adv_ex_y_target = []
    adv_ex_y_true = []
    '''
    if(train_set==True):
        file_path_adv = 'dataset/train/adversarial_images/'
        file_path_orig = 'dataset/train/original_images/'
    else:
        file_path_adv = 'dataset/test/adversarial_images/'
        file_path_orig = 'dataset/test/original_images/'
    '''
    img_num = 0
    for i in range(N):
        if(i%50==0):
            print("generating ",i,"th adversarial example")


        orig_img = np.random.randint(0, len(train_x))
        
        target_label = train_y[orig_img]
        while target_label == train_y[orig_img]:
            target_img = np.random.randint(0, len(train_x))
            target_label = train_y[target_img]
        
        noise_dist = []
        orig_dist = []
        adv_dist = []
        
        C = np.logspace(-20, 20, 100, base = 2, dtype = np.float32)
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
        
        '''
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
        
        adv_im = train_x[orig_img] + best_noise_matrix
        adv_im = np.reshape(adv_im, (28,28))
        adv_im = np.clip(adv_im, 0, 1)
        #matplotlib.pyplot.imsave(os.path.join(file_path_adv,('img_'+str(img_num)+'.png')), adv_im, cmap = 'Greys_r')
        adv_im[adv_im>0.5] = 1
        adv_im[adv_im<=0.5]= 0
        #matplotlib.pyplot.imsave(os.path.join(file_path_adv,('img_'+str(img_num)+'_bin.png')), adv_im, cmap = 'Greys_r')
        orig_im = train_x[orig_img]
        orig_im = np.clip(orig_im, 0, 1)
        orig_im = np.reshape(orig_im, (28,28))
        #matplotlib.pyplot.imsave(os.path.join(file_path_orig,('img_'+str(img_num)+'.png')), orig_im, cmap = 'Greys_r')
        img_num+=1

    #f1 = file_path_adv + "true_label.p"
    #f2 = file_path_adv + "target_label.p"
    '''
    with open(f1, 'wb') as f:
        pickle.dump(adv_ex_y_true,f)
    f.close()
    with open(f2, 'wb') as f:
        pickle.dump(adv_ex_y_target,f)
    f.close()
    '''
    return (or_ex_x, adv_ex_x, adv_ex_y_true)

# In[57]:
def append_adv_ex():
    N = num_adv_train
    beg = time.time()
    o_x, a_x, a_y = gen_adv_ex_set(N, train_set = True)
    a_x = np.array(a_x)
    end = time.time()
    '''
    if(sys.argv[4]=='bin' and bin_for_adv == True):

        a_x[a_x > 0.5] = 1.0
        a_x[a_x <= 0.5] = 0.0

    if(sys.argv[4]=='mean_filter' and mean_for_adv==True):
        radius = 2
        l = np.shape(a_x)[0]
        a_x_list = [mean(a_x[i], disk(radius)) for i in range(0, l)]
        a_x = np.asarray(a_x_list)
    ''' 
    
    M = 60000-N
    #print(np.shape(a_x))
    #print(np.shape(train_x))
    train_x_desired_app = np.concatenate((train_x[0:M], o_x), axis = 0)
    train_x_app = np.concatenate((train_x[0:M], a_x), axis = 0)
    train_y_app = np.concatenate((train_y[0:M], a_y), axis = 0)
    #print(np.shape(train_x_app))
    #print(np.shape(train_y_app))
    print("Time taken to generate ",N, " adversarial examples: ", end-beg)
    return (train_x_desired_app, train_x_app, train_y_app)

# In[58]:
def append_adv_test_ex():
    N = num_adv_test
    beg = time.time()
    o_x, a_x, a_y = gen_adv_ex_set(N, train_set = False)
    a_x = np.array(a_x)
    end = time.time()
    '''
    if(sys.argv[4]=='bin' and bin_for_adv == True):

        a_x[a_x > 0.5] = 1.0
        a_x[a_x <= 0.5] = 0.0
    if(sys.argv[4]=='mean_filter' and mean_for_adv==True):
        radius = 2
        l = np.shape(a_x)[0]
        a_x_list = [mean(a_x[i], disk(radius)) for i in range(0, l)]
        a_x = np.asarray(a_x_list)
    '''
    M = 4000-N
    #print(np.shape(a_x))
    #print(np.shape(train_x))
    test_x_desired_app = np.concatenate((test_x[0:M], o_x), axis = 0)
    test_x_app = np.concatenate((test_x[0:M], a_x), axis = 0)
    test_y_app = np.concatenate((test_y[0:M], a_y), axis = 0)
    #print(np.shape(test_x_app))
    #print(np.shape(test_y_app))
    print("Time taken to generate ",N, " adversarial examples: ", end-beg)
    return (test_x_desired_app, test_x_app, test_y_app)

train_x_desired_app, train_x_app, train_y_app = append_adv_ex()
test_x_desired_app, test_x_app, test_y_app = append_adv_test_ex()

# In[59]:
train_x_app = train_x_app.astype(np.float32)
train_x_desired_app = train_x_desired_app.astype(np.float32)
test_x_app = test_x_app.astype(np.float32)
test_x_desired_app = test_x_desired_app.astype(np.float32)
#test_x_app = test_x_app.astype(np.float32)

#train on train_x_app and train_y_app
#settings
do_train_model = True #False
batch_size = 20
latent_size = 20
nhidden = 512
lr = 0.001
num_epochs = 25 #50
model_filename = "mnist_ae_adv_trained_"+sys.argv[4]
nonlin = lasagne.nonlinearities.rectify

np.random.seed(1234) # reproducibility

#SYMBOLIC VARS
sym_x = T.matrix()
sym_x_desired = T.matrix()
sym_lr = T.scalar('lr')

#setup shared variables
sh_x_train = theano.shared(train_x_app, borrow=True)
sh_x_test = theano.shared(test_x_app, borrow=True)
sh_x_desired_train = theano.shared(train_x_desired_app, borrow=True)
sh_x_desired_test = theano.shared(test_x_desired_app, borrow=True)


nfeatures=train_x_app.shape[1]
n_train_batches = int(train_x_app.shape[0] / batch_size)
n_test_batches = int(test_x_app.shape[0] / batch_size)

### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, nfeatures))
l_noise = lasagne.layers.BiasLayer(l_in, b = np.zeros(nfeatures, dtype = np.float32), name = "NOISE")
l_noise.params[l_noise.b].remove("trainable")
l_enc_h1 = lasagne.layers.DenseLayer(l_noise, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE1')
l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=nhidden, nonlinearity=nonlin, name='ENC_DENSE2')

l_z = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='Z')

### GENERATIVE MODEL p(x|z)
l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE2')
l_dec_h1 = lasagne.layers.DenseLayer(l_dec_h1, num_units=nhidden, nonlinearity=nonlin, name='DEC_DENSE1')
l_dec_x = lasagne.layers.DenseLayer(l_dec_h1, num_units=nfeatures, nonlinearity=lasagne.nonlinearities.sigmoid, name='DEC_X_MU')

dec_x = lasagne.layers.get_output(l_dec_x, sym_x, deterministic=False)

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

#Setup the theano functions
sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

train_model = theano.function([sym_batch_index, sym_lr], loss, updates=updates,
                              givens={sym_x: sh_x_train[batch_slice], sym_x_desired: sh_x_desired_train[batch_slice]},)

test_model = theano.function([sym_batch_index], loss,
                             givens={sym_x: sh_x_test[batch_slice], sym_x_desired: sh_x_desired_test[batch_slice]},)

plot_results = theano.function([sym_batch_index], dec_x,
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

if do_train_model:
    # Training Loop
    print("Training Adversarial Model")
    for epoch in range(num_epochs):
        start = time.time()

        #shuffle train data, train model and test model
        s = np.arange(train_x_app.shape[0])
        np.random.shuffle(s)
        train_x_app = train_x_app[s]
        train_x_desired_app = train_x_desired_app[s]
        sh_x_train.set_value(train_x_app)
        sh_x_desired_train.set_value(train_x_desired_app)
        
        s = np.arange(test_x_app.shape[0])
        np.random.shuffle(s)
        test_x_app = test_x_app[s]
        test_x_desired_app = test_x_desired_app[s]
        sh_x_test.set_value(test_x_app)
        sh_x_desired_test.set_value(test_x_desired_app)
            
        train_cost = train_epoch(lr)
        test_cost = test_epoch()

        t = time.time() - start

        line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
        print (line)
    
    print ("Write model data")
    write_model(l_dec_x, model_filename)

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

print("=====================================================")
print("=====================================================")
print()
print("After Adversarial Training")
print()
print("=====================================================")
print("=====================================================")
print()

n = num_test_attacks
auddc_list = []
time_taken = []
bn = []

for i in range(n):
    start_time = time.time()
    df = orig_adv_dist(plot = True, iteration = i)
    end_time = time.time()
    AUDDC = df.at[0,'AUDDC']
    best_noise = df.at[0,'best_noise']
    bn.append(best_noise)
    auddc_list.append(AUDDC)
    time_taken.append(end_time-start_time)
    print ("Iter", i, "Time", end_time - start_time, "sec")
    print("-----------------------------------------------------------")
    
print("Average AUDDC: ", sum(auddc_list)/len(auddc_list))
print("Average time taken for attack: ", sum(time_taken)/len(time_taken))
print("Average noise added: ", sum(bn)/len(bn))
print("=====================================================")
