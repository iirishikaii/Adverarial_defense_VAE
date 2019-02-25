
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
import sys

theano.config.floatX = 'float32'

def now():
    return mktime(gmtime())

#settings
do_train_model = False #False
batch_size = 100
latent_size = 20
nhidden = 512
lr = 0.001
num_epochs = 5 #50
#model_filename = "mnist_ae"
#model_filename = "mnist_ae_adv_trained"
model_filename = sys.argv[1]
nonlin = lasagne.nonlinearities.rectify

np.random.seed(1234) # reproducibility

#SYMBOLIC VARS
sym_x = T.matrix()
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

nfeatures=train_x.shape[1]
n_train_batches = int(train_x.shape[0] / batch_size)
n_test_batches = int(test_x.shape[0] / batch_size)

print('train_x shape: ', train_x.shape)
# In[7]:


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


# In[9]:


loss = lasagne.objectives.squared_error(dec_x, sym_x.reshape((batch_size, -1)))
loss = lasagne.objectives.aggregate(loss, mode="mean")

params = lasagne.layers.get_all_params(l_dec_x, trainable=True)
#for p in params:
 #   print (p, p.get_value().shape)

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
                              givens={sym_x: sh_x_train[batch_slice]},)

test_model = theano.function([sym_batch_index], loss,
                             givens={sym_x: sh_x_test[batch_slice]},)

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
        
        train_cost = train_epoch(lr)
        test_cost = test_epoch()
        
        t = time.time() - start
        
        line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
        print (line)
    
    print ("Write model data")
    write_model(l_dec_x, model_filename)
else:
    read_model(l_dec_x, model_filename)

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

def adv_test(orig_img = 0, target_img = 1, C = 200.0, plot = True, iteration = 1):
    # Set the adversarial noise to zero
    l_noise.b.set_value(np.zeros((784,)).astype(np.float32))
    
    # Get latent variables of the target
    adv_target_z = adv_l_z(mnist_input(test_x[target_img]))
    #adv_target_z = adv_l_z(mnist_input(test_x_app[target_img]))
    adv_target_z = adv_target_z[0]
    
    original_reconstruction = adv_plot(mnist_input(test_x[orig_img]))[0]
    target_reconstruction = adv_plot(mnist_input(test_x[target_img]))[0]
    #original_reconstruction = adv_plot(mnist_input(test_x_app[orig_img]))[0]
    #target_reconstruction = adv_plot(mnist_input(test_x_app[target_img]))[0]
    
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
        #f, g = adv_function(mnist_input(test_x_app[orig_img]), adv_target_z, C)
        return float(f), g.flatten().astype(np.float64)
    
    # Noise bounds (pixels cannot exceed 0-1)
    bounds = list(zip(-test_x[orig_img], 1-test_x[orig_img]))
    #bounds = list(zip(-test_x_app[orig_img], 1-test_x_app[orig_img]))
    
    # L-BFGS-B optimization to find adversarial noise
    x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value().flatten(), bounds = bounds, fprime = None, factr = 10, m = 25)
    
    adv_img = adv_plot(mnist_input(test_x[orig_img]))[0]
    #adv_img = adv_plot(mnist_input(test_x_app[orig_img]))[0]
    
    orig_dist = mnist_dist(adv_img, test_x[orig_img])
    adv_dist = mnist_dist(adv_img, test_x[target_img])
    recon_dist = mnist_dist(adv_img, test_x[orig_img]+x)
    #orig_dist should be small
    #adv_dist should be big
    #for good attacker
    
    
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
               target_orig_recon_dist,
               #orig_img, #original image
               original_reconstruction.copy().reshape(28, 28), #otiginal recon
               #target_img, #target image
               x.copy().reshape(28, 28), #noise
               (test_x[orig_img].flatten()+x).copy().reshape(28, 28), #adv image
               adv_img.copy().reshape(28, 28) #adv recon
               )
        
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

    C = np.logspace(-10, 25, 50, base = 2, dtype = np.float32)

    for c in C:
        noise, od, ad, ore, tre, recd, otd, otrd, tord,_,_,_,_ = adv_test(orig_img, target_img, C=c, plot = False, iteration = iteration)
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

    best_noise, best_orig_dist, best_adv_dist, orig_reconstruction_dist, target_reconstruction_dist, _, _, _, _,orig_recon, x_noise, adv_im, adv_recon = adv_test(orig_img, target_img, C=bestC, plot = True, iteration = iteration)
    
    plt.ioff()
    if plot:
            fig = plt.figure()
            plt.axhline(y=target_reconstruction_dist, linewidth = 2, color = 'cyan', label = "Target reconstruction - Target")
            plt.axvline(x=orig_reconstruction_dist, linewidth = 2, color='DarkOrange', label = "Original reconstruction - Original")
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
            plt.axhline(y=target_reconstruction_dist, linewidth = 2, color = 'cyan', label = "Target reconstruction - Target")
            plt.axvline(x=np.linalg.norm(test_x[orig_img]-test_x[target_img]), linewidth = 2, color='DarkOrange', label = "Original - Target")
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
#print('C', np.mean(C))


    df = pd.DataFrame({'orig_img': orig_img,
                      'target_img': target_img,
                      'bestC': bestC,
                      #'orig_reconstruction_dist': orig_reconstruction_dist,
                      #'target_reconstruction_dist': target_reconstruction_dist,
                      #'noise_dist': noise_dist,
                      #'orig_dist': orig_dist,
                      #'adv_dist': adv_dist,
                      #'target_recon_dist': target_recon_dist,
                      #'recon_dist': recon_dist,
                      #'orig_target_dist': orig_target_dist,
                      #'orig_target_recon_dist': orig_target_recon_dist,
                      #'target_orig_recon_dist': target_orig_recon_dist,
                      #'C': C}
                      'orig_recon': [[orig_recon]],
                      'noise': [[x_noise]],
                      'adv_im': [[adv_im]],
                      'adv_recon': [[adv_recon]]
                      }

                      )

    return df

def mnist_input_single(img):
    return np.tile(img, (1, 1, 1, 1)).reshape(1, 784)

n = 15

#or_im = random.sample(range(1,len(test_x)), n)
#targ_im = random.sample(range(1,len(test_x)), n)
or_im =   [7920, 5500, 8382, 1490, 7040, 5230, 2175, 8027, 8048, 448, 2772, 5282, 4992, 941, 9379]
targ_im = [2717, 2934, 5159, 2362, 1023, 974, 4154, 7465, 550, 6973, 2644, 3927, 2300, 4672, 4043]
#print("original image list: ", or_im)
#print("target image list: ", targ_im)

for i in range(n):
    while(test_y[or_im[i]]==test_y[targ_im[i]]):
        targ_im[i] = targ_im[i]+1

for iteration in range(n):
    start_time = time.time()
    df = orig_adv_dist(orig_img = or_im[iteration], target_img = targ_im[iteration], plot = True, iteration = iteration)
    orig = test_x[or_im[iteration]]
    target = test_x[targ_im[iteration]]
    #print("shape of orig_recon: ", (df.at[0, 'orig_recon'])[0].shape)
    orig_recon = (df.at[0,'orig_recon'])[0]
    noise = (df.at[0,'noise'])[0]
    adv_im = (df.at[0, 'adv_im'])[0]
    adv_recon = (df.at[0, 'adv_recon'])[0]
    adv_bin = adv_im.copy()
    (train_x.reshape((-1, 784))/255.0).astype(np.float32)
    #binarize adv_im
    adv_bin = (adv_bin.reshape((-1, 784))).astype(np.float32)
    adv_bin[adv_bin>0.5]= 1
    adv_bin[adv_bin<=0.5] = 0
    
    #print("adv_bin shape: ", adv_bin.shape)
    #adv_bin_recon = lasagne.layers.get_output(l_dec_x, adv_bin, deterministic = True)
    l_noise.b.set_value(np.zeros((784,)).astype(np.float32))
    adv_bin_recon = adv_plot(mnist_input(adv_bin))[0]
    #print("type of adv_bin_recon: ", type(adv_bin_recon))
    #adv_bin_recon = adv_bin_recon.eval()
    print("adv_bin_recon shape: ",adv_bin_recon.shape)
    #print(adv_bin_recon[0])
    adv_bin_recon = (adv_bin_recon/255.0).astype(np.float32)
    #print("type of adv_bin_recon: ", type(adv_bin_recon))
    fig = plt.figure(figsize=(10,10))
        
    img = orig
    i = 1
    title = "Original Image"
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(4, 2, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")
    #show_mnist(test_x[orig_img], 1, "Original image")
    
    img = orig_recon
    i = 2
    title = "Original Reconstruction"
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(4, 2, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")
    
    img = noise
    i = 3
    title = "Adversarial noise"
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(4, 2, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")
    
    #show_mnist(x, 3, "Adversarial noise")
    
    img = target
    i = 4
    title = "Target image"
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(4, 2, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")
    
    #show_mnist(test_x[target_img], 4, "Target image")
    
    img = adv_im
    i = 5
    title = "Adversarial image"
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(4, 2, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")
    #show_mnist((test_x[orig_img].flatten()+x), 5, "Adversarial image")
    #show_mnist(test_x_app[target_img], 4, "Target image")
    #show_mnist((test_x_app[orig_img].flatten()+x), 5, "Adversarial image")
    
    img = adv_recon
    i = 6
    title = "Adversarial reconstruction"
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(4, 2, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")

    img = adv_bin
    i = 7
    title = "Adversarial Binarized"
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(4, 2, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")

    img = adv_bin_recon
    i = 8
    title = "Adversarial Binarized Reconstruction"
    img = img.copy().reshape(28, 28)
    img = np.clip(img, 0, 1)
    plt.subplot(4, 2, i)
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.axis("off")

    output_dir = 'results/compare_attacks/' + model_filename +'/'
    fig.savefig(os.path.join(output_dir, (str(iteration)+ '.png')))
    plt.close(fig)
    recon = lasagne.layers.get_output(l_dec_x, adv_im)
    print ("Iter", iteration, "Time", time.time() - start_time, "sec")
    print("############################################################")
