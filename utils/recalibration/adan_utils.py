import sys
import warnings

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
from tqdm import notebook

from sklearn.linear_model import LinearRegression


sys.path.append('../utils/recalibration/')
from recalibration_utils import subtractMeans

def getADANData(ref_data_dict, new_data_dict, n_steps, center = False, normalize = False):
    '''Prep data for ADAN by adjusting timeseries lengths to work with step size. Inputs are:
    
    '''

    datas = dict()
    for day, data_dict in zip(['ref', 'new'], [ref_data_dict, new_data_dict]):
        for field in ['TX', 'displacement', 'cursorPos']:
            
            if field == 'TX':
                
                # optionally apply blockwise mean subtraction
                if center:
                    train_TX, test_TX = subtractMeans(data_dict['train_TX'], data_dict['test_TX'],
                                                      method = 'blockwise', concatenate = True)
                else:
                    train_TX = np.concatenate(data_dict['train_TX'])
                    test_TX  = np.concatenate(data_dict['test_TX'])
                    
                # optionally normalize neural features
                if normalize:
                    train_SD = np.std(train_TX, axis = 0, keepdims=True)
                    train_TX /= train_SD
                    test_TX  /= train_SD
                
                datas[f'{day}_train_{field}'] = train_TX
                datas[f'{day}_test_{field}']  = test_TX
                
            else:
                for split in ['train', 'test']:
                    datas[f'{day}_{split}_{field}'] = np.concatenate(data_dict[f'{split}_{field}'])
               
    # perform length adjustment to make timelen divisible by n_steps
    for key, value in datas.items():
        adjusted_len = (len(value)//n_steps)*n_steps
        datas[key]   = value[:adjusted_len]
    
    return datas



def addRandomWalk(timeseries, strength):
    '''Apply mean drift to timeseries data using autoregressive noise. Inputs are:
    
        timeseries (batch x time x features) - data input
        strength (float)                     - noise standard deviation '''
    
    nBatch, nTime, nChannels = timeseries.shape
    noise                    = np.zeros(timeseries.shape)
    noise[:, 0, :]           = np.random.normal(np.zeros((nChannels)), strength) 
    
    for t in range(1, nTime):
        noise[:, t, :] = noise[:, t-1, :] + np.random.normal(np.zeros((nBatch, nChannels)), strength)  
        
    return timeseries + noise


def addWhiteNoise(timeseries, strength):
    '''Apply IID gaussian noise to timeseries data. Inputs are:

    timeseries (batch x time x features) - data input
    strength (float)                     - noise standard deviation '''

    noise = np.random.normal(np.zeros((timeseries.shape)), strength)    
    return timeseries + noise


def addOffset(timeseries, strength):
    '''Add constant offsets to timeseries data. Inputs are: 
    
        timeseries (batch x time x features) - data input
        strength (float)                     - offset standard deviation '''

    nBatch, nTime, nChannels = timeseries.shape
    offset = np.random.normal(np.zeros((nBatch, nChannels)), strength)  
    return timeseries + offset[:, None, :]


def addNoise(timeseries, offset_strength = 0, whitenoise_strength = 0, randomwalk_strength = 0):  
    '''Interface function for adding various noise types to data.'''
    
    if offset_strength > 0:
        timeseries = addOffset(timeseries, strength = offset_strength)
    if whitenoise_strength > 0:
        timeseries = addWhiteNoise(timeseries, strength = whitenoise_strength)
    if randomwalk_strength > 0:
        timeseries = addRandomWalk(timeseries, strength = randomwalk_strength) 
        
    return timeseries



def get_batches(x,batch_size):
    n_batches = len(x)//(batch_size)
    x = x[:n_batches*batch_size:]
    for n in range(0, x.shape[0],batch_size):
        x_batch = x[n:n+(batch_size)]
        yield x_batch
        

        
def vaf(x,xhat, precision = 4):
    x    = x - x.mean(axis=0)
    xhat = xhat - xhat.mean(axis=0)
    vaf  = (1-(np.sum(np.square(x - xhat))/np.sum(np.square(x))))*100
    return np.round(vaf, precision)



class JointNetwork(object):
    def __init__(self, spike_dim, latent_dim, emg_dim, n_layers, n_steps):
        
        self.spike_dim  = spike_dim
        self.latent_dim = latent_dim
        self.emg_dim    = emg_dim
        self.n_units    = [64, 32, self.latent_dim]
        self.n_layers   = n_layers
        self.n_steps    = n_steps
    
    def __call__(self, spike):
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)    
            
            latent,logits = self.autoencoder(spike)
            emg_hat       = self.emg_decoder(latent)
            emg_hat       = tf.identity(emg_hat, name='emg_hat')

        return latent, logits, emg_hat
    
    def autoencoder(self, input_, reuse=False):
        '''Input are:

            input_ (tf tensor) 
            spike_dim (int)    - dimensionality of input neural data
            latent_dim (int)   - dimensionality of bottleneck'''

        with tf.variable_scope('autoencoder', reuse=reuse):
            h1 = tf.layers.dense(input_,     units=self.n_units[0], activation=tf.nn.elu)
            h2 = tf.layers.dense(h1,         units=self.n_units[1], activation=tf.nn.elu)
            latent = tf.layers.dense(h2,     units=self.n_units[2], activation=None)
            h3     = tf.layers.dense(latent, units=self.n_units[1], activation=tf.nn.elu)
            h4     = tf.layers.dense(h3,     units=self.n_units[0], activation=tf.nn.elu)
            logits = tf.layers.dense(h4,     units=self.spike_dim, activation=None)
            return latent, logits


    def emg_decoder_RNN(self, latent, reuse=False):
        with tf.variable_scope('emg_decoder', reuse=reuse):
            latent = tf.reshape(latent,[-1, self.n_steps, self.latent_dim])
            layers = [tf.nn.rnn_cell.LSTMCell(num_units= self.emg_dim, activation=tf.nn.elu) for layer in range(self.n_layers)]
            multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
            outputs, states  = tf.nn.dynamic_rnn(multi_layer_cell,latent, dtype=tf.float32)
            emg_hat = tf.layers.dense(outputs, units = self.emg_dim, activation=None)
            emg_hat = tf.reshape(emg_hat,[-1, self.emg_dim])
            return emg_hat
        
    def emg_decoder(self, latent, reuse=False):
        with tf.variable_scope('emg_decoder', reuse=reuse):
            outputs = tf.layers.dense(latent, units=self.latent_dim, activation=tf.nn.elu)
            emg_hat = tf.layers.dense(outputs, units = self.emg_dim, activation=None)
            return emg_hat
        
        

def train_network(train_x, train_y, test_x, test_y, args):
    '''Train initial autoencoder/task readout network. Inputs are:
    
        train_x/test_x (time x features) - input neural features
        train_y/test_y (time x 2) - displacement vectors
        
        <args> is a dictionary holding configuration parameters: 
        
            'latent_dim' : (int) - bottleneck dimensionality in AE
            'n_layers'   : (int) - # of RNN layers in decoder
            'n_steps'    : (int) - how RNN steps through time ??
            'batch_size' : (int) - minibatch size
            'n_epochs'   : (int) - number of training epochs
            'lr'         : (float) - learning rate
            'save_dir'   : (str) - path for model save directory (None for no save)
            
            'noise_kwargs' : (dict) - dictionary for noise regularization with items:
                
                'offset_strength'     : (float) - set to 0 to skip
                'whitenoise_strength' : (float) - set to 0 to skip
                'randomwalk_strength' : (float) - set to 0 to skip
            '''
    
    assert train_x.shape[1] == test_x.shape[1], "Number of features must be same in train/test"
    assert train_y.shape[1] == test_y.shape[1], "number of output dims must be same in train/test"
    
    spike_dim  = train_x.shape[1]
    emg_dim    = train_y.shape[1]
    
    ae    = JointNetwork(spike_dim, args['latent_dim'], emg_dim, args['n_layers'], args['n_steps'])
    spike = tf.placeholder(tf.float32, (None, spike_dim), name='spike')
    emg   = tf.placeholder(tf.float32, (None, emg_dim), name='emg')
    gamma = tf.placeholder(tf.float32)

    latent, logits, emg_hat = ae(spike)

    ae_loss    = tf.reduce_mean(tf.square(logits - spike))
    emg_loss   = tf.reduce_mean(tf.square(emg_hat - emg))
    total_loss = gamma*ae_loss+emg_loss
    optimizer  = tf.train.AdamOptimizer(learning_rate=args['lr']).minimize(total_loss)
    
    n_batches = len(train_x)//args['batch_size']
    gamma_    = np.float32(1.)
    saver     = tf.train.Saver(max_to_keep=1)
    init      = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        for epoch in notebook.tqdm(range(args['n_epochs'])):
            spike_gen_obj = get_batches(train_x, args['batch_size'])
            emg_gen_obj   = get_batches(train_y, args['batch_size'])
            for ii in range(n_batches):
                emg_batch   = next(emg_gen_obj)
                spike_batch = next(spike_gen_obj)
                spike_batch = addNoise(spike_batch[None, :, :, ], **args['noise_kwargs']).squeeze()
                
                sys.stdout.flush()
                sess.run(optimizer,feed_dict={spike:spike_batch,emg:emg_batch,gamma:gamma_})

            ae_loss_  = ae_loss.eval(feed_dict = {spike:train_x, emg:train_y, gamma:gamma_})
            emg_loss_ = emg_loss.eval(feed_dict= {spike:train_x, emg:train_y, gamma:gamma_})
            gamma_    = emg_loss_/ae_loss_
            if (epoch % 50 == 0) or (epoch == args['n_epochs']-1): 
                spike_hat_tr,spike_hat_te = [logits.eval(feed_dict={spike:train_x}),logits.eval(feed_dict={spike:test_x})]
                emg_hat_tr,emg_hat_te     = [emg_hat.eval(feed_dict={spike:train_x,emg:train_y}),
                                             emg_hat.eval(feed_dict={spike:test_x,emg:test_y})]
                print("Epoch:", epoch, "\tAE_loss:",ae_loss_, "\tEMG_loss:",emg_loss_)
                print("AE Train %VAF:", vaf(train_x,spike_hat_tr),"\AE Test %VAF:", vaf(test_x,spike_hat_te))
                print("EMG Train %VAF:", vaf(train_y,emg_hat_tr),"\EMG Test %VAF:", vaf(test_y,emg_hat_te))
                
        if args['save_dir'] is not None:
            saver.save(sess, f"{args['save_dir']}/decoder")
            
    del ae, spike, emg, gamma, latent, logits, emg_hat, ae_loss, emg_loss, total_loss, optimizer
            
    return vaf(test_y, emg_hat_te) / 100, LinearRegression().fit(train_x, train_y).score(test_x, test_y)



def train_ADAN(arg, model_dir, spike_day0, emg_day0, spike_dayk, emg_dayk, spike_test):
    '''Train ADAN alignment model and predict on holdout data. Inputs are:
    
        arg (dict) - contains key-value pairs: 
        
        
        model_dir (str)                   - path to tensorflow checkpoint
        spike_day0, spike_dayk (2D float) - time x channels of reference and new sessions' neural activity
        emg_day0, emg_dayk (2D float)     - time x 2 of cursor outputs for each sessions
        spike_test (2D float)             - time x channels of holdout data from day k
        
        '''
    
    g       = tf.train.import_meta_graph(os.path.join(model_dir, 'decoder.meta'))
    graph   = tf.get_default_graph()
    spike   = graph.get_tensor_by_name("spike:0")
    emg_hat = graph.get_tensor_by_name(name="emg_hat:0")
    
    input_day0 = tf.placeholder(tf.float32, (None, arg['spike_dim']), name='input_day0')
    input_dayk = tf.placeholder(tf.float32, (None, arg['spike_dim']), name='input_dayk')
    
    def generator(input_, reuse=False):
        with tf.variable_scope('generator',initializer=tf.initializers.identity(),reuse=reuse):
            h1     = tf.layers.dense(input_,  arg['spike_dim'], activation=tf.nn.elu)
            output = tf.layers.dense(h1, arg['spike_dim'], activation=None)
        return output

    def discriminator(input_, n_units=[64,32, arg['latent_dim']], reuse=False):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            noise  = tf.random_normal(tf.shape(input_), dtype=tf.float32) * 50
            input_ = input_+noise
            h1 = tf.layers.dense(input_, units=n_units[0], activation=tf.nn.elu)
            h2 = tf.layers.dense(h1, units=n_units[1], activation=tf.nn.elu)
            latent = tf.layers.dense(h2, units=n_units[2], activation=None)
            h3 = tf.layers.dense(latent, units=n_units[1], activation=tf.nn.elu)
            h4 = tf.layers.dense(h3, units=n_units[0], activation=tf.nn.elu)
            logits = tf.layers.dense(h4, units=arg['spike_dim'], activation=None)
        return latent, logits
    
    
    # setup generator and discriminator, as well as loss fcns
    input_dayk_aligned      = generator(input_dayk)
    latent_day0,logits_day0 = discriminator(input_day0)
    latent_dayk,logits_dayk = discriminator(input_dayk_aligned)

    d_loss_0 = tf.reduce_mean(tf.abs(logits_day0-input_day0)) 
    d_loss_k = tf.reduce_mean(tf.abs(logits_dayk-input_dayk_aligned))
    d_loss = d_loss_0 - d_loss_k
    g_loss = d_loss_k

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    d_opt = tf.train.AdamOptimizer(learning_rate=arg['d_lr']).minimize(d_loss, var_list=d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=arg['g_lr']).minimize(g_loss, var_list=g_vars)
    
    
    n_batches  = min(len(spike_day0),len(spike_dayk))//(arg['batch_size'])
    a_vars     = [var.name for var in t_vars if var.name.startswith('autoencoder')]
    for i,name in enumerate(a_vars):
        tf.train.init_from_checkpoint(os.path.join(model_dir, ''), {name[:-2]:d_vars[i]})

    init = tf.global_variables_initializer()

    with tf.Session() as sess: 
        init.run()
        g.restore(sess, tf.train.latest_checkpoint(os.path.join(model_dir, '')))
        for epoch in notebook.tqdm(range(arg['n_epochs'])):
            spike_0_gen_obj = get_batches(spike_day0, arg['batch_size'])
            spike_k_gen_obj = get_batches(spike_dayk, arg['batch_size'])

            for ii in range(n_batches):
                spike_0_batch = next(spike_0_gen_obj)
                spike_k_batch = next(spike_k_gen_obj)
                sys.stdout.flush()
                _,g_loss_ = sess.run([g_opt,g_loss],feed_dict={input_day0:spike_0_batch,input_dayk:spike_k_batch})
                _,d_loss_0_,d_loss_k_ = sess.run([d_opt,d_loss_0,d_loss_k],feed_dict={input_day0:spike_0_batch,input_dayk:spike_k_batch})

            if (epoch % arg['print_every'] == 0) or (epoch == arg['n_epochs']-1):
                print("\r{}".format(epoch), "Discriminator loss_day_0:",d_loss_0_,"\Discriminator loss_day_k:",d_loss_k_)
                input_dayk_aligned_ = input_dayk_aligned.eval(feed_dict={input_dayk:spike_dayk})
                emg_dayk_aligned    = emg_hat.eval(feed_dict={spike:input_dayk_aligned_})
                emg_k_              = emg_hat.eval(feed_dict={spike:spike_dayk})
                print("EMG non-aligned VAF:", vaf(emg_dayk,emg_k_),
                      "\tEMG aligned VAF:", vaf(emg_dayk,emg_dayk_aligned),
                     "\tEMG aligned r: ", np.corrcoef(emg_dayk.flatten(), emg_dayk_aligned.flatten())[0, 1])
                
                plt.figure()
                plt.plot(emg_dayk[500:1000, :] + np.asarray([0, 400]), color = 'k')
                plt.plot(emg_dayk_aligned[500:1000, :] + np.asarray([0, 400]), color = 'r' )
                plt.show()
                
        input_test_aligned_ = input_dayk_aligned.eval(feed_dict={input_dayk:spike_test})
        emg_test_aligned    = emg_hat.eval(feed_dict={spike:input_test_aligned_})
        
    return emg_test_aligned



  