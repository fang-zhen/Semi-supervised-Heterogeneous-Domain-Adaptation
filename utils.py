import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
#=========================================================================#
#-----------------------------------------#
# define the build_s network
def build_s(x, weights, biases, activation_function):
    layer_1 = activation_function(tf.add(tf.matmul(x, weights['s_w1']), biases['s_b1']))
    #---------------------#
    layer_2 = activation_function(tf.add(tf.matmul(layer_1, weights['s_w2']), biases['s_b2']))
    #---------------------#
    layer_2 = tf.nn.l2_normalize(layer_2, dim = 1)
    return layer_2
#-----------------------------------------#
# define the build_t network
def build_t(x, weights, biases, activation_function):
    layer_1 = activation_function(tf.add(tf.matmul(x, weights['t_w1']), biases['t_b1']))
    #---------------------#
    layer_2 = activation_function(tf.add(tf.matmul(layer_1, weights['t_w2']), biases['t_b2']))
    #---------------------#
    layer_2 = tf.nn.l2_normalize(layer_2, dim = 1)
    return layer_2
#-----------------------------------------#
# define full-connect layer
def add_layer(x, weights, biases, activation_function):
    Wx_plus_b = tf.add(tf.matmul(x, weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs 
#=========================================================================#
#-----------------------------------------#
# computer MMD loss
def compute_mmd_loss(xs, xl, xu, ys, yl, pseudo_yu, class_number, nu, t0):
    xt = tf.concat([xl, xu], 0)
    d = tf.shape(xt)[1]
    mean_xs = tf.reduce_mean(xs, 0)
    mean_xt = tf.reduce_mean(xt, 0)
    margin_loss = tf.reduce_sum(tf.square(mean_xs - mean_xt))
    #---------------------------------------#
    temp = tf.ones((nu,1), dtype=tf.float32)*t0
    delta = tf.reshape(temp, [-1,1])
    #---------------------------------------#
    xs_label = tf.argmax(ys,1)
    xl_label = tf.argmax(yl,1)
    conditional_loss = tf.constant(0., tf.float32)
    for k in range(class_number):
        index_xs_k = tf.cast(tf.equal(xs_label,k), tf.int32)
        index_xl_k = tf.cast(tf.equal(xl_label,k), tf.int32)
        xs_k = tf.dynamic_partition(xs,index_xs_k,2)[1]
        xl_k = tf.dynamic_partition(xl,index_xl_k,2)[1]
        #---------------------------------#
        mean_xs_k = tf.reduce_mean(xs_k, 0)
        #---------------------------------#
        sum_xl_k = tf.reduce_sum(xl_k, 0)
        weight = tf.multiply(tf.reshape(pseudo_yu[:,k], [-1,1]), delta)
        weight_xu_k = tf.multiply(xu, tf.tile(weight, [1,d]))
        sum_xu_k = tf.reshape(tf.reduce_sum(weight_xu_k, 0), [1,-1])
        nl_k = tf.cast(tf.shape(xl_k)[0], tf.float32)
        nu_k = tf.reduce_sum(weight)
        mean_xt_k = (sum_xl_k+sum_xu_k)/(nl_k+nu_k)
        #---------------------------------#
        conditional_loss += tf.reduce_sum(tf.square(mean_xs_k - mean_xt_k))
    mmd_loss = margin_loss + conditional_loss
    return margin_loss, conditional_loss, mmd_loss

# computer MMD loss
def compute_mmd_loss_T(xl, xu, yl, pseudo_yu, class_number, nu, t0):
    # xt = tf.concat([xl, xu], 0)
    xt = xl
    d = tf.shape(xt)[1]
    mean_xt = tf.reduce_mean(xt, 0)
    mean_xu = tf.reduce_mean(xu, 0)
    margin_loss = tf.reduce_sum(tf.square(mean_xt - mean_xu))
    #---------------------------------------#
    temp = tf.ones((nu,1), dtype=tf.float32)*t0
    delta = tf.reshape(temp, [-1,1])
    #---------------------------------------#
    xt_label = tf.argmax(yl,1)
    xu_label = tf.argmax(pseudo_yu,1)
    conditional_loss = tf.constant(0., tf.float32)
    for k in range(class_number):
        index_xt_k = tf.cast(tf.equal(xt_label,k), tf.int32)
        index_xu_k = tf.cast(tf.equal(xu_label,k), tf.int32)
        xt_k = tf.dynamic_partition(xt,index_xt_k,2)[1]
        xu_k = tf.dynamic_partition(xu,index_xu_k,2)[1]
        #---------------------------------#
        mean_xt_k = tf.reduce_mean(xt_k, 0)
        #---------------------------------#
        weight = tf.multiply(tf.reshape(pseudo_yu[:,k], [-1,1]), delta)
        weight_xu_k = tf.multiply(xu, tf.tile(weight, [1,d]))
        sum_xu_k = tf.reshape(tf.reduce_sum(weight_xu_k, 0), [1,-1])
        nu_k = tf.reduce_sum(weight) + 1
        mean_xu_k = (sum_xu_k)/(nu_k)
        #---------------------------------#
        conditional_loss += tf.reduce_sum(tf.square(mean_xt_k - mean_xu_k))
    mmd_loss = margin_loss + conditional_loss
    return margin_loss, conditional_loss, mmd_loss
#-----------------------------------------#
def plot_embedding(x, xs_label, xl_label, xu_label):
    ns = xs_label.shape[0]
    nl = xl_label.shape[0]
    nu = xu_label.shape[0]
    xs = x[0:ns,:]
    xl = x[ns:ns+nl,:]
    xu = x[ns+nl:ns+nl+nu,:]
    plt.scatter(xs[:,0],xs[:,1],100,marker='*',c=xs_label[:,0],label='xs')
    plt.scatter(xl[:,0],xl[:,1],100,marker='o',c=xl_label[:,0],label='xl')
    plt.scatter(xu[:,0],xu[:,1],35,marker='o',c=xu_label[:,0],label='xu')
    plt.legend(loc='upper right')
#-----------------------------------------#
def plot_all_data(target_tsne, xs_label, xl_label, xu_label):
    plt.title("Target space")
    plot_embedding(target_tsne, xs_label, xl_label, xu_label)
    plt.show()
#=========================================================================#
