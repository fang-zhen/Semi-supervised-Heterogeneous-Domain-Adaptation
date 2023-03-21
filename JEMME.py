import tensorflow as tf
import utils

class JEMME(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.ds = config['ds']
        self.dt = config['dt']
        self.ns = config['ns']
        self.nl = config['nl']
        self.nu = config['nu']
        self.class_number = config['class_number']
        self.beta = config['beta']
        self.tau = config['tau']
        self.d = config['d']
        self.nt = self.nl+self.nu
        self.startk = config['startk']
        self.create_model()


    def create_model(self):
        #-----------------------------------------#
        with tf.name_scope('inputs'):
            self.input_xs = tf.placeholder(tf.float32, [None, self.ds], name='input_xs')
            self.input_ys = tf.placeholder(tf.int32, [None, self.class_number], name='input_ys')
            self.input_xl = tf.placeholder(tf.float32, [None, self.dt], name='input_xl')
            self.input_yl = tf.placeholder(tf.int32, [None, self.class_number], name='input_yl')
            self.input_xu = tf.placeholder(tf.float32, [None, self.dt], name='input_xu')
            self.input_yu = tf.placeholder(tf.int32, [None, self.class_number], name='input_yu')
            self.learning_rate = tf.placeholder(tf.float32, [], name='lr')
            self.t = tf.placeholder(tf.float32, [], name='t') # the iter number
            self.T1 = tf.placeholder(tf.float32, [], name='T1')
            self.T2 = tf.placeholder(tf.float32, [], name='T2')
            self.input_xt = tf.concat([self.input_xl, self.input_xu], 0, name='input_xt')
            self.input_ya = tf.concat([self.input_ys, self.input_yl], 0, name='input_ya')
            self.SelectK = tf.cast(tf.reduce_min([(self.t+self.startk)/self.T2 * self.nu,self.nu]),tf.int32) # 500 for sentiment 100 for normal
        #------------------------------------------#
        # set the number of each layer
        s_h = int((self.ds+self.d)/2) # 2 layers, int((self.ds+self.d)/2) for normal
        #----------------------------#
        # set the number of each layer
        t_h = int((self.dt+self.d)/2) # 2 layers int((self.dt+self.d)/2) for normal
        #------------------------------------------#
        # set the parameters of each layer
        n_w = {
           's_w1': tf.Variable(tf.truncated_normal([self.ds, s_h], stddev=0.01)),
           's_w2': tf.Variable(tf.truncated_normal([s_h, self.d], stddev=0.01)),
            #-----------------------------------------------------------------#
            't_w1': tf.Variable(tf.truncated_normal([self.dt, t_h], stddev=0.01)),
            't_w2': tf.Variable(tf.truncated_normal([t_h, self.d], stddev=0.01)),
        }
        n_b = {
            's_b1': tf.Variable(tf.truncated_normal([s_h], stddev=0.01)),
            's_b2': tf.Variable(tf.truncated_normal([self.d], stddev=0.01)),
            #-----------------------------------------------------------#
            't_b1': tf.Variable(tf.truncated_normal([t_h], stddev=0.01)),
            't_b2': tf.Variable(tf.truncated_normal([self.d], stddev=0.01)),
        }
        n_w1 = {
            's_w1': tf.Variable(tf.truncated_normal([self.ds, s_h], stddev=0.02)),
            's_w2': tf.Variable(tf.truncated_normal([s_h, self.d], stddev=0.02)),
            # -----------------------------------------------------------------#
            't_w1': tf.Variable(tf.truncated_normal([self.dt, t_h], stddev=0.02)),
            't_w2': tf.Variable(tf.truncated_normal([t_h, self.d], stddev=0.02)),
        }
        n_b1 = {
            's_b1': tf.Variable(tf.truncated_normal([s_h], stddev=0.02)),
            's_b2': tf.Variable(tf.truncated_normal([self.d], stddev=0.02)),
            # -----------------------------------------------------------#
            't_b1': tf.Variable(tf.truncated_normal([t_h], stddev=0.02)),
            't_b2': tf.Variable(tf.truncated_normal([self.d], stddev=0.02)),
        }
        #------------------------------------------#
        # build projection network phi_s(Xs)
        self.projection_xs = utils.build_s(self.input_xs, n_w, n_b, tf.nn.leaky_relu)
        # build projection network phi_t(Xt)
        self.projection_xt = utils.build_t(self.input_xt, n_w, n_b, tf.nn.leaky_relu)
        self.projection_xl = tf.slice(self.projection_xt, [0, 0], [self.nl, -1])
        self.projection_xu = tf.slice(self.projection_xt, [self.nl, 0], [self.nu, -1])

        self.all_data = tf.concat([self.projection_xs, self.projection_xt], 0)


        #------------------------------------------#
        # set the parameters of 1st classifier layer
        f_w = tf.Variable(tf.truncated_normal([self.d, self.class_number], stddev=0.01))
        f_b = tf.Variable(tf.truncated_normal([self.class_number], stddev=0.01))
        self.f_x_logits = utils.add_layer(self.all_data, f_w, f_b, tf.nn.leaky_relu)
        self.f_xa_logits = tf.slice(self.f_x_logits, [0, 0], [self.ns+self.nl, -1]) # extract f_xa_logists from f_x_logists, xa is all labeled data
        self.f_xu_logits = tf.slice(self.f_x_logits, [self.ns+self.nl, 0], [self.nu, -1]) # extract f_xu_logists from f_x_logists
        self.pseudo_yu = tf.nn.softmax(self.f_xu_logits)
        Top_index_yu = tf.transpose([tf.nn.top_k(tf.reduce_max(self.pseudo_yu,1),k=self.SelectK)[1]],perm=[1, 0])

        # set the parameters of 2nd classifier layer
        self.all_data1 = self.projection_xt
        f_w1 = tf.Variable(tf.truncated_normal([self.d, self.class_number], stddev=0.02))
        f_b1 = tf.Variable(tf.truncated_normal([self.class_number], stddev=0.02))
        self.f_x_logits1 = utils.add_layer(self.all_data1, f_w1, f_b1, tf.nn.leaky_relu)

        self.f_xl_logits1 = tf.slice(self.f_x_logits1, [0, 0], [self.nl, -1])  # extract f_xa1_logists from f_x_logists1, xa is all labeled target data
        self.f_xu_logits1 = tf.slice(self.f_x_logits1, [self.nl, 0],[self.nu, -1])  # extract f_xu_logists1 from f_x_logists1
        self.pseudo_yu1 = tf.nn.softmax(self.f_xu_logits1)
        Top_index_yu1 = tf.transpose([tf.nn.top_k(tf.reduce_max(self.pseudo_yu1, 1), k=self.SelectK)[1]],perm=[1, 0])
        #------------------------------------------#
        with tf.name_scope('loss_f_xa'):
            self.loss_f_xa = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_ya, logits=self.f_xa_logits))
            # self.loss_f_xa1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_ya, logits=self.f_xa_logits1))
            self.loss_f_xa1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_yl, logits=self.f_xl_logits1))
        #------------------------------------------#

        # in this paper, we use this weighting scheme
        self.t0 = (self.t) / (self.T2)

        # MMD loss: computer_mmd_loss(xs, xl, xu, ys, yl, pseudo_yu, alpha, class_number, dt)
        # turn on crossing
        with tf.name_scope('mmd_loss'):
            self.margin_loss, self.conditional_loss, self.mmd_loss = utils.compute_mmd_loss(self.projection_xs,
                                                                                            self.projection_xl,
                                                                                            tf.gather_nd(self.projection_xu,Top_index_yu1),
                                                                                            self.input_ys,
                                                                                            self.input_yl,
                                                                                            tf.gather_nd(self.pseudo_yu1,Top_index_yu1),
                                                                                            self.class_number, self.SelectK,
                                                                                            self.t0)
        with tf.name_scope('mmd_loss1'):
            self.margin_loss1, self.conditional_loss1, self.mmd_loss1 = utils.compute_mmd_loss_T(
                self.projection_xl, tf.gather_nd(self.projection_xu,Top_index_yu), self.input_yl, tf.gather_nd(self.pseudo_yu,Top_index_yu), self.class_number, self.SelectK,
                self.t0)


        #------------------------------------------#
        # reguralization term

        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(n_w['s_w1']))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(n_w['s_w2']))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(n_w['t_w1']))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(n_w['t_w2']))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(f_w))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(f_w1))
        self.reg = tf.add_n(tf.get_collection("loss"))
        #------------------------------------------#

        with tf.name_scope('total_loss'):
            # Hyperparameter setting
            hyper = [1, 1, 20, 0.5] # 1, 1, 20, 0.5 for CLEF-DA; 1, 1, 20, 1 for R21457; 1, 1, 0.5, 0.5 for WIKI; 10, 1, 5, 0.5 for S2D; 25, 1, 15, 0.5 for D2S; 1 (5), 1, 20, 1 for CIFAR10; 1, 1, 2, 1 for CIFAR100, 1,1,1,1 for FOOD
            self.total_loss = self.loss_f_xa  \
                              + hyper[0] * self.beta*self.mmd_loss \
                              + hyper[1] * self.reg + hyper[2] * self.beta * self.mmd_loss1 \
                              + hyper[3] * self.loss_f_xa1

        #------------------------------------------#
        # the accuracy of xa
        pred_xa = tf.nn.softmax(self.f_xa_logits)
        correct_pred_xa = tf.equal(tf.argmax(self.input_ya,1), tf.argmax(pred_xa,1))
        self.xa_acc = tf.reduce_mean(tf.cast(correct_pred_xa, tf.float32))
        #------------------------------------------#
        self.f_xs_logits = tf.slice(self.f_xa_logits, [0, 0], [self.ns, -1]) # extract f_xs_logists from f_xa_logists
        self.f_xl_logits = tf.slice(self.f_xa_logits, [self.ns, 0], [self.nl, -1]) # extract f_xl_logists from f_xa_logists
        # the accuracy of xs
        pred_xs = tf.nn.softmax(self.f_xs_logits)
        correct_pred_xs = tf.equal(tf.argmax(self.input_ys,1), tf.argmax(pred_xs,1))
        self.xs_acc = tf.reduce_mean(tf.cast(correct_pred_xs, tf.float32))
        # the accuracy of xl
        pred_xl = tf.nn.softmax(self.f_xl_logits1)
        correct_pred_xl = tf.equal(tf.argmax(self.input_yl,1), tf.argmax(pred_xl,1))
        self.xl_acc = tf.reduce_mean(tf.cast(correct_pred_xl, tf.float32))
        # the accuracy of xu
        correct_pred_xu = tf.equal(tf.argmax(self.input_yu,1), tf.argmax(self.pseudo_yu1,1))
        self.xu_acc = tf.reduce_mean(tf.cast(correct_pred_xu, tf.float32))
        #------------------------------------------#
        dynamic_lr = self.learning_rate
        self.train_step = tf.train.AdamOptimizer(dynamic_lr).minimize(self.total_loss)

        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess.run(init)