import tensorflow as tf
import numpy as np
from JEMME import JEMME
def run_J(acc_jmea_list,config,config_data):
    with tf.Session() as sess:
        model = JEMME(sess=sess, config=config)
        #------------------------------------------#
        ys = config_data['ys']
        yl = config_data['yl']
        yu = config_data['yu']
        xs = config_data['xs']
        xl = config_data['xl']
        xu = config_data['xu']
        lr = config_data['lr']
        T = config_data['T']
        T1 = config_data['T1']
        T2 = config_data['T2']
        ys_r,yl_r,yu_r = sess.run([ys,yl,yu])
        train_feed = {model.input_xs: xs, model.input_ys: ys_r, model.input_xl: xl, model.input_yl: yl_r,
                model.input_xu: xu, model.input_yu: yu_r, model.learning_rate: lr, model.T1: T1, model.T2: T2, model.t: 0}

        loss_stn = np.zeros((T,1))

        for t in range(T):
        #------------------------------------------#
            # training feature network
            train_feed.update({model.t: t})
            sess.run(model.train_step, feed_dict=train_feed)
            total_loss = sess.run(model.total_loss, feed_dict=train_feed)
            loss_stn[t][0] = total_loss
            if t % 100 == 0:
                print("the total_loss is: " + str(total_loss))
                print("----------------------------")
                #------------------------------------------#
                xa_acc, xs_acc, xl_acc, xu_acc = sess.run([model.xa_acc, model.xs_acc, model.xl_acc, model.xu_acc], feed_dict=train_feed) # Compute final evaluation on test data
                loss_f_xa, margin_loss, conditional_loss, mmd_loss, loss_f_xa1, margin_loss1, conditional_loss1, mmd_loss1 = \
                    sess.run([model.loss_f_xa, model.margin_loss, model.conditional_loss,
                              model.mmd_loss, model.loss_f_xa1, model.margin_loss1, model.conditional_loss1, model.mmd_loss1], feed_dict=train_feed)
                t0 = sess.run(model.t0, feed_dict=train_feed)
                print("t0 is: " + str(t0))
                print("the accuracy of f(xa) is: " + str(xa_acc))
                print("the accuracy of f(xs) is: " + str(xs_acc))
                print("the accuracy of f(xl) is: " + str(xl_acc))
                print("the accuracy of f(xu) is: " + str(xu_acc))
                print("----------------------------")
                print("the loss_f_xa is: " + str(loss_f_xa))
                print("the margin_loss is: " + str(margin_loss))
                print("the conditional_loss is: " + str(conditional_loss))
                print("the mmd_loss is: " + str(mmd_loss))
                print("----------------------------")
                print("the loss_f_xa1 is: " + str(loss_f_xa1))
                print("the margin_loss1 is: " + str(margin_loss1))
                print("the conditional_loss1 is: " + str(conditional_loss1))
                print("the mmd_loss1 is: " + str(mmd_loss1))
                print("===============================")
        xu_acc = sess.run(model.xu_acc, feed_dict=train_feed)*100 # Get the final accuracy of xu
        print("the accuracy of f(xu) is: " + str(xu_acc))

        acc_jmea_list.append(xu_acc) # record accuracy of xu