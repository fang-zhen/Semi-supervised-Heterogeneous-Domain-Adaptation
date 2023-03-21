import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import multiprocessing
import scipy.io as sio  # read .mat files
from sklearn import preprocessing  # Normalization data
import numpy as np
import add_dependencies as ad  # add some dependencies
from run_J import run_J


# -----------------------------------------#
# This is the code for JMEA in the paper "Semi-supervised Heterogeneous Domain Adaptation: Theory and Algorithms",
# written by Zhen Fang, Jie Lu, Feng Liu, Guangquan Zhang,
# published in IEEE-TPAMI, 2022.
# If you have any question, please email zhen.fang@uts.edu.au or fengliu.ml@gmail.com.
# We ackownledge the code provided by STN's authors, and our code is based on the code of STN.
# The python version is 3.7.3, and the tensorflow is 1.14.0. We also test 1.15.0 version but fail.
# So we recommend running the code under the tensorflow 1.14.0.
# Please refer to add_dependencies.py to set up the path of datasets.
# -----------------------------------------#


# read mat files
# -----------------------------------------#
# imageCLEFRV
source_exp = [ad.S_II_R2V_new, ad.S_IP_R2V_new, ad.S_IB_R2V_new, ad.S_IC_R2V_new, ad.S_PI_R2V_new, ad.S_PP_R2V_new, ad.S_PB_R2V_new, ad.S_PC_R2V_new, ad.S_BI_R2V_new, ad.S_BP_R2V_new, ad.S_BB_R2V_new, ad.S_BC_R2V_new, ad.S_CI_R2V_new, ad.S_CP_R2V_new, ad.S_CB_R2V_new, ad.S_CC_R2V_new]
target_exp = [ad.T_II_R2V_new, ad.T_IP_R2V_new, ad.T_IB_R2V_new, ad.T_IC_R2V_new, ad.T_PI_R2V_new, ad.T_PP_R2V_new, ad.T_PB_R2V_new, ad.T_PC_R2V_new, ad.T_BI_R2V_new, ad.T_BP_R2V_new, ad.T_BB_R2V_new, ad.T_BC_R2V_new, ad.T_CI_R2V_new, ad.T_CP_R2V_new, ad.T_CB_R2V_new, ad.T_CC_R2V_new]
results_name = 'imageCLEF_RV_new'
iter = 20
# imageCLEFVR
# source_exp = [ad.S_II_V2R_new, ad.S_IP_V2R_new, ad.S_IB_V2R_new, ad.S_IC_V2R_new, ad.S_PI_V2R_new, ad.S_PP_V2R_new, ad.S_PB_V2R_new, ad.S_PC_V2R_new, ad.S_BI_V2R_new, ad.S_BP_V2R_new, ad.S_BB_V2R_new, ad.S_BC_V2R_new, ad.S_CI_V2R_new, ad.S_CP_V2R_new, ad.S_CB_V2R_new, ad.S_CC_V2R_new]
# target_exp = [ad.T_II_V2R_new, ad.T_IP_V2R_new, ad.T_IB_V2R_new, ad.T_IC_V2R_new, ad.T_PI_V2R_new, ad.T_PP_V2R_new, ad.T_PB_V2R_new, ad.T_PC_V2R_new, ad.T_BI_V2R_new, ad.T_BP_V2R_new, ad.T_BB_V2R_new, ad.T_BC_V2R_new, ad.T_CI_V2R_new, ad.T_CP_V2R_new, ad.T_CB_V2R_new, ad.T_CC_V2R_new]
# results_name = 'imageCLEF_VR_new'
# iter = 20
# R21457
# source_exp = [ad.SOPE, ad.SOPE, ad.SOPL, ad.SOPL, ad.SPPL, ad.SPPL]
# target_exp = [ad.TOPL, ad.TPPL, ad.TOPE, ad.TPPL, ad.TOPE, ad.TOPL]
# results_name = 'Deep_R21457'
# iter = 20
# office_S2D
# source_exp = [ad.SAS, ad.SAS, ad.SAS, ad.SAS, ad.SWS, ad.SWS, ad.SWS, ad.SWS, ad.SCS, ad.SCS, ad.SCS, ad.SCS]
# target_exp = [ad.TAD, ad.TWD, ad.TDD, ad.TCD, ad.TAD, ad.TWD, ad.TDD, ad.TCD, ad.TAD, ad.TWD, ad.TDD, ad.TCD]
# results_name = 'Deep_office_S2D'
# iter = 20
# office_D2S
# source_exp = [ad.SAD, ad.SAD, ad.SAD, ad.SAD, ad.SWD, ad.SWD, ad.SWD, ad.SWD, ad.SCD, ad.SCD, ad.SCD, ad.SCD]
# target_exp = [ad.TAS, ad.TWS, ad.TDS, ad.TCS, ad.TAS, ad.TWS, ad.TDS, ad.TCS, ad.TAS, ad.TWS, ad.TDS, ad.TCS]
# results_name = 'Deep_office_D2S'
# iter = 20

# BT_1
# source_exp = [ad.S50_8_1,  ad.S101_8_1, ad.S50_59_1, ad.S101_59_1]
# target_exp = [ad.T101_C10_1,  ad.T50_C10_1, ad.T101_C100_1, ad.T50_C100_1]
# results_name = 'Deep_BT_1'
# iter = 10
# source_exp = [ad.S50_8_1, ad.S101_8_1]
# target_exp = [ad.T101_C10_1, ad.T50_C10_1]
# results_name = 'Deep_BT_1'
# iter = 10
# BT_3
# source_exp = [ad.S50_8_3, ad.S101_8_3, ad.S50_59_3, ad.S101_59_3]
# target_exp = [ad.T101_C10_3,  ad.T50_C10_3, ad.T101_C100_3, ad.T50_C100_3]
# results_name = 'Deep_BT_3'
# iter = 10
# BT_5
# source_exp = [ad.S50_8_5, ad.S101_8_5, ad.S50_59_5, ad.S101_59_5]
# target_exp = [ad.T101_C10_5,  ad.T50_C10_5, ad.T101_C100_5, ad.T50_C100_5]
# results_name = 'Deep_BT_5'
# iter = 10

# WIKI_5
# source_exp = [ad.SImg_5, ad.STxt_5]
# target_exp = [ad.TTxt_5, ad.TImg_5]
# results_name = 'Deep_WIKI_5'
# iter = 10

# FOOD_1
# source_exp = [ad.SImg_1_food, ad.STxt_1_food]
# target_exp = [ad.TTxt_1_food, ad.TImg_1_food]
# results_name = 'Deep_food_1'
# iter = 10
# FOOD_3
# source_exp = [ad.SImg_3_food, ad.STxt_3_food]
# target_exp = [ad.TTxt_3_food, ad.TImg_3_food]
# results_name = 'Deep_food_3'
# iter = 10
# FOOD_5
# source_exp = [ad.STxt_5_food, ad.SImg_5_food]
# target_exp = [ad.TImg_5_food, ad.TTxt_5_food]
# results_name = 'Deep_food_5'
# iter = 10



Task_num = range(len(source_exp))
# ===========================================================#
# --------------------------------------------------------#
if __name__ == "__main__":
    # parameters
    tf.set_random_seed(1234)
    beta = 0.001  # control MMD loss 0.001 (good for cifar10 not for cifar100 (0.00001))
    lr = 0.001  # learning rate 0.001 for almost cases
    T = 300  # the total iter number 300 for normal
    T1 = 0
    T2 = T
    d = 256# the dimension of common subspace 256 for almost cases
    tau = 0.001  # control regularization term, cannot be an integer
    startk = 100  # 100 for most cases, 150 for S2D
    # ===========================================================#
    length = len(source_exp)
    acc_jmea_list = multiprocessing.Manager().list()
    acc_jmea = np.zeros((iter, length))

    for i in Task_num:
        acc = 0
        print("Source domain: " + source_exp[i])
        print("Target domain: " + target_exp[i])
        # load data
        source = sio.loadmat(source_exp[i])
        target = sio.loadmat(target_exp[i])
        for j in range(0, iter):
            print("====================iteration[" + str(j + 1) + "]====================")
            # -------------------------------------#
            if results_name == 'Deep_Cross-lin':
                xl = target['training_features'][j, 0]  # read labeled target data
                # xl = preprocessing.normalize(xl, norm='l2')
                xl_label = target['training_labels'][j, 0] - 1  # read labeled target data labels, form 0 start

                xu = target['testing_features'][j, 0]  # read unlabeled target data
                # xu = preprocessing.normalize(xu, norm='l2')
                xu_label = target['testing_labels'][j, 0] - 1  # read unlabeled target data labels, form 0 start

                xs = source['source_features'][j, 0]  # read source data
                xs_label = source['source_labels'][j, 0] - 1  # read source data labels, form 0 start
                # xs = preprocessing.normalize(xs, norm='l2')
            elif results_name == 'Deep_Sentiment':
                xl = np.transpose(target['training_features1'][0, j], (1, 0))  # read labeled target data
                # xl = preprocessing.normalize(xl, norm='l2')
                xl_label = target['training_labels1'][0, j] - 1  # read labeled target data labels, form 0 start

                xu = np.transpose(target['testing_features1'][0, j], (1, 0))  # read unlabeled target data
                # xu = preprocessing.normalize(xu, norm='l2')
                xu_label = target['testing_labels1'][0, j] - 1  # read unlabeled target data labels, form 0 start

                xs = np.transpose(source['source_features1'][0, j], (1, 0))  # read source data
                xs_label = source['source_labels1'][0, j] - 1  # read source data labels, form 0 start
                # xs = preprocessing.normalize(xs, norm='l2')
            else:
                xl = target['training_features'][0, j]  # read labeled target data
                # xl = preprocessing.normalize(xl, norm='l2')
                xl_label = target['training_labels'][0, j] - 1  # read labeled target data labels, form 0 start
                # print(xl_label.max())

                xu = target['testing_features'][0, j]  # read unlabeled target data
                # xu = preprocessing.normalize(xu, norm='l2')
                xu_label = target['testing_labels'][0, j] - 1  # read unlabeled target data labels, form 0 start
                # print(xu_label.max())

                xs = source['source_features'][0, j]  # read source data
                xs_label = source['source_labels'][0, j] - 1  # read source data labels, form 0 start
                # print(xs_label.max())

            ns, ds = xs.shape
            nl, dt = xl.shape
            print(nl)
            nu, _ = xu.shape

            # norm for whole data
            # xs = preprocessing.normalize(xs, norm='l2', axis=0)
            # xt = np.concatenate([xl,xu],0)
            # xt = preprocessing.normalize(xt, norm='l2',axis=0)
            # xl = xt[:nl,:]
            # xu = xt[nl:,:]

            nt = nl + nu
            class_number = len(np.unique(xl_label));

            yl = tf.one_hot(xl_label, class_number, on_value=1, off_value=0)  # one-hot label
            yl = tf.reshape(yl, [nl, class_number])  # shape: nl, class_number
            yu = tf.one_hot(xu_label, class_number, on_value=1, off_value=0)  # one-hot label
            yu = tf.reshape(yu, [nu, class_number])  # shape: nu, class_number
            ys = tf.one_hot(xs_label, class_number, on_value=1, off_value=0)  # one-hot label
            ys = tf.reshape(ys, [ns, class_number])  # shape: ns, class_number
            config = {'ds': ds, 'dt': dt, 'ns': ns, 'nl': nl, 'nu': nu, 'class_number': class_number, 'beta': beta,
                      'tau': tau, 'd': d, 'startk': startk}
            config_data = {'ys': ys, 'yl': yl, 'yu': yu, 'xs': xs, 'xl': xl,
                           'xu': xu, 'lr': lr, 'T': T, 'T1': T1, 'T2': T2,
                           'xs_label': xs_label, 'xl_label': xl_label, 'xu_label': xu_label}
            tf.set_random_seed(1234)
            # run_stn(acc_jmea_list, config, config_data)
            p = multiprocessing.Process(target=run_J, args=(acc_jmea_list, config, config_data))
            p.start()
            p.join()
            print(acc_jmea_list)
            acc_jmea[j][i] = acc_jmea_list[i * iter + j]
            print(np.mean(acc_jmea, axis=0), np.std(acc_jmea, axis=0))
    print(np.mean(acc_jmea, axis=0), np.std(acc_jmea, axis=0))
    np.savetxt('results/' + results_name + '_JEMME_test.csv', acc_jmea, delimiter=',')