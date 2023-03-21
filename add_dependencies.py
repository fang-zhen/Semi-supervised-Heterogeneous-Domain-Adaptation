VR_NEW = './data/VR_NEW/'
BT = './data/BT/'
R2 = './data/Return_Text_new'
FOOD = './data/FOOD/'
OR = './data/Office_Caltech10'
WK = './data/DeepWIKI/'

# SURF -> DeCAF6
# C, W -> A
SCS = OR + '/randsource_Caltech10_SURF_L10.mat'
SWS = OR + '/randsource_webcam_SURF_L10.mat'
TAD = OR + '/randtarget_amazon_DeCAF6.mat'
# A, W -> C
SAS = OR + '/randsource_amazon_SURF_L10.mat'
TCD = OR + '/randtarget_Caltech10_DeCAF6.mat'
# C, A -> W
TWD = OR + '/randtarget_webcam_DeCAF6.mat'

TDD = OR + '/randtarget_dslr_DeCAF6.mat'

# DeCAF6 -> SURF
# C, W -> A
SCD = OR + '/randsource_Caltech10_DeCAF6.mat'
SWD = OR + '/randsource_webcam_DeCAF6.mat'
TAS = OR + '/randtarget_amazon_SURF_L10.mat'
# A, W -> C
SAD = OR + '/randsource_amazon_DeCAF6.mat'
TCS = OR + '/randtarget_Caltech10_SURF_L10.mat'
# C, A -> W
TWS = OR + '/randtarget_webcam_SURF_L10.mat'
TDS = OR + '/randtarget_dslr_SURF_L10.mat'

# Reuters-21578

SOPE = R2 + '/randsource_OPES1.mat'
SOPL = R2 + '/randsource_OPLS1.mat'
SPPL = R2 + '/randsource_PPLS1.mat'
TOPE = R2 + '/randtarget_OPET1.mat'
TOPL = R2 + '/randtarget_OPLT1.mat'
TPPL = R2 + '/randtarget_OPET1.mat'


# VGG2ResNet

S_II_R2V_new = VR_NEW + 'irivs.mat'
S_IP_R2V_new = VR_NEW + 'irpvs.mat'
S_IB_R2V_new = VR_NEW + 'irbvs.mat'
S_IC_R2V_new = VR_NEW + 'ircvs.mat'

S_PI_R2V_new = VR_NEW + 'privs.mat'
S_PP_R2V_new = VR_NEW + 'prpvs.mat'
S_PB_R2V_new = VR_NEW + 'prbvs.mat'
S_PC_R2V_new = VR_NEW + 'prcvs.mat'

S_BI_R2V_new = VR_NEW + 'brivs.mat'
S_BP_R2V_new = VR_NEW + 'brpvs.mat'
S_BB_R2V_new = VR_NEW + 'brbvs.mat'
S_BC_R2V_new = VR_NEW + 'brcvs.mat'

S_CI_R2V_new = VR_NEW + 'crivs.mat'
S_CP_R2V_new = VR_NEW + 'crpvs.mat'
S_CB_R2V_new = VR_NEW + 'crbvs.mat'
S_CC_R2V_new = VR_NEW + 'crcvs.mat'

T_II_R2V_new = VR_NEW + 'irivt.mat'
T_IP_R2V_new = VR_NEW + 'irpvt.mat'
T_IB_R2V_new = VR_NEW + 'irbvt.mat'
T_IC_R2V_new = VR_NEW + 'ircvt.mat'

T_PI_R2V_new = VR_NEW + 'privt.mat'
T_PP_R2V_new = VR_NEW + 'prpvt.mat'
T_PB_R2V_new = VR_NEW + 'prbvt.mat'
T_PC_R2V_new = VR_NEW + 'prcvt.mat'

T_BI_R2V_new = VR_NEW + 'brivt.mat'
T_BP_R2V_new = VR_NEW + 'brpvt.mat'
T_BB_R2V_new = VR_NEW + 'brbvt.mat'
T_BC_R2V_new = VR_NEW + 'brcvt.mat'

T_CI_R2V_new = VR_NEW + 'crivt.mat'
T_CP_R2V_new = VR_NEW + 'crpvt.mat'
T_CB_R2V_new = VR_NEW + 'crbvt.mat'
T_CC_R2V_new = VR_NEW + 'crcvt.mat'

# ResNet2VGG

S_II_V2R_new = VR_NEW + 'ivirs.mat'
S_IP_V2R_new = VR_NEW + 'ivprs.mat'
S_IB_V2R_new = VR_NEW + 'ivbrs.mat'
S_IC_V2R_new = VR_NEW + 'ivcrs.mat'

S_PI_V2R_new = VR_NEW + 'pvirs.mat'
S_PP_V2R_new = VR_NEW + 'pvprs.mat'
S_PB_V2R_new = VR_NEW + 'pvbrs.mat'
S_PC_V2R_new = VR_NEW + 'pvcrs.mat'

S_BI_V2R_new = VR_NEW + 'bvirs.mat'
S_BP_V2R_new = VR_NEW + 'bvprs.mat'
S_BB_V2R_new = VR_NEW + 'bvbrs.mat'
S_BC_V2R_new = VR_NEW + 'bvcrs.mat'

S_CI_V2R_new = VR_NEW + 'cvirs.mat'
S_CP_V2R_new = VR_NEW + 'cvprs.mat'
S_CB_V2R_new = VR_NEW + 'cvbrs.mat'
S_CC_V2R_new = VR_NEW + 'cvcrs.mat'

T_II_V2R_new = VR_NEW + 'ivirt.mat'
T_IP_V2R_new = VR_NEW + 'ivprt.mat'
T_IB_V2R_new = VR_NEW + 'ivbrt.mat'
T_IC_V2R_new = VR_NEW + 'ivcrt.mat'

T_PI_V2R_new = VR_NEW + 'pvirt.mat'
T_PP_V2R_new = VR_NEW + 'pvprt.mat'
T_PB_V2R_new = VR_NEW + 'pvbrt.mat'
T_PC_V2R_new = VR_NEW + 'pvcrt.mat'

T_BI_V2R_new = VR_NEW + 'bvirt.mat'
T_BP_V2R_new = VR_NEW + 'bvprt.mat'
T_BB_V2R_new = VR_NEW + 'bvbrt.mat'
T_BC_V2R_new = VR_NEW + 'bvcrt.mat'

T_CI_V2R_new = VR_NEW + 'cvirt.mat'
T_CP_V2R_new = VR_NEW + 'cvprt.mat'
T_CB_V2R_new = VR_NEW + 'cvbrt.mat'
T_CC_V2R_new = VR_NEW + 'cvcrt.mat'

# BT

S50_8_1 = BT + 'Data_random_L1/Source8_50_L1.mat'
S50_59_1 = BT + 'Data_random_L1/Source59_50_L1.mat'
S101_8_1 = BT + 'Data_random_L1/Source8_101_L1.mat'
S101_59_1 = BT + 'Data_random_L1/Source59_101_L1.mat'

T50_C10_1 = BT + 'Data_random_L1/CIFAR8_50_L1.mat'
T50_C100_1 = BT + 'Data_random_L1/CIFAR59_50_L1.mat'
T101_C10_1 = BT + 'Data_random_L1/CIFAR8_101_L1.mat'
T101_C100_1 = BT + 'Data_random_L1/CIFAR59_101_L1.mat'

# BT_3

S50_8_3 = BT + 'Data_random_L3/Source8_50_L3.mat'
S50_59_3 = BT + 'Data_random_L3/Source59_50_L3.mat'
S101_8_3 = BT + 'Data_random_L3/Source8_101_L3.mat'
S101_59_3 = BT + 'Data_random_L3/Source59_101_L3.mat'

T50_C10_3 = BT + 'Data_random_L3/CIFAR8_50_L3.mat'
T50_C100_3 = BT + 'Data_random_L3/CIFAR59_50_L3.mat'
T101_C10_3 = BT + 'Data_random_L3/CIFAR8_101_L3.mat'
T101_C100_3 = BT + 'Data_random_L3/CIFAR59_101_L3.mat'

# BT_5

S50_8_5 = BT + 'Data_random_L5/Source8_50_L5.mat'
S50_59_5 = BT + 'Data_random_L5/Source59_50_L5.mat'
S101_8_5 = BT + 'Data_random_L5/Source8_101_L5.mat'
S101_59_5 = BT + 'Data_random_L5/Source59_101_L5.mat'

T50_C10_5 = BT + 'Data_random_L5/CIFAR8_50_L5.mat'
T50_C100_5 = BT + 'Data_random_L5/CIFAR59_50_L5.mat'
T101_C10_5 = BT + 'Data_random_L5/CIFAR8_101_L5.mat'
T101_C100_5 = BT + 'Data_random_L5/CIFAR59_101_L5.mat'


# WIKI_5 (we lost the data for WIKI_3 and WIKI_7)

SImg_5 = WK + 'Data_random_L5_Wiki/T_Image_BT_L5.mat'
STxt_5 = WK + 'Data_random_L5_Wiki/T_TEXT_BB_L5.mat'

TImg_5 = WK + 'Data_random_L5_Wiki/S_Image_BT_L5.mat'
TTxt_5 = WK + 'Data_random_L5_Wiki/S_TEXT_BB_L5.mat'

# FOOD_1

SImg_1_food = FOOD + 'Data_random_L1_Food/Food_S_Image_L1.mat'
STxt_1_food = FOOD + 'Data_random_L1_Food/Food_S_Text_L1.mat'

TImg_1_food = FOOD + 'Data_random_L1_Food/Food_T_Image_L1.mat'
TTxt_1_food = FOOD + 'Data_random_L1_Food/Food_T_Text_L1.mat'

# FOOD_3

SImg_3_food = FOOD + 'Data_random_L3_Food/Food_S_Image_L3.mat'
STxt_3_food = FOOD + 'Data_random_L3_Food/Food_S_Text_L3.mat'

TImg_3_food = FOOD + 'Data_random_L3_Food/Food_T_Image_L3.mat'
TTxt_3_food = FOOD + 'Data_random_L3_Food/Food_T_Text_L3.mat'

# FOOD_5

SImg_5_food = FOOD + 'Data_random_L5_Food/Food_S_Image_L5.mat'
STxt_5_food = FOOD + 'Data_random_L5_Food/Food_S_Text_L5.mat'

TImg_5_food = FOOD + 'Data_random_L5_Food/Food_T_Image_L5.mat'
TTxt_5_food = FOOD + 'Data_random_L5_Food/Food_T_Text_L5.mat'