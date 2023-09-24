# SIMDATA_ROOT_DIR = f'/media/kenta/Extreme SSD/dataset/carla_VR/'
# LOGDATE = '230724/'
# LOGDIR = 'log_115111/'

CONST_SYS_DELAY = 5 # System delay: 5frame x 50ms = 250ms


'''
Body pose data
'''
POSE_DATA_ROOT_DIR = f'/media/kenta/ExtremePro/ego_exo/main/'
BIG_SEQUENCE = f'01_walk/'
SUB_SEQUENCE = f'002_walk/'
SUB_DIR = f'exo/cam01/images/'
POSE_SUB_DIR = f'processed_data/vis_contact_poses3d/cam07/rgb/'

IMAGE_DATA_ROOT_DIR = f'/media/kenta/ExtremePro/ego_exo/common/time_synced_exo'
IMAGE_DATA_TARGET_DIR = f'/{BIG_SEQUENCE}{SUB_DIR}'

'''
Sim data
'''
SIMDATA_ROOT_DIR = f'/media/kenta/ExtremePro/dataset/carla_VR/'
LOGDATE = f'230817/'

# LOGTIME: Depend on participants/scenarios
PERSON_TAG = 'vr_celine-vr'
# LOGTIME = f'131449' # vr_celine 1
LOGTIME = f'131652' # vr_celine 2
# LOGTIME = f'131838' # vr_celine 3

# PERSON_TAG = 'vr_nathaniel-vr'
# LOGTIME = f'140247' # vr_ 1
# LOGTIME = f'140349' # vr_ 2
# LOGTIME = f'140441' # vr_ 3

# PERSON_TAG = 'vr_darren-vr'
# LOGTIME = f'143631' # vr_ 1
# LOGTIME = f'143745' # vr_ 2
# LOGTIME = f'143900' # vr_ 3

# PERSON_TAG = 'vr_dan-vr'
# # LOGTIME = f'150008' # vr_ 1
# # LOGTIME = f'150105' # vr_ 2
# LOGTIME = f'150201' # vr_ 3


LOGDIR = f'log_{LOGTIME}_{PERSON_TAG}/'
SIMLOG_FNAME = f'logdata_08172023_{LOGTIME}.csv'