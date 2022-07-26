import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
DATASET_PATH = os.path.join(DATA_PATH, 'covid19-ct-scans')
IMAGES_PATH = os.path.join(DATA_PATH, 'img')
CT_IMAGES_PATH = os.path.join(IMAGES_PATH, 'ct')
LUNGS_IMAGES_PATH = os.path.join(IMAGES_PATH, 'lungs')
COVID_IMAGES_PATH = os.path.join(IMAGES_PATH, 'covid')

MODEL_PATH = os.path.join(PROJECT_PATH, 'models')
LUNGS_MODEL_WEIGHTS_PATH = os.path.join(MODEL_PATH, 'lungs.h5')
COVID_MODEL_WEIGHTS_PATH = os.path.join(MODEL_PATH, 'covid.h5')
LUNGS_HISTORY_PATH = os.path.join(MODEL_PATH, 'lungs_history.csv')
COVID_HISTORY_PATH = os.path.join(MODEL_PATH, 'covid_history.csv')

IMG_SIZE = 256
MASK_THRESHOLD = 0.25
