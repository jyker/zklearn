import os
import numpy as np
from keras.models import load_model
from zk.dataset.dataset import DataSet
from zk.dataset.partitioner import CrossValidateResamplePartitioner
from zk.dataset.data_column import NestNPColumns
from sklearn.metrics import classification_report, confusion_matrix

# ======================================================
#               Global setting
# ======================================================
# path
HOME = os.environ['HOME']
DATASETPATH = os.path.join(HOME, 'DataSet/kaggle/MCC2015/asm_L_Nara')
MODELPATH = os.path.join('saved', 'asm_L_Nara[0]-054-0.008.h5')

# crossvalid
NB_BLOCKS = 10
TEST_BLOCKS = [0]

# model
INPUT_SHAPE = (6400, 64, 1)

# test
NB_CLASS = 9
BATCHSIZE = 64
EPOCH = 1

# label
LABEL = [
    'Ramnit',
    'Lollipop',
    'Kelihos_ver3',
    'Vundo',
    'Simda',
    'Tracur',
    'Kelihos_ver1',
    'Obfuscator.ACY',
    'Gatak'
]

def get_columns(datapath):
    return NestNPColumns(datapath)

def get_partition(nb_blocks, in_blocks, resampling=None):
    return CrossValidateResamplePartitioner(nb_blocks, in_blocks, resampling)

def make_dataset(datapath, nb_blocks, in_blocks, resampling=None,
                 batch_size=BATCHSIZE, shape=INPUT_SHAPE, nb_class=NB_CLASS):
    return DataSet(get_columns(datapath),
                   get_partition(nb_blocks, in_blocks, resampling),
                   batch_size,
                   shape,
                   nb_class,
                   False)

def get_dataset():
    return make_dataset(DATASETPATH, NB_BLOCKS, TEST_BLOCKS)


def test():
    test_gen = get_dataset()
    model = load_model(MODELPATH)

    score = model.evaluate_generator(
        generator=test_gen,
        steps=len(test_gen)
    )
    print("Accuracy", score)

    # Confution Matrix and Classification Report
    Y_pred = model.predict_generator(test_gen, len(test_gen))
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_gen.labels, y_pred))
    print('Classification Report')
    print(classification_report(test_gen.labels, y_pred, target_names=LABEL))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test()
