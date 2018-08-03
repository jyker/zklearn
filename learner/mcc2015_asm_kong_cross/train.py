import os
import time
import keras
import numpy as np
from zk.learn.base import CONFIG
from zk.learn.model import ImageBasedMalNet, ImageBasedMalNet_default_config
from zk.dataset.dataset import DataSet
from zk.dataset.partitioner import CrossValidateResamplePartitioner
from zk.dataset.data_column import NestNPColumns
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.utils import class_weight
from typing import Dict, List


# ============================================================================
#                              config
# ============================================================================
def make_config(tag):
    CK = CONFIG.KEYS
    config = {}
    if tag == 'asm_kong_1_6400':
        config[CK.INPUT_SHAPE] = (6400, 64, 1)
        config[CK.KERNEL_SIZE] = (8, 2)
        config[CK.STRIDES] = (2, 1)
        config[CK.POOL_STRIDE] = (2, 1)
        config[CK.BATCHSIZE] = 64

    return config

# ============================================================================
#                            dataset_generator
# ============================================================================
def get_compile_model(cfg):
    CK = CONFIG.KEYS
    model = ImageBasedMalNet(cfg)
    model.summary()
    model = model.compile(
        cfg[CK.OPTIMIZER],
        cfg[CK.LOSS],
        cfg[CK.METRICS],
        cfg[CK.GPUS]
    )
    model.summary()

    return model


def get_columns(datapath):
    return NestNPColumns(datapath)

def get_partition(nb_blocks, in_blocks, resampling=None):
    return CrossValidateResamplePartitioner(nb_blocks, in_blocks, resampling)

def make_dataset(datapath, nb_blocks, in_blocks, batch_size,
                 shape, nb_class, resampling=None):
    return DataSet(get_columns(datapath),
                   get_partition(nb_blocks, in_blocks, resampling),
                   batch_size,
                   shape,
                   nb_class)

def get_dataset(cfg, train_valid):
    CK = CONFIG.KEYS
    if not cfg[CK.IS_RESAMPLE]:
        resample_target = None
    else:
        resample_target = cfg[CK.RESAMPLE_TARGET]

    if train_valid == 0:
        return make_dataset(
            cfg[CK.DATA_PATH],
            cfg[CK.NB_BLOCKS],
            cfg[CK.TRAIN_BLOCKS],
            cfg[CK.BATCHSIZE],
            cfg[CK.INPUT_SHAPE],
            cfg[CK.NB_CLASS],
            cfg[CK.RESAMPLE_TARGET])
    else:
        return make_dataset(
            cfg[CK.DATA_PATH],
            cfg[CK.NB_BLOCKS],
            cfg[CK.VALID_BLOCKS],
            cfg[CK.BATCHSIZE],
            cfg[CK.INPUT_SHAPE],
            cfg[CK.NB_CLASS])
 
  
# ============================================================================
#                                   trian 
# ============================================================================
def train(cfg, tag):
    CK = CONFIG.KEYS
    # Helper: Save the model.
    timestamp1 = time.strftime("%Y-%m-%d-%H", time.localtime())
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(cfg[CK.CURRENT_PATH], 'saved', tag + str(cfg[CK.VALID_BLOCKS]) +'-{epoch:03d}-{val_loss:.3f}.h5'),
        verbose=1,
        monitor="val_loss",
        mode="min",
        save_best_only=True)
    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=os.path.join(cfg[CK.CURRENT_PATH], 'logs', "mcc2015"))
    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=50,
        mode="min")
    # Helper: Save results.
    timestamp2 = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    csv_logger = CSVLogger(os.path.join(cfg[CK.CURRENT_PATH], 'logs', tag + str(cfg[CK.VALID_BLOCKS]) +'.log'))
    
    model = get_compile_model(cfg)
    train_gen = get_dataset(cfg, train_valid=0)
    valid_gen = get_dataset(cfg, train_valid=1)

    # calculate class_weight
    c_w = class_weight.compute_class_weight('balanced', np.unique(train_gen.labels), train_gen.labels)
    print(c_w)

    # time start train
    start_time = time.time()
    # fit
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=len(train_gen),
        epochs=cfg[CK.EPOCH],
        verbose=1,
        callbacks=[tensorboard, early_stopper, csv_logger, checkpointer],
        validation_data=valid_gen,
        validation_steps=len(valid_gen),
        shuffle=True,
        class_weight=c_w,
        workers=64)
    # time end train
    end_time = time.time()
    with open('logs/time_cost.log', 'a') as f:
        line = tag + "10-fold at v-{} t-{} time cost {} h".format(
            cfg[CK.VALID_BLOCKS],
            cfg[CK.TRAIN_BLOCKS],
            (end_time - start_time) / 3600
        )
        f.write(line + os.linesep)

if __name__ == "__main__":
    # keras.backend.set_image_data_format("channels_first")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    
    # init_config
    CK = CONFIG.KEYS
    cfg = ImageBasedMalNet_default_config()
    # path
    cfg[CK.CURRENT_PATH] = os.path.dirname(__file__)
    cfg[CK.HOME_PATH] = os.environ['HOME']
    cfg[CK.MAIN_PATH] = os.path.join(cfg[CK.HOME_PATH], 'DataSet/kaggle/MCC2015')

    # tag_list
    main_path = cfg[CK.MAIN_PATH]
    tag_list = [
        'asm_kong_1_6400',
    ]
    dataset = [os.path.join(main_path, i) for i in tag_list]

    for path, tag in zip(dataset, tag_list):    # channel-test
        cfg[CK.DATA_PATH] = path
        for i in range(1,10): # 10-folder
            a = set(range(10))
            b = set([i])
            print("valid block: {}".format(b))
            print("train block: {}".format(a-b))
            cfg[CK.VALID_BLOCKS] = list(b)
            cfg[CK.TRAIN_BLOCKS] = list(a - b)
            cfg.update(make_config(tag))
            train(cfg, tag)