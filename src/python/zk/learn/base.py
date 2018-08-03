class CONFIG:
    class KEYS:
        # path
        CURRENT_PATH = 'current_path'
        HOME_PATH = 'home_path'
        MAIN_PATH = 'main_path'
        DATA_PATH = 'data_path'
        # model config
        NB_CLASS = 'nb_class'
        INPUT_SHAPE = 'input_shape'
        KERNEL_SIZE = 'kernel_size'
        STRIDES = 'strides'
        POOL_STRIDE = 'pool'
        PADDING = 'padding'
        ACTIVATION = 'activation'
        OUT_FEATURES = 'out_features'
        DROP_OUT = 'drop_out'
        # crossvalid
        NB_BLOCKS = 'nb_blocks'
        TRAIN_BLOCKS = 'train_blocks'
        VALID_BLOCKS = 'valid_blocks'
        # resample
        IS_RESAMPLE = 'is_resample'
        RESAMPLE_TARGET = 'resample_target'
        # train config
        GPUS = 'gpus'
        LEARNING_RATE = 'learning_rate'
        DECAY = 'decay'
        MOMENTUM = 'momentum'
        LOSS = 'loss'
        OPTIMIZER = 'optimizer'
        METRICS = 'metric'
        BATCHSIZE = 'batch_size'
        EPOCH = 'epoch'