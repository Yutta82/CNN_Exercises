import argparse

import tensorflow as tf
from keras.models import load_model
from keras.src.callbacks import ModelCheckpoint

from Neuronal_Structural_Segmentation.utils.augmentation import Augmentation
from Neuronal_Structural_Segmentation.utils.model import *


def build_args():
    # 构造参数解析
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--train', required=False, default='dataset/train/', help='path to train data set')
    ap.add_argument('-t', '--test', required=False, default='dataset/test/', help='path to test data set')
    ap.add_argument('-s', '--steps', required=False, type=int, default=10, help='steps per epoch for train')
    ap.add_argument('-e', '--epochs', required=False, type=int, default=5, help='epochs for train model')
    args = vars(ap.parse_args())
    return args


def train():
    """
    模型训练
    :return: 0
    """
    args = build_args()

    strategy = tf.distribute.MirroredStrategy()
    data_gen_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    aug = Augmentation()
    generator = aug.train_generator(batch_size=2,
                                    train_path=['train'],
                                    images_folder='images',
                                    masks_folder='ground_truth',
                                    aug_dict=data_gen_args,
                                    save_to_dir=None)

    with strategy.scope():
        model = u_net()
    model_checkpoint = ModelCheckpoint(filepath='./check/final_model.keras',
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=True)
    model.fit(generator,
              steps_per_epoch=args['steps'],
              epochs=args['epochs'],
              callbacks=[model_checkpoint])


def predict():
    args = build_args()
    aug = Augmentation()
    model = load_model('./checkpoint/final_model.keras')

    test_generator_ = aug.test_generator(args['test'])
    results = model.predict(test_generator_, 30, verbose=1)
    aug.save_result(args['test'], results)
