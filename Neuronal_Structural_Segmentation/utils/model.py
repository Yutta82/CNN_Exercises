from keras import Model
from keras.layers import *
from keras.src.optimizers import Adam
from tensorflow.keras.models import *


def u_net(input_size=(256, 256, 1)):
    """
    构建 U-Net 模型。

    U-Net 网络分为收缩 (contracting) 部分和扩张 (expansive) 部分，
    收缩部分采用卷积 -> 卷积 -> 池化 -> Dropout 的组合，
    扩张部分采用转置卷积（上采样） -> 连接 -> 卷积 -> 卷积的组合，
    最终使用 1x1 卷积将特征映射为输出。

    :param pretrained_weights: 如果提供，则加载预训练的模型参数
    :param input_size: 输入图像尺寸，这里默认为 256x256 的灰度图像（通道数为 1）
    :return: 构建好的 U-Net 模型
    """

    # 输入层
    # 定义模型的输入，形状为 input_size
    inputs = Input(input_size)

    # 编码器部分（下采样）
    # Block 1
    conv1 = Conv2D(64, (3, 3), padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Activation('relu')(conv1)  # 先调用激活函数
    conv1 = BatchNormalization()(conv1)  # 再调用BatchNorm
    conv1 = Conv2D(64, (3, 3), padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), padding='same',
                   kernel_initializer='he_normal')(drop1)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), padding='same',
                   kernel_initializer='he_normal')(drop2)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.5)(pool3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), padding='same',
                   kernel_initializer='he_normal')(drop3)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.5)(pool4)

    # 最底层 Block 5
    conv5 = Conv2D(1024, (3, 3), padding='same',
                   kernel_initializer='he_normal')(drop4)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same',
                   kernel_initializer='he_normal')(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    # 解码器部分（上采样）
    # Block 6
    up6 = UpSampling2D(size=(2, 2), interpolation='bilinear')(drop5)
    # up6 = Conv2D(512, (2, 2), activation='relu', padding='same',
    #              kernel_initializer='he_normal')(up6)
    merge6 = Concatenate()([conv4, up6])
    drop6 = Dropout(0.5)(merge6)
    conv6 = Conv2D(512, (3, 3), padding='same',
                   kernel_initializer='he_normal')(drop6)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same',
                   kernel_initializer='he_normal')(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)

    # Block 7
    up7 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)
    # up7 = Conv2D(256, (2, 2), activation='relu', padding='same',
    #              kernel_initializer='he_normal')(up7)
    merge7 = Concatenate()([conv3, up7])
    drop7 = Dropout(0.5)(merge7)
    conv7 = Conv2D(256, (3, 3), padding='same',
                   kernel_initializer='he_normal')(drop7)
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same',
                   kernel_initializer='he_normal')(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7)

    # Block 8
    up8 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv7)
    # up8 = Conv2D(128, (2, 2), activation='relu', padding='same',
    #              kernel_initializer='he_normal')(up8)
    merge8 = Concatenate()([conv2, up8])
    drop8 = Dropout(0.5)(merge8)
    conv8 = Conv2D(128, (3, 3), padding='same',
                   kernel_initializer='he_normal')(drop8)
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8)

    # Block 9
    up9 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8)
    # up9 = Conv2D(64, (2, 2), activation='relu', padding='same',
    #              kernel_initializer='he_normal')(up9)
    merge9 = Concatenate()([conv1, up9])
    drop9 = Dropout(0.5)(merge9)
    conv9 = Conv2D(64, (3, 3), padding='same',
                   kernel_initializer='he_normal')(drop9)
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3, 3), padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9)

    # 输出层
    # 二分类任务（如前景/背景分割）使用sigmoid是合理的；如果是多类别语义分割输出通道数改为 num_classes，然后 softmax 作为激活函数
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(conv9)

    # 使用定义的 inputs 和 outputs 构建模型
    model = Model(inputs, outputs)

    # 编译模型，设置 Adam 优化器、二分类交叉熵损失以及 accuracy 指标
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # 输出模型结构的详细摘要信息
    model.summary()

    # 返回构建好的模型
    return model
