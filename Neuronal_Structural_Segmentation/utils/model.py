from keras import Model
from keras.layers import *
from keras.src.optimizers import Adam
from tensorflow.keras.models import *


def u_net(pretrained_weights=None, input_size=(256, 256, 1), start_neurons=64):
    """
    构建 U-Net 模型。

    U-Net 网络分为收缩 (contracting) 部分和扩张 (expansive) 部分，
    收缩部分采用卷积 -> 卷积 -> 池化 -> Dropout 的组合，
    扩张部分采用转置卷积（上采样） -> 连接 -> 卷积 -> 卷积的组合，
    最终使用 1x1 卷积将特征映射为输出。

    :param pretrained_weights: 如果提供，则加载预训练的模型参数
    :param input_size: 输入图像尺寸，这里默认为 256x256 的灰度图像（通道数为 1）
    :param start_neurons: 第一层卷积核数量，后续层按倍数增加
    :return: 构建好的 U-Net 模型
    """

    # -------------------- 输入层 --------------------
    # 定义模型的输入，形状为 input_size
    inputs = Input(input_size)

    # -------------------- 第一层 --------------------
    # 第一层第一卷积：卷积核个数为 start_neurons * 1，卷积核大小为 3x3，激活函数为 ReLU，使用 same 填充，权重初始化为 he_normal
    conv1 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    # 对第一卷积输出进行 BatchNormalization（批归一化）
    conv1 = BatchNormalization()(conv1)
    # 第一层第二卷积：同上
    conv1 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    # 对第一层输出进行最大池化，池化窗口大小为 2x2
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 在池化结果上添加 Dropout，丢弃率为 0.25
    pool1 = Dropout(0.25)(pool1)

    # -------------------- 第二层 --------------------
    # 第二层第一卷积：卷积核个数为 start_neurons * 2
    conv2 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    # 批归一化
    conv2 = BatchNormalization()(conv2)
    # 第二层第二卷积：同上
    conv2 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    # 对第二层输出进行 2x2 最大池化
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 添加 Dropout，丢弃率为 0.5
    pool2 = Dropout(0.5)(pool2)

    # -------------------- 第三层 --------------------
    # 第三层第一卷积：卷积核个数为 start_neurons * 4
    conv3 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    # 批归一化
    conv3 = BatchNormalization()(conv3)
    # 第三层第二卷积：同上
    conv3 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    # 对第三层输出进行 2x2 最大池化
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 添加 Dropout，丢弃率为 0.5
    pool3 = Dropout(0.5)(pool3)

    # -------------------- 第四层 --------------------
    # 第四层第一卷积：卷积核个数为 start_neurons * 8
    conv4 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    # 批归一化
    conv4 = BatchNormalization()(conv4)
    # 第四层第二卷积：同上
    conv4 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    # 对第四层输出进行 2x2 最大池化
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 添加 Dropout，丢弃率为 0.5
    pool4 = Dropout(0.5)(pool4)

    # -------------------- 第五层 --------------------
    # 第五层（最深层）第一卷积：卷积核个数为 start_neurons * 16
    conv5 = Conv2D(start_neurons * 16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    # 批归一化
    conv5 = BatchNormalization()(conv5)
    # 第五层第二卷积：同上
    conv5 = Conv2D(start_neurons * 16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)

    # -------------------- 扩张部分（上采样） --------------------
    # 第四层上采样：使用转置卷积，将 conv5 的特征图上采样，卷积核个数为 start_neurons * 8
    deconv4 = Conv2DTranspose(start_neurons * 8, kernel_size=3, strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal'
                              )(conv5)
    # 将第四层卷积输出 conv4 与上采样结果进行连接（skip connection）
    uconv4 = concatenate([conv4, deconv4])
    # 对连接后的特征图添加 Dropout，丢弃率为 0.5
    uconv4 = Dropout(0.5)(uconv4)
    # 对连接结果进行卷积，卷积核个数为 start_neurons * 8
    uconv4 = Conv2D(start_neurons * 8, kernel_size=3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv4)
    # 批归一化
    uconv4 = BatchNormalization()(uconv4)
    # 再进行一次卷积操作，卷积核个数保持不变
    uconv4 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv4)

    # 第三层上采样：使用转置卷积上采样 uconv4，卷积核个数为 start_neurons * 4
    deconv3 = Conv2DTranspose(start_neurons * 4, 3, strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(uconv4)
    # 将第三层的卷积输出 conv3 与上采样结果连接
    uconv3 = concatenate([conv3, deconv3])
    # 添加 Dropout，丢弃率为 0.5
    uconv3 = Dropout(0.5)(uconv3)
    # 对连接后的特征图进行卷积操作，卷积核个数为 start_neurons * 4
    uconv3 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv3)
    # 批归一化
    uconv3 = BatchNormalization()(uconv3)
    # 再进行一次卷积操作，卷积核个数保持不变
    uconv3 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv3)

    # 第二层上采样：使用转置卷积上采样 uconv3，卷积核个数为 start_neurons * 2
    deconv2 = Conv2DTranspose(start_neurons * 2, 3, strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(uconv3)
    # 将第二层卷积输出 conv2 与上采样结果连接
    uconv2 = concatenate([conv2, deconv2])
    # 添加 Dropout，丢弃率为 0.5
    uconv2 = Dropout(0.5)(uconv2)
    # 对连接后的特征图进行卷积操作，卷积核个数为 start_neurons * 2
    uconv2 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv2)
    # 批归一化
    uconv2 = BatchNormalization()(uconv2)
    # 再进行一次卷积操作，卷积核个数保持不变
    uconv2 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv2)

    # 第一层上采样：使用转置卷积上采样 uconv2，卷积核个数为 start_neurons * 1
    deconv1 = Conv2DTranspose(start_neurons * 1, 3, strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(uconv2)
    # 将第一层卷积输出 conv1 与上采样结果连接
    uconv1 = concatenate([conv1, deconv1])
    # 添加 Dropout，丢弃率为 0.5
    uconv1 = Dropout(0.5)(uconv1)
    # 对连接后的特征图进行卷积操作，卷积核个数为 start_neurons * 1
    uconv1 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv1)
    # 批归一化
    uconv1 = BatchNormalization()(uconv1)
    # 再进行一次卷积操作，卷积核个数保持不变
    uconv1 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv1)

    # -------------------- 输出层 --------------------
    # 使用 1x1 卷积将特征图映射为单通道输出，并使用 sigmoid 激活函数输出概率
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(uconv1)

    # 使用定义的 inputs 和 outputs 构建模型
    model = Model(inputs, outputs)

    # 编译模型，设置 Adam 优化器、二分类交叉熵损失以及 accuracy 指标
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # 打印提示信息
    print("模型概述信息如下：")
    # 输出模型结构的详细摘要信息
    model.summary()

    # 如果提供了预训练权重，则加载这些权重
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    # 返回构建好的模型
    return model
