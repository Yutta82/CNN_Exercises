# 这是一个卷积神经网络的练习库
目前有两个项目：
1. 基于TensorFlow框架与[Chest X-Ray Images (Pneumonia)数据集](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)的肺炎诊断

   关于模型：
   1. 选择VGG作为baseline，针对二分类任务进行定制。既保留了VGG的深度结构，又通过现代技术提高了效率和性能。
   2. 引入深度可分离卷积(SeparableConv2D层)，提高计算效率，减少参数量。
   3. 引入批归一化(BatchNormalization层)，加速训练，稳定输出。
   4. 添加具有合理数量神经元的密集层。以更高的学习率进行训练，并试验密集层中的神经元数量。也针对网络深度进行此操作。
   5. 一旦知道了较好的深度，就开始以较低的学习率和衰减来训练网络。
   
   网络结构
   ```text
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
   ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
   │ ImageInput (InputLayer)              │ (None, 224, 224, 3)         │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv1_1 (Conv2D)                     │ (None, 224, 224, 64)        │           1,792 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv1_2 (Conv2D)                     │ (None, 224, 224, 64)        │          36,928 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ pool1 (MaxPooling2D)                 │ (None, 112, 112, 64)        │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv2_1 (SeparableConv2D)            │ (None, 112, 112, 128)       │           8,896 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv2_2 (SeparableConv2D)            │ (None, 112, 112, 128)       │          17,664 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ pool2 (MaxPooling2D)                 │ (None, 56, 56, 128)         │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv3_1 (SeparableConv2D)            │ (None, 56, 56, 256)         │          34,176 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ bn1 (BatchNormalization)             │ (None, 56, 56, 256)         │           1,024 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv3_2 (SeparableConv2D)            │ (None, 56, 56, 256)         │          68,096 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ bn2 (BatchNormalization)             │ (None, 56, 56, 256)         │           1,024 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv3_3 (SeparableConv2D)            │ (None, 56, 56, 256)         │          68,096 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ pool3 (MaxPooling2D)                 │ (None, 28, 28, 256)         │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv4_1 (SeparableConv2D)            │ (None, 28, 28, 512)         │         133,888 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ bn3 (BatchNormalization)             │ (None, 28, 28, 512)         │           2,048 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv4_2 (SeparableConv2D)            │ (None, 28, 28, 512)         │         267,264 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ bn4 (BatchNormalization)             │ (None, 28, 28, 512)         │           2,048 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ Conv4_3 (SeparableConv2D)            │ (None, 28, 28, 512)         │         267,264 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ pool4 (MaxPooling2D)                 │ (None, 14, 14, 512)         │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ flatten (Flatten)                    │ (None, 100352)              │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ fc1 (Dense)                          │ (None, 1024)                │     102,761,472 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ dropout1 (Dropout)                   │ (None, 1024)                │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ fc2 (Dense)                          │ (None, 512)                 │         524,800 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ dropout2 (Dropout)                   │ (None, 512)                 │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ fc3 (Dense)                          │ (None, 2)                   │           1,026 │
   └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 104,197,506 (397.48 MB)
    Trainable params: 104,194,434 (397.47 MB)
    Non-trainable params: 3,072 (12.00 KB)
   ```

2. 基于TensorFlow框架与[中科院自动化研究所的手写汉字数据集](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip)的手写汉字识别
   
   网络结构：
   ```text
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
   ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
   │ image_batch (InputLayer)             │ (None, 64, 64, 1)           │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ conv1 (Conv2D)                       │ (None, 64, 64, 64)          │             640 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ max_pooling2d (MaxPooling2D)         │ (None, 32, 32, 64)          │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ conv2 (Conv2D)                       │ (None, 32, 32, 128)         │          73,856 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ max_pooling2d_1 (MaxPooling2D)       │ (None, 16, 16, 128)         │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ conv3 (Conv2D)                       │ (None, 16, 16, 256)         │         295,168 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ max_pooling2d_2 (MaxPooling2D)       │ (None, 8, 8, 256)           │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ flatten (Flatten)                    │ (None, 16384)               │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ fc1 (Dense)                          │ (None, 1024)                │      16,778,240 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ dropout (Dropout)                    │ (None, 1024)                │               0 │
   ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
   │ fc2 (Dense)                          │ (None, 3755)                │       3,848,875 │
   └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 20,996,779 (80.10 MB)
    Trainable params: 20,996,779 (80.10 MB)
    Non-trainable params: 0 (0.00 B)
   ```
   
3. 基于TensorFlow框架与[ISBI Challenge: Segmentation of neuronal structures in EM stacks.](http://brainiac2.mit.edu/isbi_challenge/)的神经元结构分割
   
   关于模型：
   1. 选择u-net作为baseline

   网络结构
   ```text
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
   │ input_layer (InputLayer)      │ (None, 256, 256, 1)       │               0 │ -                          │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d (Conv2D)               │ (None, 256, 256, 64)      │             640 │ input_layer[0][0]          │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation (Activation)       │ (None, 256, 256, 64)      │               0 │ conv2d[0][0]               │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization           │ (None, 256, 256, 64)      │             256 │ activation[0][0]           │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_1 (Conv2D)             │ (None, 256, 256, 64)      │          36,928 │ batch_normalization[0][0]  │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_1 (Activation)     │ (None, 256, 256, 64)      │               0 │ conv2d_1[0][0]             │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_1         │ (None, 256, 256, 64)      │             256 │ activation_1[0][0]         │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ max_pooling2d (MaxPooling2D)  │ (None, 128, 128, 64)      │               0 │ batch_normalization_1[0][… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ dropout (Dropout)             │ (None, 128, 128, 64)      │               0 │ max_pooling2d[0][0]        │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_2 (Conv2D)             │ (None, 128, 128, 128)     │          73,856 │ dropout[0][0]              │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_2 (Activation)     │ (None, 128, 128, 128)     │               0 │ conv2d_2[0][0]             │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_2         │ (None, 128, 128, 128)     │             512 │ activation_2[0][0]         │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_3 (Conv2D)             │ (None, 128, 128, 128)     │         147,584 │ batch_normalization_2[0][… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_3 (Activation)     │ (None, 128, 128, 128)     │               0 │ conv2d_3[0][0]             │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_3         │ (None, 128, 128, 128)     │             512 │ activation_3[0][0]         │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ max_pooling2d_1               │ (None, 64, 64, 128)       │               0 │ batch_normalization_3[0][… │
   │ (MaxPooling2D)                │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ dropout_1 (Dropout)           │ (None, 64, 64, 128)       │               0 │ max_pooling2d_1[0][0]      │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_4 (Conv2D)             │ (None, 64, 64, 256)       │         295,168 │ dropout_1[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_4 (Activation)     │ (None, 64, 64, 256)       │               0 │ conv2d_4[0][0]             │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_4         │ (None, 64, 64, 256)       │           1,024 │ activation_4[0][0]         │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_5 (Conv2D)             │ (None, 64, 64, 256)       │         590,080 │ batch_normalization_4[0][… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_5 (Activation)     │ (None, 64, 64, 256)       │               0 │ conv2d_5[0][0]             │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_5         │ (None, 64, 64, 256)       │           1,024 │ activation_5[0][0]         │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ max_pooling2d_2               │ (None, 32, 32, 256)       │               0 │ batch_normalization_5[0][… │
   │ (MaxPooling2D)                │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ dropout_2 (Dropout)           │ (None, 32, 32, 256)       │               0 │ max_pooling2d_2[0][0]      │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_6 (Conv2D)             │ (None, 32, 32, 512)       │       1,180,160 │ dropout_2[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_6 (Activation)     │ (None, 32, 32, 512)       │               0 │ conv2d_6[0][0]             │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_6         │ (None, 32, 32, 512)       │           2,048 │ activation_6[0][0]         │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_7 (Conv2D)             │ (None, 32, 32, 512)       │       2,359,808 │ batch_normalization_6[0][… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_7 (Activation)     │ (None, 32, 32, 512)       │               0 │ conv2d_7[0][0]             │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_7         │ (None, 32, 32, 512)       │           2,048 │ activation_7[0][0]         │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ max_pooling2d_3               │ (None, 16, 16, 512)       │               0 │ batch_normalization_7[0][… │
   │ (MaxPooling2D)                │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ dropout_3 (Dropout)           │ (None, 16, 16, 512)       │               0 │ max_pooling2d_3[0][0]      │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_8 (Conv2D)             │ (None, 16, 16, 1024)      │       4,719,616 │ dropout_3[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_8 (Activation)     │ (None, 16, 16, 1024)      │               0 │ conv2d_8[0][0]             │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_8         │ (None, 16, 16, 1024)      │           4,096 │ activation_8[0][0]         │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_9 (Conv2D)             │ (None, 16, 16, 1024)      │       9,438,208 │ batch_normalization_8[0][… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_9 (Activation)     │ (None, 16, 16, 1024)      │               0 │ conv2d_9[0][0]             │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_9         │ (None, 16, 16, 1024)      │           4,096 │ activation_9[0][0]         │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ dropout_4 (Dropout)           │ (None, 16, 16, 1024)      │               0 │ batch_normalization_9[0][… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ up_sampling2d (UpSampling2D)  │ (None, 32, 32, 1024)      │               0 │ dropout_4[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ concatenate (Concatenate)     │ (None, 32, 32, 1536)      │               0 │ batch_normalization_7[0][… │
   │                               │                           │                 │ up_sampling2d[0][0]        │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ dropout_5 (Dropout)           │ (None, 32, 32, 1536)      │               0 │ concatenate[0][0]          │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_10 (Conv2D)            │ (None, 32, 32, 512)       │       7,078,400 │ dropout_5[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_10 (Activation)    │ (None, 32, 32, 512)       │               0 │ conv2d_10[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_10        │ (None, 32, 32, 512)       │           2,048 │ activation_10[0][0]        │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_11 (Conv2D)            │ (None, 32, 32, 512)       │       2,359,808 │ batch_normalization_10[0]… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_11 (Activation)    │ (None, 32, 32, 512)       │               0 │ conv2d_11[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_11        │ (None, 32, 32, 512)       │           2,048 │ activation_11[0][0]        │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ up_sampling2d_1               │ (None, 64, 64, 512)       │               0 │ batch_normalization_11[0]… │
   │ (UpSampling2D)                │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ concatenate_1 (Concatenate)   │ (None, 64, 64, 768)       │               0 │ batch_normalization_5[0][… │
   │                               │                           │                 │ up_sampling2d_1[0][0]      │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ dropout_6 (Dropout)           │ (None, 64, 64, 768)       │               0 │ concatenate_1[0][0]        │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_12 (Conv2D)            │ (None, 64, 64, 256)       │       1,769,728 │ dropout_6[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_12 (Activation)    │ (None, 64, 64, 256)       │               0 │ conv2d_12[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_12        │ (None, 64, 64, 256)       │           1,024 │ activation_12[0][0]        │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_13 (Conv2D)            │ (None, 64, 64, 256)       │         590,080 │ batch_normalization_12[0]… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_13 (Activation)    │ (None, 64, 64, 256)       │               0 │ conv2d_13[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_13        │ (None, 64, 64, 256)       │           1,024 │ activation_13[0][0]        │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ up_sampling2d_2               │ (None, 128, 128, 256)     │               0 │ batch_normalization_13[0]… │
   │ (UpSampling2D)                │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ concatenate_2 (Concatenate)   │ (None, 128, 128, 384)     │               0 │ batch_normalization_3[0][… │
   │                               │                           │                 │ up_sampling2d_2[0][0]      │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ dropout_7 (Dropout)           │ (None, 128, 128, 384)     │               0 │ concatenate_2[0][0]        │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_14 (Conv2D)            │ (None, 128, 128, 128)     │         442,496 │ dropout_7[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_14 (Activation)    │ (None, 128, 128, 128)     │               0 │ conv2d_14[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_14        │ (None, 128, 128, 128)     │             512 │ activation_14[0][0]        │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_15 (Conv2D)            │ (None, 128, 128, 128)     │         147,584 │ batch_normalization_14[0]… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_15 (Activation)    │ (None, 128, 128, 128)     │               0 │ conv2d_15[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_15        │ (None, 128, 128, 128)     │             512 │ activation_15[0][0]        │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ up_sampling2d_3               │ (None, 256, 256, 128)     │               0 │ batch_normalization_15[0]… │
   │ (UpSampling2D)                │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ concatenate_3 (Concatenate)   │ (None, 256, 256, 192)     │               0 │ batch_normalization_1[0][… │
   │                               │                           │                 │ up_sampling2d_3[0][0]      │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ dropout_8 (Dropout)           │ (None, 256, 256, 192)     │               0 │ concatenate_3[0][0]        │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_16 (Conv2D)            │ (None, 256, 256, 64)      │         110,656 │ dropout_8[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_16 (Activation)    │ (None, 256, 256, 64)      │               0 │ conv2d_16[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_16        │ (None, 256, 256, 64)      │             256 │ activation_16[0][0]        │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_17 (Conv2D)            │ (None, 256, 256, 64)      │          36,928 │ batch_normalization_16[0]… │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ activation_17 (Activation)    │ (None, 256, 256, 64)      │               0 │ conv2d_17[0][0]            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ batch_normalization_17        │ (None, 256, 256, 64)      │             256 │ activation_17[0][0]        │
   │ (BatchNormalization)          │                           │                 │                            │
   ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
   │ conv2d_18 (Conv2D)            │ (None, 256, 256, 1)       │              65 │ batch_normalization_17[0]… │
   └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
    Total params: 31,401,345 (119.79 MB)
    Trainable params: 31,389,569 (119.74 MB)
    Non-trainable params: 11,776 (46.00 KB)

   ```   