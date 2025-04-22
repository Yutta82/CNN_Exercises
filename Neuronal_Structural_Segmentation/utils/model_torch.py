import torch
import torch.nn as nn


def u_net(input_size=(256, 256, 1)):
    """
    构建 U-Net 模型。
    """
    # 提取输入通道数（例如灰度图为 1）
    in_channels = input_size[2] if len(input_size) == 3 else 1

    class UNet(nn.Module):
        def __init__(self, in_channels):
            super(UNet, self).__init__()
            # 编码器部分
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64)
            )
            self.pool1 = nn.MaxPool2d(2)
            self.drop1 = nn.Dropout2d(0.5)

            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128)
            )
            self.pool2 = nn.MaxPool2d(2)
            self.drop2 = nn.Dropout2d(0.5)

            self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256)
            )
            self.pool3 = nn.MaxPool2d(2)
            self.drop3 = nn.Dropout2d(0.5)

            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512)
            )
            self.pool4 = nn.MaxPool2d(2)
            self.drop4 = nn.Dropout2d(0.5)

            self.conv5 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(1024),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(1024)
            )
            self.drop5 = nn.Dropout2d(0.5)

            # 解码器部分
            self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv6 = nn.Sequential(
                # 双线性插值不会改变通道数，所以up6仍是1024通道，跳跃连接后为1024+512=1536通道
                nn.Conv2d(1536, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512)
            )
            self.drop6 = nn.Dropout2d(0.5)

            self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv7 = nn.Sequential(
                nn.Conv2d(768, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256)
            )
            self.drop7 = nn.Dropout2d(0.5)

            self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv8 = nn.Sequential(
                nn.Conv2d(384, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128)
            )
            self.drop8 = nn.Dropout2d(0.5)

            self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv9 = nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64)
            )
            self.drop9 = nn.Dropout2d(0.5)

            self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
            self.out_activation = nn.Sigmoid()

        def forward(self, x):
            c1 = self.conv1(x)
            p1 = self.pool1(c1)
            d1 = self.drop1(p1)

            c2 = self.conv2(d1)
            p2 = self.pool2(c2)
            d2 = self.drop2(p2)

            c3 = self.conv3(d2)
            p3 = self.pool3(c3)
            d3 = self.drop3(p3)

            c4 = self.conv4(d3)
            p4 = self.pool4(c4)
            d4 = self.drop4(p4)

            c5 = self.conv5(d4)
            d5 = self.drop5(c5)

            up6 = self.up6(d5)
            merge6 = torch.cat([c4, up6], dim=1)
            d6 = self.drop6(merge6)
            c6 = self.conv6(d6)

            up7 = self.up7(c6)
            merge7 = torch.cat([c3, up7], dim=1)
            d7 = self.drop7(merge7)
            c7 = self.conv7(d7)

            up8 = self.up8(c7)
            merge8 = torch.cat([c2, up8], dim=1)
            d8 = self.drop8(merge8)
            c8 = self.conv8(d8)

            up9 = self.up9(c8)
            merge9 = torch.cat([c1, up9], dim=1)
            d9 = self.drop9(merge9)
            c9 = self.conv9(d9)

            out = self.out_conv(c9)
            out = self.out_activation(out)
            return out

    model = UNet(in_channels)
    print(model)
    return model
