from Neuronal_Structural_Segmentation.utils.decorator_time import display_time
from Neuronal_Structural_Segmentation.utils.run_torch import *


@display_time(text="total consume")
def main():
    args = build_args()
    mode = args['mode']
    if mode == 'train':
        train(args)
    elif mode == 'predict':
        predict(args)


if __name__ == '__main__':
    main()
    # import netron
    # myNet = u_net()  # 实例化 resnet18
    # x = torch.randn(1, 1, 256, 256)  # 随机生成一个输入
    # modelData = r"checkpoint/demo.pth"  # 定义模型数据保存的路径
    # torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
    # netron.start(modelData)  # 输出网络结构
