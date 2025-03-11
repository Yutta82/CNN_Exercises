from utils.run import train
from utils.decorator_time import display_time


@display_time(text="total consume")
def main():
    """
    主函数
    :return: 0
    """
    train()


if __name__ == '__main__':
    main()
