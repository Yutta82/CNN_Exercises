from utils.decorator_time import display_time
from utils.run import *


@display_time(text="total consume")
def main():
    mode = 'predict'
    if mode == 'train':
        train()
    elif mode == 'predict':
        predict()


if __name__ == '__main__':
    main()
