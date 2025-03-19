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
