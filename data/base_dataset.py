import torch.utils.data as data
from PIL import Image
from abc import ABC, abstractmethod
import albumentations as A
import albumentations.pytorch


class BaseDataset(data.Dataset, ABC):

    def __init__(self, opt):

        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):

        return parser

    @abstractmethod
    def __len__(self):

        return 0

    @abstractmethod
    def __getitem__(self, index):

        pass

def get_video_transform(opt):
    transform_list = []
    config = opt.preprocess
    if 'crop' in config: 
        if opt.isTrain:
            transform_list.append(A.RandomCrop(opt.crop_size_h,opt.crop_size_w))
        else:
            transform_list.append(A.CenterCrop(opt.crop_size_h,opt.crop_size_w))

    if 'colorJitter' in config:
        transform_list.append(A.ColorJitter(brightness=(0.5,1)))

    if 'GaussianBlur' in config:
        transform_list.append(A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.005, 1)))

    if 'Brightness' in config:
        transform_list.append(A.RandomBrightness(limit=[0.3,1.9]))

    if 'Gamma' in config:
        transform_list.append(A.RandomGamma (gamma_limit=[38,113]))

    if 'Scale' in config:
        transform_list.append(A.ShiftScaleRotate(shift_limit = [0, 0], scale_limit = [1,1.3],rotate_limit = [0,0]))

    if 'Constrast' in config:
        transform_list.append(A.RandomContrast(limit = [0.1, 1.7]))


    transform_list.append(A.Normalize((0.5), (0.5)))
    transform_list.append(A.pytorch.ToTensorV2())


    return A.Compose(transform_list,additional_targets={'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image'})



def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
