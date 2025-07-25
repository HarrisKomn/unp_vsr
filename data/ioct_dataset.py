import os
from data.base_dataset import BaseDataset,get_video_transform
from data.image_folder import make_dataset
import albumentations as A
import util.util as util
import random
import torch
from pathlib import Path
from models.basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor


class ioctDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        my_opt={
            'num_frame': 5,
            'dataroot_gt':"",
            'dataroot_lq':"",
            'meta_info_file':"",
            'random_reverse': False,
            'scale':1,
            'io_backend':'disk',
            'gt_size':256,
            'use_hflip': False,
            'use_rot': True
        }


        if opt.isTrain:

            my_opt['dataroot_gt'] = Path(__file__).resolve().parent.parent / 'dataset' / 'trainB'
            my_opt['dataroot_lq'] = Path(__file__).resolve().parent.parent / 'dataset'
            my_opt['meta_info_file'] = Path(__file__).resolve().parent.parent / 'data_txts' / 'meta_info_train.txt'

        elif opt.phase == 'val':

            my_opt['dataroot_gt'] = Path(__file__).resolve().parent.parent / 'dataset' / 'valB'
            my_opt['dataroot_lq'] = Path(__file__).resolve().parent.parent / 'dataset'
            my_opt['meta_info_file'] = Path(__file__).resolve().parent.parent / 'data_txts' / 'meta_info_val.txt'

        elif opt.phase == 'test':

            my_opt['dataroot_gt'] = Path(__file__).resolve().parent.parent / 'dataset' / 'testB'
            my_opt['dataroot_lq'] = Path(__file__).resolve().parent.parent / 'dataset'
            my_opt['meta_info_file'] = Path(__file__).resolve().parent.parent / 'data_txts' / 'meta_info_test.txt'

        self.my_opt = my_opt
        self.opt = opt
        self.gt_root, self.lq_root = Path(my_opt['dataroot_gt']), Path(my_opt['dataroot_lq'])

        with open(my_opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        self.file_client = None
        self.is_lmdb = False

        self.neighbor_list = [i for i in range(my_opt['num_frame'])]

        # temporal augmentation configs
        self.random_reverse = my_opt['random_reverse']
        logger = get_root_logger()
        logger.info(f'Random reverse is {self.random_reverse}.')

        self.B_paths = sorted(make_dataset(self.gt_root, float("inf")) ) # load images from '/path/to/data/trainB'
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient('disk')

        key = self.keys[index]
        clip,type, seq = key.split('/')  # key example: 00001/0001

        index_B = random.randint(0, self.B_size - 1)
        img_gt_path = self.gt_root / (str("{:04}".format(index_B)) + '.png')

        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32 = False, flag = 'grayscale')

        img_lqs = []
        img_lq_path=""

        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip /type/ seq / f'im_{neighbor}.png'
                #img_lq_path = self.lq_root / (clip+type + seq+'im_4') / f'im_{neighbor}.png'
                # if not os.path.exists(img_lq_path):
                #     img_lq_path = self.lq_root / clip / type / seq / f'im_{neighbor-6}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=False, flag='grayscale')

            img_lqs.append(img_lq)
        img_lqs.append(img_gt)

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)

        # Video transformation using addiational targets to apply the same transformation to all the input sequence
        self.data_transform=get_video_transform(modified_opt)
        img_results=[]


        if self.opt.isTrain:
            img_lqs_transf = self.data_transform(image  = img_lqs[0],
                                                 image1 = img_lqs[1],
                                                 image2 = img_lqs[2],
                                                 image3 = img_lqs[3],
                                                 image4 = img_lqs[4],
                                                 image5 = img_lqs[5])

        else:
            to_Resize = A.Resize(304, 452)
            img_lqs_transf = self.data_transform( image  = img_lqs[0],
                                                  image1 = img_lqs[1],
                                                  image2 = img_lqs[2],
                                                  image3 = img_lqs[3],
                                                  image4 = img_lqs[4],
                                                  image5 = to_Resize(image = img_lqs[5])['image'])

        gt_name_str = 'image' + str(self.my_opt['num_frame'])
        img_gt_transf=img_lqs_transf[gt_name_str]

        img_lqs_transf = {key: value for key, value in img_lqs_transf.items() if key != (gt_name_str)}

        for i in range(0, len(img_lqs_transf)):
            img_results.append(img_lqs_transf['image'] if i==0 else img_lqs_transf['image'+str(i)])

        img_lqs_transf=torch.stack(img_results, dim=0)

        A_path = os.path.join(img_lq_path)
        B_path = os.path.join(img_gt_path)


        return {'A': img_lqs_transf, 'B': img_gt_transf, 'key': key, 'A_paths': A_path}

    def __len__(self):
        return len(self.keys)

