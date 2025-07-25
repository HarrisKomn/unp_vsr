import argparse
import os
from util import util
import torch
import models
import data
from configs.ParseConfig import parse_config
import time


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters

        parser.add_argument('--config_file', default=False, help='Configuration file')
        parser.add_argument('--wandb_project', default=False, help='Name of the wandb project')
        parser.add_argument('--wandb_entity', default=False, help='Name of the wandb account')

        parser.add_argument('--amp', type=util.str2bool, default=False,
                            help='enables or disables automatic mixed precision')
        parser.add_argument('--run_local', default=False, help='If run locally True')

        parser.add_argument('--apply_augmentation', default=False, help='apply online augmentation')
        parser.add_argument('--apply_segm_loss', default=False, help='include segmentation loss term')
        parser.add_argument('--test_save_as_seq', default=False, help='include segmentation loss term')
        parser.add_argument('--test_checkpoint_list', default=[], help='')

        parser.add_argument('--dataroot', default='placeholder', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--apply_segm_crop', default=False, help='include segmentation crop ')

        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--nce_type', type=str, default='experiment_name', choices=['loss_on_5_frame', 'loss_on_all_gen'],help='name of the nce_type.')

        parser.add_argument('--easy_label', type=str, default='experiment_name', help='Interpretable name')
        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='cut', help='chooses which model to use.')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='n_layers', choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'], help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], help='specify generator architecture')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
        parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
        parser.add_argument('--init_type', type=str, default='normal', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')
        parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')

        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--load_size_w', type=int, default=256, help='scale images to this size')
        parser.add_argument('--load_size_h', type=int, default=256, help='scale images to this size')

        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--crop_size_w', type=int, default=256, help='then crop to this size')
        parser.add_argument('--crop_size_h', type=int, default=256, help='then crop to this size')


        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--random_scale_max', type=float, default=3.0, help='(used for single image translation) Randomly scale the image by the specified factor as data augmentation.')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--epoch_counter', type=int, default=0, help='monitor training epoch')

        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # parameters related to StyleGAN2-based networks
        parser.add_argument('--stylegan2_G_num_downsampling',
                            default=1, type=int,
                            help='Number of downsampling layers used by StyleGAN2Generator')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # parse again with new defaults
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        file_path = './configs/BasicVSR/' + opt.config_file
        config = parse_config(file_path)

        if '_s.json' in opt.config_file:
            opt.gpu_ids = '4'

        else:
            opt.gpu_ids = '0'

        if 'eval' in config:
            opt.eval = config['eval']=='True'

        if 'dataroot' in config:
            opt.dataroot = config['dataroot']

        if 'test_checkpoint_list' in config:
            opt.test_checkpoint_list = config['test_checkpoint_list']

        if 'model' in config:
            opt.model = config['model']

        if 'name' in config:
            opt.name = config['name']

        if 'dataset_mode' in config:
            opt.dataset_mode = config['dataset_mode']

        if 'CUT_mode' in config:
            opt.CUT_mode = config['CUT_mode']

        if 'n_epochs' in config:
            opt.n_epochs = config['n_epochs']

        if 'nce_type' in config:
            opt.nce_type = config['nce_type']

        if 'nce_includes_all_negatives_from_minibatch' in config:
            opt.nce_includes_all_negatives_from_minibatch = config['nce_includes_all_negatives_from_minibatch']=='True'


        if 'lr' in config:
            opt.lr = config['lr']

        if 'epoch' in config:
            opt.epoch = config['epoch']

        if 'no_dropout' in config:
            opt.no_dropout = config['no_dropout'] == 'True'

        if 'no_antialias' in config:
            opt.no_antialias = config['no_antialias'] == 'True'

        if 'no_antialias_up' in config:
            opt.no_antialias_up = config['no_antialias_up'] == 'True'

        #opt.continue_train=config["continue_train"]
        #if opt.isTrain:

        if 'phase' in config:
            opt.phase = config['phase']

        if 'preprocess' in config:
            opt.preprocess = config['preprocess']

        if 'direction' in config:
            opt.direction = config['direction']

        if 'apply_augmentation' in config:
            opt.apply_augmentation=config['apply_augmentation']=='True'

        if 'netG' in config:
            opt.netG = config['netG']

        if 'batch_size' in config:
            opt.batch_size = config['batch_size']

        if 'load_size' in config:
            opt.load_size = config['load_size']

        if 'load_size_h' in config:
            opt.load_size_h = config['load_size_h']

        if 'load_size_w' in config:
            opt.load_size_w = config['load_size_w']

        if 'crop_size' in config:
            opt.crop_size = config['crop_size']

        if 'crop_size_h' in config:
            opt.crop_size_h = config['crop_size_h']

        if 'crop_size_w' in config:
            opt.crop_size_w = config['crop_size_w']

        if 'normG' in config:
            opt.normG=config['normG']

        if 'lambda_NCE_idt' in config:
            opt.lambda_NCE_idt=config['lambda_NCE_idt']

        if 'lambda_NCE' in config:
            opt.lambda_NCE=config['lambda_NCE']

        if 'netF_nc' in config:
            opt.netF_nc=config['netF_nc']

        if 'nce_T' in config:
            opt.nce_T=config['nce_T']

        if 'num_patches' in config:
            opt.num_patches=config['num_patches']

        if 'lambda_NCE_gen' in config:
            opt.lambda_NCE_gen=config['lambda_NCE_gen']

        if 'lambda_perc' in config:
            opt.lambda_perc=config['lambda_perc']

        if 'lambda_sSR' in config:
            opt.lambda_sSR=config['lambda_sSR']

        if 'nce_idt' in config:
            opt.nce_idt = config['nce_idt'] == "True"

        if 'apply_segm_loss' in config:
            opt.apply_segm_loss = config['apply_segm_loss'] == "True"

        if 'apply_segm_crop' in config:
            opt.apply_segm_crop = config['apply_segm_crop'] == "True"

        if 'lambda_segm' in config:
            opt.lambda_segm = config['lambda_segm']

        if 'lambda_GAN' in config:
            opt.lambda_GAN=config["lambda_GAN"]

        if 'lambda_A' in config:
            opt.lambda_A = config["lambda_A"]

        if 'lambda_B' in config:
            opt.lambda_B = config["lambda_B"]

        if 'lambda_identity' in config:
            opt.lambda_identity = config["lambda_identity"]

        if 'lgm_mode' in config:
            opt.lgm_mode=config['lgm_mode']=='True'

        if opt.isTrain:
            timestr = time.strftime("%Y%m%d_%H%M%S")
            # checkpoint_name= timestr +'_' + opt.name
            opt.name = timestr + '_' + opt.name

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
