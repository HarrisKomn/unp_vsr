import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import wandb
from util import html
import os
from util.visualizer import save_video_images
from collections import OrderedDict
from cleanfid import fid
import argparse
import copy
from pathlib import Path


if __name__ == '__main__':

    ##################################################
    # Train Initialization
    ###################################################

    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    wandb.init(project=opt.wandb_project, entity=opt.wandb_entity , config=vars(opt), mode='online')


    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1


    #test_opts = TestOptions().parse()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    val_opts, _ = parser.parse_known_args()


    ##################################################
    # Validation Initialization
    ###################################################
    val_opts.phase = 'val'
    val_opts.num_threads = 0   # test code only supports num_threads = 0
    val_opts.batch_size  = 1   # test code only supports batch_size = 1
    val_opts.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    val_opts.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    val_opts.display_id = -1
    val_opts.preprocess = 'resize'
    val_opts.load_size_h = 304
    val_opts.load_size_w = 440
    val_opts.crop_size_h = 304
    val_opts.crop_size_w = 440

    val_opts_real = copy.deepcopy(val_opts)
    val_opts_real.dataset_mode = 'ioct'
    val_opts_real.dataroot = ""

    val_opts_real.isTrain = False
    val_opts_real.load_size = 0
    val_opts_real.crop_size = 320
    val_opts_real.apply_segm_loss = False
    val_opts_real.max_dataset_size = float('inf')
    val_opts_real.preprocess = 'crop'

    val_dataset = create_dataset(val_opts_real)  # create a dataset given opt.dataset_mode and other options

    reference_images = (Path(__file__).resolve().parent / 'dataset' / 'valB').as_posix()

    ###################################################################################################################


    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        opt.epoch_counter=epoch

        dataset.set_epoch(epoch)
        html_visuals = []

        for i, data in enumerate(dataset):  # inner loop within one epoch
            wandb.config.update({"epoch":  epoch},allow_val_change=True)
            wandb.log({"epoch_counter": epoch})

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:

                model.data_dependent_initialize(data)
                model.setup(opt)         # regular setup: load and print networks; create schedulers
                model.parallelize()

            model.set_input(data)  # unpack data from dataset and apply preprocessing

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            # wandb
            wandb.log({"loss": model.get_current_losses()})


            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()


        if epoch % opt.save_epoch_freq == 0:      # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        model.update_learning_rate()  # update learning rates at the end of every epoch.



        test_metric_freq = 1

        if epoch % test_metric_freq == 0:
            model.eval()
            opt.isTrain = False

            print('Evaluating FID for validation set at epoch %d, iters %d' % (epoch, total_iters))
            web_dir = os.path.join(opt.checkpoints_dir, opt.name,'FID_images_' + '{}_{}'.format(val_opts.phase, epoch))

            print('creating web directory', web_dir)
            webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))


            for i, data in enumerate(val_dataset):

                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference

                visuals = model.get_current_visuals()  # get image results
                reduced_visuals = OrderedDict()
                reduced_visuals["real_A"] = visuals['real_A']
                reduced_visuals["fake_B"] = visuals['fake_B']

                img_path = model.get_image_paths()  # get image paths

                save_video_images(webpage, reduced_visuals, img_path,model_name = opt.model)

            source_images = os.path.join(web_dir, "images/fake_B/")

            fid_score = fid.compute_fid(source_images, reference_images, mode="clean", num_workers=0)

            wandb.log({"fid_score": fid_score})

            opt.isTrain = True


