import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_video_images
from util import html
from collections import OrderedDict
import torch


if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options


    if (opt.phase=='test'):

        list_of_checkpoints=opt.test_checkpoint_list
        for k in range(len(list_of_checkpoints)):

            if (not list_of_checkpoints[k]=='latest'):
                opt.epoch = int(list_of_checkpoints[k])

            model = create_model(opt)      # create a model given opt.model and other options

            if 'crop' in opt.preprocess:

                web_dir = os.path.join(opt.results_dir, opt.name,
                                       '{}_{}_{}_{}_{}'.format(opt.phase, opt.epoch, opt.preprocess, opt.crop_size_w, opt.crop_size_h))

            print('creating web directory', web_dir)
            webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

            for i, data in enumerate(dataset):
                if (i % 1 == 0):
                    if i == 0:
                        model.setup(opt)
                        model.parallelize()

                        if opt.eval:
                            model.eval()

                    with torch.no_grad():
                        model.set_input(data)  # unpack data from data loader
                        model.test()           # run inference


                    visuals = model.get_current_visuals()  # get image results

                    reduced_visuals = OrderedDict()  # create a reduced visuals Dict to save only real_A, fake_B, real_B
                    reduced_visuals["real_A"] = visuals['real_A']
                    reduced_visuals["fake_B"] = visuals['fake_B']

                    img_path = model.get_image_paths()     # get image paths
                    if i % 1 == 0:  # save images to an HTML file
                        print('processing (%04d)-th image... %s' % (i, img_path))

                    save_video_images(webpage, reduced_visuals, img_path, aspect_ratio=1., width=opt.display_winsize, model_name = opt.model)






