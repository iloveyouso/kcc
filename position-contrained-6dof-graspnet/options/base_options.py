import argparse
import os
from utils import utils
import torch
import shutil
import yaml

import numpy as np
import random


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument(
            '--dataset_root_folder', '-d',
            type=str,
            default='/data/bjkim/kcc/full_dataset',
            help='path to root directory of the dataset.')
        self.parser.add_argument('--num_objects_per_batch',
                                 type=int,
                                 default=1,
                                 help='data batch size.')
        self.parser.add_argument('--num_grasps_per_object',
                                 type=int,
                                 default=64)
        self.parser.add_argument('--num_threads',
                                 default=5,
                                 type=int,
                                 help='# threads for loading data')
        self.parser.add_argument(
            '--gpu_ids',
            type=str,
            default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir',
                                 type=str,
                                 default='./checkpoints',
                                 help='models are saved here')
        self.parser.add_argument(
            '--serial_batches',
            action='store_true',
            help='if true, takes meshes in order, otherwise takes them randomly'
        )
        self.parser.add_argument('--latent_size', type=int, default=2)
        self.parser.add_argument(
            '--model_scale',
            type=int,
            default=1,
            help='the scale of the parameters. Use scale >= 1. Scale=2 increases the number of parameters in model by 4x.'
        )
        self.parser.add_argument(
            '--grasps_folder_name',
            type=str,
            default='grasps',
            help='Directory that contains the grasps. Will be joined with the dataset_root_folder and the file names as defined in the splits.'
        )
        self.parser.add_argument(
            '--pointnet_radius',
            help='Radius for ball query for PointNet++, just the first layer',
            type=float,
            default=0.02)
        self.parser.add_argument(
            '--pointnet_nclusters',
            help='Number of cluster centroids for PointNet++, just the first layer',
            type=int,
            default=128)
        self.parser.add_argument(
            '--init_type',
            type=str,
            default='normal',
            help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument(
            '--init_gain',
            type=float,
            default=0.02,
            help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument(
            '--manual_seed', type=int, help='manual seed')
        self.parser.add_argument('--dataset_split_ratio', '-dsr',
                                 type=float,
                                 nargs=3, default=[0.8, 0.15, 0.05],
                                 help='This splits the training dataset into a train, test, and validation set according to the ratios specified.')
        self.parser.add_argument('--extra_name', '-en',
                                 type=str,
                                 default="",
                                 help='Extra name to append to folder name.')
        self.parser.add_argument(
            '--no_vis',
            action='store_true',
            help='if true, no visualizations are stored')
        self.parser.add_argument('--caching', '-c',
                                 action='store_true',
                                 help='If true, then once training data is read from a file it is cached in a large list. Warning, this can consume a lot of RAM memory!')
 

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        assert sum(self.opt.dataset_split_ratio) == 1, "The split ratio of the dataset must sum to one."
        self.opt.is_train = self.is_train  # train or test
        self.opt.validate = self.validate
        self.opt.batch_size = self.opt.num_objects_per_batch * \
            self.opt.num_grasps_per_object
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        args = vars(self.opt)

        if self.opt.manual_seed is None:
            self.opt.manual_seed = random.randint(1, 10000)

        self.set_random_seed()

        if self.opt.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            name ="vae" 
            name += "_lr_" + str(self.opt.lr).split(".")[-1] + "_bs_" + str(
                self.opt.batch_size)
            name += "_scale_" + str(self.opt.model_scale) + "_npoints_" + str(
                self.opt.pointnet_nclusters) + "_radius_" + str(
                    self.opt.pointnet_radius).split(".")[-1]
            name += "_latent_size_" + str(self.opt.latent_size)
            if self.opt.extra_name != "":
                name += "_"+self.opt.extra_name
            self.opt.name = name
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            if os.path.isdir(expr_dir) and not self.opt.continue_train:
                option = "Directory " + expr_dir +\
                    " already exists and you have not chosen to continue to train.\nDo you want to override that training instance with a new one the press (Y/N)."
                print(option)
                while True:
                    choice = input()
                    if choice.upper() == "Y":
                        print("Overriding directory " + expr_dir)
                        shutil.rmtree(expr_dir)
                        utils.mkdir(expr_dir)
                        break
                    elif choice.upper() == "N":
                        print(
                            "Terminating. Remember, if you want to continue to train from a saved instance then run the script with the flag --continue_train"
                        )
                        return None
            else:
                utils.mkdir(expr_dir)

            yaml_path = os.path.join(expr_dir, 'opt.yaml')
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(args, yaml_file)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt

    def set_random_seed(self):
        torch.manual_seed(self.opt.manual_seed)
        np.random.seed(self.opt.manual_seed)
        random.seed(self.opt.manual_seed)