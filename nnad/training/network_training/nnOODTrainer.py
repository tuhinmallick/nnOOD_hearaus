from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import matplotlib
import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.optim import lr_scheduler

from nnad.configuration import default_num_processes
from nnad.evaluation.evaluator import aggregate_scores
from nnad.inference.export_utils import save_data_as_file
from nnad.network_architecture.generic_UNet import Generic_UNet
from nnad.network_architecture.initialisation import InitWeights_He
from nnad.network_architecture.neural_network import AnomalyScoreNetwork
from nnad.self_supervised_task.self_sup_task import SelfSupTask
from nnad.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_default_augmentation, get_patch_size
from nnad.training.dataloading.dataset_loading import DataLoader, load_dataset_filenames, load_npy_or_npz, \
    unpack_dataset
from nnad.training.network_training.network_trainer import NetworkTrainer
from nnad.preprocessing.normalisation import denormalise
from nnad.preprocessing.preprocessing import GenericPreprocessor
from nnad.utils.file_operations import save_json, save_pickle, load_pickle

matplotlib.use('agg')


class nnOODTrainer(NetworkTrainer):
    def __init__(self, plans_file, fold, task_class: Type[SelfSupTask], output_folder: Optional[Path] = None,
                 dataset_directory: Optional[Path] = None, stage=None, unpack_data=True,
                 deterministic=True, fp16=False, load_dataset_ram=False, preprocessor_class=GenericPreprocessor):
        """
        :param plans_file: the pkl file generated by preprocessing. This file will determine all design choices
        :param fold: can be either [0 ... 5) for cross-validation, 'all' to train on all available training data or
        None if you wish to load some checkpoint and do inference only
        :param task_class: self-supervised task used to train network to spot anomalies
        :param output_folder: where to store parameters, plot progress and to the validation
        :param dataset_directory: the parent directory in which the preprocessed data is stored. This is required
        because the split information is stored in this directory. For running prediction only this input is not
        required and may be set to None
        :param stage: The plans file may contain several stages (used for lowres / highres / pyramid). Stage must be
        specified for training:
        if stage 1 exists then stage 1 is the high resolution stage, otherwise it's 0
        :param unpack_data: if False, npz preprocessed data will not be unpacked to npy. This consumes less space but
        is considerably slower! Running unpack_data=False with 2d should never be done!
        :param deterministic:
        IMPORTANT: If you inherit from nnOODTrainer and the init args change then you need to redefine self.init_args
        in your init accordingly. Otherwise checkpoints won't load properly!
        """
        super(nnOODTrainer, self).__init__(deterministic, fp16)
        self.unpack_data = unpack_data
        self.init_args = (plans_file, fold, task_class, output_folder, dataset_directory, stage, unpack_data,
                          deterministic, fp16, load_dataset_ram, preprocessor_class)
        # Set through arguments from init
        self.stage = stage
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.task = task_class()
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder
        self.fold = fold
        self.load_dataset_ram = load_dataset_ram
        self.preprocessor_class = preprocessor_class

        self.plans = None

        self.folder_with_preprocessed_data = None

        # Set in self.initialize()

        self.dl_tr = self.dl_val = None
        # Loaded automatically from plans_file
        self.num_input_channels = self.net_pool_per_axis = self.patch_size = self.batch_size = self.modalities = \
            self.threeD = self.base_num_features = self.intensity_properties = self.normalization_schemes = \
            self.net_num_pool_op_kernel_sizes = self.net_conv_kernel_sizes = self.all_sample_identifiers = \
            self.transpose_forward = self.transpose_backward = self.valid_data_augs = self.num_disable_skip = None

        # Set in setup_DA_params
        self.basic_generator_patch_size = self.data_aug_params = None

        self.loss = self.task.loss

        self.track_metrics = self.track_ap = True
        self.track_auroc = False  # Disabled as unnecessary + slows training
        self.online_eval_overflow = []
        self.online_eval_ap = []
        self.online_eval_auroc = []

        self.do_dummy_2D_aug = None

        self.inference_pad_border_mode = 'constant'
        self.inference_pad_kwargs = {'constant_values': 0}

        self.update_fold(fold)

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 75
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.3

        self.conv_per_stage = None

        # Used for deep supervision, which is not enabled by default
        self.deep_supervision_scales = None

    def update_fold(self, fold):
        """
        Used to swap between folds for inference (ensemble of models from cross-validation)
        DO NOT USE DURING TRAINING AS THIS WILL NOT UPDATE THE DATASET SPLIT AND THE DATA AUGMENTATION GENERATORS
        :param fold:
        :return:
        """
        if fold is None:
            return
        if isinstance(fold, str):
            assert fold == 'all', 'If self.fold is a string then it must be \'all\''
            if self.output_folder.name.endswith(str(self.fold)):
                self.output_folder = self.output_folder_base
            self.output_folder = self.output_folder / str(fold)
        else:
            if self.output_folder.name.endswith(f'fold_{str(self.fold)}'):
                self.output_folder = self.output_folder_base
            self.output_folder = self.output_folder / f'fold_{fold}'
        self.fold = fold

    def setup_DA_params(self):
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            if self.do_dummy_2D_aug:
                self.data_aug_params['dummy_2D'] = True
                self.print_to_log_file('Using dummy2d data augmentation')
                self.data_aug_params['elastic_deform_alpha'] = \
                        default_2D_augmentation_params['elastic_deform_alpha']
                self.data_aug_params['elastic_deform_sigma'] = \
                        default_2D_augmentation_params['elastic_deform_sigma']
                self.data_aug_params['rotation_x'] = default_2D_augmentation_params['rotation_x']
        else:
            self.do_dummy_2D_aug = False
            self.data_aug_params = default_2D_augmentation_params

        def list_to_range(new_list):
            assert len(new_list) == 2, f'Attempted to convert list {new_list} to range, but length is not 2?'
            return tuple(new_list)

        for aug, aug_args in self.valid_data_augs.items():
            self.data_aug_params[f'do_{aug}'] = True

            if aug == 'elastic':
                for p in ['deform_alpha', 'deform_sigma']:
                    self.data_aug_params[f'elastic_{p}'] = list_to_range(aug_args[p])

            elif aug == 'scaling':
                self.data_aug_params['scale_range'] = list_to_range(aug_args['scale_range'])

            elif aug == 'rotation':

                rot_max_rad = aug_args['rot_max'] / 360 * 2 * np.pi
                rot_range_rad = (-rot_max_rad, rot_max_rad)
                self.data_aug_params['rotation_x'] = rot_range_rad

                if self.threeD:
                    self.data_aug_params['rotation_y'] = rot_range_rad
                    self.data_aug_params['rotation_z'] = rot_range_rad

            elif aug == 'gamma':
                self.data_aug_params['gamma_range'] = list_to_range(aug_args['gamma_range'])

            elif aug == 'mirror':
                mirror_axes = aug_args['mirror_axes']

                for a in mirror_axes:
                    assert a < len(self.patch_size), f'Cannot mirror in axis {a} as patch size only hase dimensions' \
                                                         f'{self.patch_size}'

                self.data_aug_params['mirror_axes'] = tuple(mirror_axes)

            elif aug == 'additive_brightness':

                for p in ['mu', 'sigma']:
                    self.data_aug_params[f'additive_brightness_{p}'] = aug_args[p]

            elif aug in ['gaussian_noise', 'gaussian_blur', 'brightness_multiplicative', 'contrast_aug', 'sim_low_res']:
                assert False, f'Augmentation "{aug}" is not implemented'
            else:
                assert False, f'Unknown data augmentation: {aug}, parameters: {aug_args}'

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params['patch_size_for_spatial_transform'] = self.patch_size

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param force_load_plans: 
        :param training:
        :return:
        """

        if not self.was_initialized:
            self.output_folder.mkdir(parents=True, exist_ok=True)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.folder_with_preprocessed_data = \
                self.dataset_directory / f'{self.plans["data_identifier"]}_stage{self.stage}'
            if training:

                if self.unpack_data:
                    self.print_to_log_file('Unpacking dataset')
                    unpack_dataset(self.folder_with_preprocessed_data, self.all_sample_identifiers)
                    self.print_to_log_file('Done')
                else:
                    self.print_to_log_file(
                        'INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you '
                        'will wait all winter for your model to finish!')

                self.dl_tr, self.dl_val = self.get_basic_generators()

                self.tr_gen, self.val_gen = get_default_augmentation(self.dl_tr, self.dl_val,
                                                                     self.data_aug_params[
                                                                         'patch_size_for_spatial_transform'],
                                                                     self.task.label_is_seg(),
                                                                     self.data_aug_params,
                                                                     deep_supervision_scales=
                                                                     self.deep_supervision_scales)
                self.print_to_log_file('TRAINING KEYS:\n %s' % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file('VALIDATION KEYS:\n %s' % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self.was_initialized = True

        else:
            self.print_to_log_file('Attempted to initialise trainer multiple times')

    def initialize_network(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """

        net_num_pool = len(self.net_num_pool_op_kernel_sizes)

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # Notice we set inference_apply_nonlin to identity, as we apply it outside
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, net_num_pool,
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs, net_nonlin, net_nonlin_kwargs,
                                    self.deep_supervision_scales is not None, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    num_disable_skip=self.num_disable_skip)

        self.network.inference_apply_nonlin = self.task.inference_nonlin

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, 'self.initialize_network must be called first'
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                           patience=self.lr_scheduler_patience,
                                                           verbose=True, threshold=self.lr_scheduler_eps,
                                                           threshold_mode='abs')

    def plot_network_architecture(self):
        try:
            import hiddenlayer as hl
            if torch.cuda.is_available():
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                                   transforms=None)
            else:
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)),
                                   transforms=None)
            g.save(self.output_folder / 'network_architecture.pdf')
            del g
        except Exception as e:
            self.print_to_log_file('Unable to plot network architecture:')
            self.print_to_log_file(e)

            self.print_to_log_file('\nprinting the network instead:\n')
            self.print_to_log_file(self.network)
            self.print_to_log_file('\n')
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith('__') and not callable(getattr(self, k)):
                dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, self.output_folder / 'debug.json')

        import shutil

        shutil.copy(self.plans_file, self.output_folder_base / 'plans.pkl')

    def run_training(self):
        self.save_debug_information()
        super(nnOODTrainer, self).run_training()

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                    'If self.stage is None then there can be only one stage in the plans file. That seems to not be the ' \
                    'case. Please specify which stage of the cascade must be trained'
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']
        self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
        self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.all_sample_identifiers = self.plans['dataset_properties']['sample_identifiers']

        self.intensity_properties = plans['dataset_properties']['intensity_properties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.modalities = plans['modalities']
        # Add extra dimensions for positional encoding
        self.num_input_channels = plans['num_modalities'] + len(self.patch_size)
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']
        self.valid_data_augs: Dict[str, Any] = plans['dataset_properties']['data_augs']
        # For backwards compatibility, default to 0 if not present in plans
        self.num_disable_skip = plans['num_disable_skip'] if 'num_disable_skip' in plans.keys() else 0

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError(f'Invalid patch size in plans file: {str(self.patch_size)}')

        self.conv_per_stage = plans['conv_per_stage']

    def load_dataset(self):
        self.print_to_log_file('Loading dataset')
        self.dataset = load_dataset_filenames(self.folder_with_preprocessed_data, self.all_sample_identifiers)

    def get_basic_generators(self):
        self.load_dataset()

        self.task.calibrate(self.dataset, self.plans)

        self.do_split()

        dl_tr = DataLoader(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.task,
                           self.batch_size, False, oversample_foreground_percent=self.oversample_foreground_percent,
                           load_dataset_ram=self.load_dataset_ram,
                           data_has_foreground=self.plans['dataset_properties']['has_uniform_background'])
        dl_val = DataLoader(self.dataset_val, self.patch_size, self.patch_size, self.task, self.batch_size, False,
                            oversample_foreground_percent=self.oversample_foreground_percent,
                            load_dataset_ram=self.load_dataset_ram,
                            data_has_foreground=self.plans['dataset_properties']['has_uniform_background'])

        return dl_tr, dl_val

    def preprocess_sample(self, input_folder, sample_identifier):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_folder: folder containing test case
        :param sample_identifier:
        :return:
        """

        preprocessor = self.preprocessor_class(self.modalities, self.normalization_schemes, self.transpose_forward,
                                               self.intensity_properties, False)

        d, properties = preprocessor.preprocess_test_case(
            self.plans['plans_per_stage'][self.stage]['current_spacing'], input_folder, sample_identifier)
        return d, properties

    def preprocess_predict_sample(self, input_folder: Path, sample_identifier: str, output_file: Optional[Path] = None,
                                  mixed_precision: bool = True) -> None:
        """
        Use this to predict new data
        :param sample_identifier:
        :param input_folder:
        :param output_file:
        :param mixed_precision:
        :return:
        """
        print('Preprocessing...')
        d, properties = self.preprocess_sample(input_folder, sample_identifier)
        print('Predicting...')
        pred = self.predict_preprocessed_data(d, do_mirroring=self.data_aug_params['do_mirror'],
                                              mirror_axes=self.data_aug_params['mirror_axes'],
                                              use_sliding_window=True, step_size=0.5,
                                              use_gaussian=True, pad_border_mode='constant',
                                              pad_kwargs={'constant_values': 0},
                                              verbose=True, all_in_gpu=False,
                                              mixed_precision=mixed_precision)[1]
        pred = pred.transpose([0] + [i + 1 for i in self.transpose_backward])

        if 'export_params' in self.plans.keys():
            force_separate_z = self.plans['export_params']['force_separate_z']
            interpolation_order = self.plans['export_params']['interpolation_order']
            interpolation_order_z = self.plans['export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0

        print('Resampling to original spacing and nifti export...')
        save_data_as_file(pred, output_file, properties, interpolation_order, None, None, output_file, None,
                          force_separate_z=force_separate_z, interpolation_order_z=interpolation_order_z)
        print('Done')

    def predict_preprocessed_data(self, data: np.ndarray, do_mirroring: bool = True,
                                  mirror_axes: Tuple[int] = None,
                                  use_sliding_window: bool = True, step_size: float = 0.5,
                                  use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                  pad_kwargs: dict = None, all_in_gpu: bool = False,
                                  verbose: bool = True, mixed_precision: bool = True) -> np.ndarray:
        """
        :param mixed_precision:
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params['do_mirror'], 'Cannot do mirroring as test time augmentation when training ' \
                                                          'was done without mirroring'

        valid = [AnomalyScoreNetwork, nn.DataParallel]
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                   use_sliding_window=use_sliding_window, step_size=step_size,
                                   patch_size=self.patch_size, use_gaussian=use_gaussian,
                                   pad_border_mode=pad_border_mode, pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                   verbose=verbose, mixed_precision=mixed_precision)
        self.network.train(current_mode)
        return ret

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 export_kwargs: dict = None):
        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, 'must initialize, ideally with checkpoint (or train first)'
        if self.dataset_val is None:
            self.load_dataset()
            self.task.calibrate(self.dataset, self.plans)
            self.do_split()

        if export_kwargs is None:
            if 'export_params' in self.plans.keys():
                force_separate_z = self.plans['export_params']['force_separate_z']
                interpolation_order = self.plans['export_params']['interpolation_order']
                interpolation_order_z = self.plans['export_params']['interpolation_order_z']
            else:
                # Same as parameters used for preprocessing resampling, as our scores are continuous values.
                force_separate_z = None
                interpolation_order = 3
                interpolation_order_z = 0
        else:
            force_separate_z = export_kwargs['force_separate_z']
            interpolation_order = export_kwargs['interpolation_order']
            interpolation_order_z = export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = self.output_folder / validation_folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save': save,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'export_kwargs': export_kwargs,
                         }
        save_json(my_input_args, output_folder / 'validation_args.json')

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                print('WARNING: We did not train with mirroring so you cannot do inference with mirroring enabled')
                do_mirroring = False
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_processes)
        results = []

        ext = 'png' if 'png' in self.modalities[0] else 'nii.gz'

        # Will only be used on first iteration, so can't produce first element of validation set as some tasks (like
        # original FPI) show no difference when interpolating a sample with itself.
        def random_not_first_sample(load_mask):
            random_key = np.random.choice(list(self.dataset_val.keys())[1:])
            return load_npy_or_npz(self.dataset[random_key]['data_file'], 'r', load_mask), \
                   load_pickle(self.dataset[random_key]['properties_file'])

        for k in self.dataset_val.keys():

            properties = load_pickle(self.dataset[k]['properties_file'])
            sample_id = properties['sample_id']

            data_path = output_folder / f'{sample_id}.{ext}'
            pred_path = output_folder / f'{sample_id}_pred.{ext}'
            label_path = output_folder / f'{sample_id}_label.{ext}'

            npz_data_path = output_folder / f'{sample_id}.npz'
            npz_pred_path = output_folder / f'{sample_id}_pred.npz'
            npz_label_path = output_folder / f'{sample_id}_label.npz'

            if overwrite or (not data_path.is_file() or (save and not npz_data_path.is_file())):
                data = load_npy_or_npz(self.dataset[k]['data_file'], 'r',
                                       load_mask=self.plans['dataset_properties']['has_uniform_background'])

                if self.plans['dataset_properties']['has_uniform_background']:
                    mask = data[1]
                    data = data[0]
                else:
                    mask = None

                print(k, data.shape)

                augmented_data, label = self.task(data, mask, properties, random_not_first_sample)

                pred = self.predict_preprocessed_data(augmented_data, do_mirroring=do_mirroring,
                                                      mirror_axes=mirror_axes, use_sliding_window=use_sliding_window,
                                                      step_size=step_size, use_gaussian=use_gaussian,
                                                      all_in_gpu=all_in_gpu, mixed_precision=self.fp16)

                pred = pred.transpose([0] + [i + 1 for i in self.transpose_backward])
                label = label.transpose([0] + [i + 1 for i in self.transpose_backward])

                '''There is a problem with python process communication that prevents us from communicating obejcts
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (i I think) does not allow for
                long enough strings (lol). This could be fixed by changing i to l (for long) but that would require
                manually patching system python code. We circumvent that problem here by saving pred to a npy file that
                will then be read (and finally deleted) by the Process. save_score_as_nifti can take
                either filename or np.ndarray and will handle this automatically'''

                for data, numpy_path, file_path, need_denorm in [(augmented_data, npz_data_path, data_path, True),
                                                                 (pred, npz_pred_path, pred_path, False),
                                                                 (label, npz_label_path, label_path, False)]:
                    if np.prod(data.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(numpy_path, data)
                        data = numpy_path

                    if need_denorm:
                        postprocess_fn = denormalise
                        postprocess_args = (self.normalization_schemes, self.intensity_properties,
                                            properties['channel_intensity_properties'])
                    else:
                        postprocess_fn = postprocess_args = None

                    results.append(export_pool.starmap_async(save_data_as_file,
                                                             ((data, file_path, properties, interpolation_order,
                                                               postprocess_fn, postprocess_args,
                                                               numpy_path if save else None, None, force_separate_z,
                                                               interpolation_order_z),
                                                              )
                                                             )
                                   )
                # If possible use npz files for evaluation, as have less data conversion issues (like png).
                if save:
                    pred_gt_tuples.append((npz_pred_path, npz_label_path))
                else:
                    pred_gt_tuples.append((pred_path, label_path))

        for i in results:
            i.get()

        self.print_to_log_file('Finished prediction')

        # evaluate raw predictions
        self.print_to_log_file('Evaluation of raw predictions')
        task = self.dataset_directory.name
        job_name = self.experiment_name
        results = aggregate_scores(pred_gt_tuples,
                                   json_output_file=output_folder / 'summary.json',
                                   json_name=f'{job_name} val tiled {use_sliding_window}',
                                   json_author='Anonymous', json_task=task, num_threads=default_num_processes)

        # Our tasks are synthetic, so I don't think we want to make any decisions about post-processing based on our
        # validation results (unlike nnUNet)
        self.print_to_log_file('Validation metrics:')
        self.print_to_log_file(results)

        self.network.train(current_mode)

    def run_online_evaluation(self, output, target):
        if not self.track_metrics:
            return
        # TODO: do I need to apply final inference?? YESSSSS
        # TODO: why take only first channel??
        output = output[:, 0].detach().cpu().numpy().flatten()
        target = target[:, 0].detach().cpu().numpy().flatten()

        # Binarise labels, so we can compute average precision/ ROC AUC
        target[target != 0] = 1
        if not (target > 0).any():
            self.online_eval_overflow.append((output, target))
        else:
            if len(self.online_eval_overflow) > 0:
                all_outputs = [o for o, _ in self.online_eval_overflow]
                all_outputs.append(output)
                output = np.concatenate(all_outputs)

                all_targets = [t for _, t in self.online_eval_overflow]
                all_targets.append(target)
                target = np.concatenate(all_targets)

                self.online_eval_overflow = []

            if self.track_ap:
                self.online_eval_ap.append(metrics.average_precision_score(target, output))

            if self.track_auroc:
                self.online_eval_auroc.append(metrics.roc_auc_score(target, output))

    def finish_online_evaluation(self):
        results = {}

        if self.track_metrics:

            print_smth = self.track_ap or self.track_auroc
            metrics_record = []

            if self.track_auroc:
                average_pixel_auroc = np.mean(self.online_eval_auroc)
                self.print_to_log_file('Area Under the Receiver Operating Characteristic curve: ', average_pixel_auroc)
                results['AUROC'] = average_pixel_auroc
                metrics_record.append(average_pixel_auroc)

            if self.track_ap:
                average_pixel_ap = np.mean(self.online_eval_ap)
                self.print_to_log_file('Average precision score: ', average_pixel_ap)
                results['AP'] = average_pixel_ap
                metrics_record.append(average_pixel_ap)

            if print_smth:
                self.print_to_log_file('(Estimated by averaging over validation batches, so not exact)')

            if metrics_record:
                self.all_val_eval_metrics.append(metrics_record)

            self.online_eval_ap = []
            self.online_eval_auroc = []

        return results

    def save_checkpoint(self, fname: Path, save_optimizer=True):
        super(nnOODTrainer, self).save_checkpoint(fname, save_optimizer)
        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans

        save_pickle(info, fname.with_suffix('.pkl'))
