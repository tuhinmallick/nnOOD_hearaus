from multiprocessing import Pool, Queue, Process
from pathlib import Path
import shutil
from typing import List, Optional, Union

import numpy as np
import torch

from nnad.inference.export_utils import save_data_as_file
from nnad.inference.model_restore import load_model_and_checkpoint_files
from nnad.preprocessing.preprocessing import resample_data
from nnad.training.dataloading.dataset_loading import load_npy_or_npz
from nnad.training.network_training.nnOODTrainer import nnOODTrainer
from nnad.utils.file_operations import load_pickle
from nnad.utils.miscellaneous import get_sample_ids_and_files


def predict_from_folder(model: Path, input_folder_path: Path, output_folder_path: Path, 
                        folds: Union[List[str], List[int]], save_npz: bool, num_threads_preprocessing: int, 
                        num_threads_nifti_save: int, lowres_scores: Optional[Path], part_id: int, num_parts: int, tta: bool,
                        mixed_precision: bool = True, overwrite_existing: bool = True, 
                        overwrite_all_in_gpu: bool = None, step_size: float = 0.5,
                        checkpoint_name: str = 'model_final_checkpoint', export_kwargs: dict = None):
    """
    Predicts segmentation maps from a folder of images using a given model.

    Args:
        model (Path): Path to the saved model.
        input_folder_path (Path): Path to the folder with input images.
        output_folder_path (Path): Path to the folder where the predicted segmentation maps will be saved.
        folds (Union[List[str], List[int]]): List of folds to run. If 'all' then all available folds will be run.
        save_npz (bool): If True, saves the predictions in the npz format.
        num_threads_preprocessing (int): Number of threads to use for pre-processing.
        num_threads_nifti_save (int): Number of threads to use for saving the predictions in the NIFTI format.
        lowres_scores (Optional[Path]): Path to the folder with low resolution scores, used when running the cascade model. 
                                        If not None, must point to a directory.
        part_id (int): The ID of the part that will be run.
        num_parts (int): The number of parts the input is split into.
        tta (bool): If True, applies test-time augmentation (TTA) to the predictions.
        mixed_precision (bool): If True, uses mixed precision.
        overwrite_existing (bool): If True, overwrites the existing files in the output folder. 
        overwrite_all_in_gpu (bool): If not None then it will overwrite the entire batch (all parts) at once. 
                                     This significantly reduces CPU-GPU transfer times, but might cause issues with 
                                     very large input sizes. Use at your own risk.
        step_size (float): Size of steps between crops for the sliding window inference.
        checkpoint_name (str): Name of the checkpoint file.
        export_kwargs (dict): Dictionary with additional arguments passed to the `save_data_as_file` function.
    """
    # Check if input folder exists
    assert input_folder_path.is_dir(), f'Input folder path is not a valid directory: {input_folder_path}'
    
    # Create output folder if it does not exist
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Copy plans.pkl file from the model directory to output directory
    plans_path = Path(model, 'plans.pkl')
    shutil.copy(plans_path, output_folder_path)
    
    # Check if plans.pkl file exists
    assert plans_path.is_file(), 'Folder with saved model weights must contain a plans.pkl file'

    # Get expected modalities from plans.pkl
    expected_modalities = load_pickle(plans_path)['dataset_properties']['modalities']

    # Check integrity of input folder
    sample_id_to_files = get_sample_ids_and_files(input_folder_path, expected_modalities)
    sample_ids = [s_id for (s_id, _) in sample_id_to_files]

    # Check if lowres_scores directory exists and if lowres scores for all files are available if specified
    if lowres_scores is not None:
        assert lowres_scores.is_dir(), 'If lowres_scores is not None then it must point to a directory'
        
        missing_lowres = False
        for sample_id, _ in sample_id_to_files:
            if not (lowres_scores / f'{sample_id}.npy').is_file() and\
            not (lowres_scores / f'{sample_id}.npz').is_file():
                print('Missing file for sample: ', sample_id)
                missing_lowres = True

        assert not missing_lowres, 'Provide lowres scores for missing files listed above.'

    # Set all_in_gpu flag based on overwrite_all_in_gpu argument
    all_in_gpu = False if overwrite_all_in_gpu is None else overwrite_all_in_gpu

    # Set input file suffix based on the suffix of the first file in the input folder
    input_file_suffix = (
        '.png' if sample_id_to_files[0][1][0].suffix == '.png' else '.nii.gz'
    )

    # Call predict_cases() method to make predictions on the specified part of the input folder
    return predict_cases(model, sample_ids[part_id::num_parts], input_folder_path, input_file_suffix,
                        output_folder_path, folds, save_npz, num_threads_preprocessing, num_threads_nifti_save,
                        lowres_scores, tta, mixed_precision=mixed_precision, overwrite_existing=overwrite_existing,
                        all_in_gpu=all_in_gpu,
                        step_size=step_size, checkpoint_name=checkpoint_name,
                        export_kwargs=export_kwargs)



def predict_cases(model: Path, sample_ids: List[str], input_folder_path: Path, input_file_suffix: str,
                  output_folder_path: Path, folds, save_npz, num_threads_preprocessing, num_threads_nifti_save,
                  scores_from_prev_stage: Optional[Path] = None, do_tta=True, mixed_precision=True,
                  overwrite_existing=False, all_in_gpu=False, step_size=0.5, checkpoint_name='model_final_checkpoint',
                  export_kwargs: dict = None) -> None:
    """
    Predicts segmentation maps for a given set of images using the specified model.

    Args:
        model: Path object specifying the folder where the model is saved, which must contain fold_x subfolders
        sample_ids: List of sample IDs
        input_folder_path: Path object for the folder containing images for the samples
        input_file_suffix: Suffix of input files, which must be either .png or .nii.gz
        output_folder_path: Path object for the directory where output files will be stored
        folds: Tuple of integers specifying which folds to use for predictions (default: (0, 1, 2, 3, 4)). This can also be
        set to 'all' or a subset of the five folds (for example, (0, ) to use only fold_0).
        save_npz: Boolean value indicating whether to save npz files (default: False)
        num_threads_preprocessing: Number of threads to use for preprocessing
        num_threads_nifti_save: Number of threads to use for Nifti saving
        scores_from_prev_stage: Optional Path object specifying scores from a previous stage
        do_tta: Boolean value indicating whether to perform test-time augmentation (default: True)
        mixed_precision: If True, overwrites what the model has in its init (default: True)
        overwrite_existing: Boolean value indicating whether to overwrite existing files (default: False)
        all_in_gpu: Boolean value indicating whether to use GPU for predictions (default: False)
        step_size: Float value specifying the step size for sliding window inference (default: 0.5)
        checkpoint_name: Name of the checkpoint file for the model (default: 'model_final_checkpoint')
        export_kwargs: Optional dictionary of keyword arguments to pass to the exporting process

    Returns:
        None
    """

    if scores_from_prev_stage is not None:  # check if scores from previous stage were provided
        assert scores_from_prev_stage.is_dir()  # ensure that the path to the scores exists and is a directory

    if not overwrite_existing:  # check if existing predictions should be overwritten
        print('Number of cases:', len(sample_ids))  # print number of cases being predicted
        # if save_npz=True then we should also check for missing npz files

        def missing_output_file(s_id: str):  # define function to check for missing output files
            found_file = False
            found_npz = False

            for f in output_folder_path.iterdir():  # iterate over files in output folder

                if f.name == (s_id + input_file_suffix):  # check if corresponding input file exists
                    found_file = True
                    # If we don't care about the npz, or we've already found it, then nothing is missing so return False
                    if not save_npz or found_npz:  # check if npz file exists or we don't care about it
                        return False

                if save_npz and f.name == f'{s_id}.npz':  # check if corresponding npz file exists
                    found_npz = True
                    if found_file:
                        return False

            return True

        sample_ids = list(filter(missing_output_file, sample_ids))  # filter out cases with existing predictions

        print('Number of cases that still need to be predicted:', len(sample_ids))  # print number of remaining cases

    if not sample_ids:  # check if there are any cases left to predict
        print('No samples to predict, so skipping rest of prediction process')
        return

    print('Emptying cuda cache')  # clear GPU cache before starting predictions
    torch.cuda.empty_cache()

    print('Loading parameters for folds,', folds)  # load model parameters for specified folds
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)  # load model checkpoint files
    assert isinstance(trainer, nnOODTrainer)  # ensure that the model is an instance of nnOODTrainer

    if export_kwargs is None:  # check if export arguments are specified
        if 'export_params' in trainer.plans.keys():  # check if export parameters are defined in the model plan
            force_separate_z = trainer.plans['export_params']['force_separate_z']
            interpolation_order = trainer.plans['export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['export_params']['interpolation_order_z']
        else:  # use default export parameters if not specified in model plan
            force_separate_z = None
            interpolation_order = 3
            interpolation_order_z = 0
    else:  # use specified export parameters
        force_separate_z = export_kwargs['force_separate_z']
        interpolation_order = export_kwargs['interpolation_order']
        interpolation_order_z = export_kwargs['interpolation_order_z']

    print('Starting preprocessing generator')  # start preprocessing generator for input data
    preprocessing = preprocess_multithreaded(trainer, sample_ids, input_folder_path, output_folder_path,
                                             num_threads_preprocessing, scores_from_prev_stage)
    print('Starting prediction...')  # start prediction process

    # Create a Pool object to manage multiple sub-processes, set the number of sub-processes to be num_threads_nifti_save
    with Pool(num_threads_nifti_save) as pool:
        results = []

        # iterate through all preprocessed samples
        for preprocessed in preprocessing:
            # retrieve sample id, data and sample properties
            sample_id, (data, sample_properties) = preprocessed

            # if data is a Path object, load the data from file and delete the file
            if isinstance(data, Path):
                real_data = np.load(data)
                data.unlink()
                data = real_data

            # print the sample id for which prediction is being done
            print('Predicting', sample_id)

            # load the first checkpoint for the trainer
            trainer.load_checkpoint_ram(params[0], False)

            # predict scores for the data using the first checkpoint
            scores = trainer.predict_preprocessed_data(
                data, do_mirroring=do_tta and trainer.data_aug_params['do_mirror'],
                mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True, step_size=step_size,
                use_gaussian=True, all_in_gpu=all_in_gpu, mixed_precision=mixed_precision)

            # iterate through remaining checkpoints, if any, and average the scores
            for p in params[1:]:
                trainer.load_checkpoint_ram(p, False)
                scores += trainer.predict_preprocessed_data(
                    data, do_mirroring=do_tta and trainer.data_aug_params['do_mirror'],
                    mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True, step_size=step_size,
                    use_gaussian=True, all_in_gpu=all_in_gpu, mixed_precision=mixed_precision)

            if len(params) > 1:
                scores /= len(params)

            # check if the predictions need to be transposed
            transpose_forward = trainer.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = trainer.plans.get('transpose_backward')
                scores = scores.transpose([0] + [i + 1 for i in transpose_backward])

            # set npz_file to None if save_npz is False, else set it to the appropriate path
            npz_file = output_folder_path / f'{sample_id}.npz' if save_npz else None

            # if the size of the scores tensor is too large, save it to a npy file temporarily
            bytes_per_voxel = 4
            if all_in_gpu:
                bytes_per_voxel = 2
            if np.prod(scores.shape) > (2e9 / bytes_per_voxel * 0.85):
                print('This output is too large for python process-process communication. Saving output temporarily to disk')
                temp_file = output_folder_path / f'{sample_id}.npy'
                np.save(temp_file, scores)
                scores = temp_file

            # create a list of arguments to pass to save_data_as_file function
            args = (scores, output_folder_path / f'{sample_id}{input_file_suffix}',
                    sample_properties, interpolation_order, None, None, npz_file, None,
                    force_separate_z, interpolation_order_z)

            # add the starmap result to the results list
            results.append(pool.starmap_async(save_data_as_file, (args,)))

        # wait for all the results to complete
        print('inference done. Now waiting for the exporting to finish...')
        _ = [i.get() for i in results]



def preprocess_multithreaded(trainer: nnOODTrainer, sample_ids: List[str], input_folder_path: Path,
                             output_folder_path: Path, num_processes=2, scores_from_prev_stage: Optional[Path] = None):
    """Preprocesses the input data for a set of sample IDs using multiple processes in parallel.

    Args:
        trainer (nnOODTrainer): The model trainer with a defined `preprocess_sample` function to preprocess each sample.
        sample_ids (List[str]): A list of sample IDs to preprocess.
        input_folder_path (Path): The path to the folder containing the input data.
        output_folder_path (Path): The path to the folder to save the preprocessed data.
        num_processes (int, optional): The number of processes to use for preprocessing. Defaults to 2.
        scores_from_prev_stage (Optional[Path], optional): The path to a folder with scores from a previous stage. Defaults to None.

    Yields:
        Tuple[str, Tuple[np.ndarray, Dict]]: A tuple containing the sample ID and a tuple with the preprocessed data and its properties.

    Raises:
        AssertionError: If `trainer` is not an instance of `nnOODTrainer`.

    """

    # Determine the number of processes to use
    num_processes = min(len(sample_ids), num_processes)

    # Create a queue with capacity for one item
    q = Queue(1)

    # Create a process for each thread to preprocess data and add it to the queue
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_sample, q,
                                                            sample_ids[i::num_processes],
                                                            input_folder_path,
                                                            output_folder_path,
                                                            scores_from_prev_stage,
                                                            trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        # Wait for all processes to complete
        while end_ctr != num_processes:
            # Get a preprocessed item from the queue
            item = q.get()
            if item == 'end':
                # If the item is the signal to end, increment the end counter and continue
                end_ctr += 1
                continue
            else:
                # Yield the item
                yield item

    finally:
        # If an exception is raised, terminate any still running processes and join them
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

        # Close the queue
        q.close()

        
        
def preprocess_save_to_queue(preprocess_fn, q: Queue, sample_ids: List[str], input_folder: Path,
                             output_folder: Path, scores_from_prev_stage: Optional[Path], transpose_forward):
    """
    Preprocesses the data for the given sample IDs and saves the results to a multiprocessing queue.

    Args:
        preprocess_fn (Callable): The function to use for preprocessing each sample.
        q (Queue): The multiprocessing queue where to save the preprocessed results.
        sample_ids (List[str]): The list of sample IDs to preprocess.
        input_folder (Path): The path to the input folder where to look for the samples.
        output_folder (Path): The path to the output folder where to save the preprocessed results.
        scores_from_prev_stage (Optional[Path]): The path to the scores of the previous stage for each sample.
            If None, scores are not loaded.
        transpose_forward (Optional[Tuple[int]]): The axes order to transpose the loaded scores. 
            If None, scores are not transposed.

    Returns:
        None

    """

    # Check if scores_from_prev_stage is not None and is a directory
    if scores_from_prev_stage is not None:
        assert scores_from_prev_stage.is_dir(), 'scores_from_prev_stage in preprocess_save_to_queue is not None, but ' \
                                                'not a directory!'

    # Iterate over each sample ID
    for sample_id in sample_ids:
        try:
            # Preprocess the data for the current sample
            print('Preprocessing ',  sample_id)
            data, properties = preprocess_fn(input_folder, sample_id)

            # If scores from previous stage exist, load them and append to the data
            if scores_from_prev_stage is not None:
                scores_prev: np.array = load_npy_or_npz(scores_from_prev_stage / f'{sample_id}.npz', 'r')

                # Check if the shapes match
                assert (np.array(scores_prev.shape) == properties['original_size']).all(),\
                       'image and scores from previous stage don\'t have the same pixel array shape! image: ' \
                       f'{properties["original_size"]}, scores_prev: {scores_prev.shape}'

                scores_prev = scores_prev.transpose(transpose_forward)
                scores_reshaped = resample_data(scores_prev, data.shape[1:])
                data = np.vstack((data, scores_reshaped)).astype(np.float32)

            # Check if the size of data exceeds the maximum limit for process communication
            if np.prod(data.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print('This output is too large for python process-process communication. Saving output temporarily to '
                      'disk')
                output_file_path = output_folder / f'{sample_id}.npy'
                np.save(output_file_path, data)
                data = output_file_path

            # Put the preprocessed data and properties into the queue
            q.put((sample_id, (data, properties)))

        # If the process is interrupted by the user
        except KeyboardInterrupt:
            raise KeyboardInterrupt

        # If there is any other exception during preprocessing
        except Exception as e:
            print('Error in', sample_id)
            print(e)

    # Signal the end of the queue
    q.put('end')

    # Check if there were any errors in the preprocessing
    if errors_in := []:
        print('There were some errors in the following cases:', errors_in)
        print('These cases were ignored.')
    else:
        print('This worker has ended successfully, no errors to report')

