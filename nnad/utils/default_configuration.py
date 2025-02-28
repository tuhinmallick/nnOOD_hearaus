from pathlib import Path

import nnad
from nnad.experiment_planning.utils import summarise_plans
from nnad.paths import default_plans_identifier, preprocessed_data_base, results_base
from nnad.utils.file_operations import load_pickle
from nnad.utils.miscellaneous import recursive_find_python_class


def get_default_configuration(network: str, dataset_name: str, task_name: str, network_trainer: str,
                              plans_identifier=default_plans_identifier, silent=False):

    assert network in {
        'lowres',
        'fullres',
        'cascade_fullres',
    }, 'network can only be one of the following: \'lowres\', \'fullres\', \'cascade_fullres\''

    prep_data_base = Path(preprocessed_data_base)
    dataset_directory = prep_data_base / dataset_name
    plans_file = prep_data_base / dataset_name / plans_identifier

    plans = load_pickle(plans_file)
    possible_stages = list(plans['plans_per_stage'].keys())

    if network in {'cascade_fullres', 'lowres'} and len(possible_stages) == 1:
        raise RuntimeError('lowres/cascade_fullres only applies if there is more than one stage. This task does '
                           'not require the cascade. Run fullres instead')

    stage = 0 if network == 'lowres' else possible_stages[-1]
    trainer_class = recursive_find_python_class([Path(nnad.__path__[0], 'training', 'network_training').__str__()],
                                                network_trainer, current_module='nnad.training.network_training')
    task_class = recursive_find_python_class([Path(nnad.__path__[0], 'self_supervised_task').__str__()],
                                             task_name, current_module='nnad.self_supervised_task')

    output_folder_name = Path(
        results_base,
        dataset_name,
        task_name,
        network,
        f'{network_trainer}__{plans_identifier}',
    )

    if not silent:
        print('###############################################')
        print(f'I am running the following nnUNet: {network}')
        print('My trainer class is: ', trainer_class)
        print('My task class is: ', task_class)
        print('For that I will be using the following configuration:')
        summarise_plans(plans)
        print('I am using stage %d from these plans' % stage)

        print('\nI am using data from this folder: ', dataset_directory / f'{plans["data_identifier"]}_stage{stage}')

        print('###############################################')
    return plans_file, output_folder_name, dataset_directory, stage, trainer_class, task_class
