import os,sys,shutil, traceback
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union, Optional
import cv2
import numpy as np
import pandas as pd
from nnad.data.dataset_conversion.utils import generate_dataset_json
from nnad.paths import raw_data_base, DATASET_JSON_FILE

script_dir = Path(os.path.realpath(__file__)).parent

def organise_heraeus_data(in_dir: Union[str, Path], data_type: str):
    # storing the input directory
    in_dir_path = Path(in_dir)
    assert in_dir_path.is_dir(), 'Not a valid directory: ' + in_dir
    import pdb;pdb.set_trace()

    
    # in_test_labels_path = in_dir_path / 'ground_truth'
    # assert in_test_labels_path.is_dir()

    # test_dirs = [d for d in in_test_examples_path.iterdir() if d.is_dir()]
    # assert len(test_dirs) > 1, 'Test must include good and bad examples'
    # nnood_raw_data_base - Folder containing raw data of each dataset
    out_dir_path = Path(raw_data_base) / f'heraus_{data_type}'
    out_dir_path.mkdir(parents=True, exist_ok=True)

    out_train_path = out_dir_path / 'imagesTr'
    out_train_path.mkdir(parents=True, exist_ok=True)
    progression_bar = tqdm(in_dir_path.iterdir(), total = len(list(os.listdir(in_dir_path))),  file=sys.stdout)
    # Copy normal training data
    for folder in progression_bar:
        # To check that the folder is a directory and is not already copied
        if os.path.isdir(folder) == True :
            for file in os.scandir(folder):
                file_name = file.name
                progression_bar.set_description(f"Copying {file_name}")
                frame_number, ext = file_name.split('.')
                # imagesTr/ folder, containing normal training images. 
                # Images must follow naming convention <sample_id>_MMMM.[png|nii.gz], 
                # where MMMM is the modality number.
                # modality_number =(folder.name.split('-')[0]).split('_')[-1]
                number = folder.name.split('-')[-1]
                if (folder.name.split('-')[0]).split('_')[-2]!='Einschluss':
                    shutil.copy(file, out_train_path / f'normal_{number}_{frame_number}_0000.{ext}')
                else:
                    shutil.copy(file, out_train_path / f'einschluss_{number}_{frame_number}_0000.{ext}')
    import pdb; pdb.set_trace()

    out_test_path = out_dir_path / 'imagesTs'
    out_test_path.mkdir(parents=True, exist_ok=True)

    out_test_labels_path = out_dir_path / 'labelsTs'
    out_test_labels_path.mkdir(parents=True, exist_ok=True)
    progression_bar_2 = tqdm(in_dir_path.parent.joinpath('heraus_test').iterdir(), total = len(list(os.listdir(in_dir_path.parent.joinpath('heraus_test')))),  file=sys.stdout)

    for folder in progression_bar_2:
        if os.path.isdir(folder):
            for file in os.scandir(folder):
                file_name = file.name
                progression_bar_2.set_description(f"Copying {file_name}")
                frame_number, ext = file_name.split('.')
                # imagesTr/ folder, containing normal training images. 
                # Images must follow naming convention <sample_id>_MMMM.[png|nii.gz], 
                # where MMMM is the modality number.
                number = folder.name.split('-')[-1]
                shutil.copy(file, out_train_path / f'normal_{number}_{frame_number}_0000.{ext}')

    data_augs = {
        'scaling': {'scale_range': [0.97, 1.03]}
    }

    generate_dataset_json(out_dir_path / DATASET_JSON_FILE, out_train_path, out_test_path, ('png-xray',),
                          out_dir_path.name,
                          dataset_description='Images from the Heraus dataset; '
                                              f'views of {data_type} adult patients (over 18), with the test set only '
                                              'including patients which had a bounding box provided.',
                          data_augs=data_augs)

if __name__ == '__main__':
    try:
        # Folder of image class, or root mvtec dataset folder
        in_root_dir: str = sys.argv[1]
        data_type: str = sys.argv[2]
        if len(sys.argv) == 4 and sys.argv[3] == 'full_dataset':
            print('Processing entire Heraus Dataset')
            in_root_path = Path(in_root_dir)

            for in_class_dir in in_root_path.iterdir():
                if not in_class_dir.is_dir():
                    continue

                print(f'Processing {in_class_dir}...')

                organise_heraeus_data(in_class_dir)

        else:
            print('Processing single folder...')
            organise_heraeus_data(in_root_dir, data_type)
        print('Done!')
    except Exception as e:
        print(traceback.format_exc())

    # python nnad/data/dataset_conversion/convert_heraeus.py /home/tuhin/Data/heraus_images png