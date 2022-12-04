
import os
from pathlib import Path
from typing import Union, Optional

import cv2
import numpy as np
import pandas as pd

from nnood.data.dataset_conversion.utils import generate_dataset_json
from nnood.paths import raw_data_base, DATASET_JSON_FILE

# Dataset available at:
# https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

script_dir = Path(os.path.realpath(__file__)).parent

xray_list_dir = script_dir / 'chestxray14_lists'

train_male_list_path = xray_list_dir / 'norm_MaleAdultPA_train_list.txt'
test_male_list_path = xray_list_dir / 'anomaly_MaleAdultPA_test_list.txt'

train_female_list_path = xray_list_dir / 'norm_FemaleAdultPA_train_list.txt'
test_female_list_path = xray_list_dir / 'anomaly_FemaleAdultPA_test_list.txt'

bbox_data_file_path = xray_list_dir / 'BBox_List_2017.csv'
bbox_csv = pd.read_csv(bbox_data_file_path, index_col=0, usecols=['Image Index', 'Bbox [x', 'y', 'w', 'h]'])

train_test_dict = {
    'male': (train_male_list_path, test_male_list_path),
    'female': (train_female_list_path, test_female_list_path)
}


def organise_xray_data(in_dir: Union[str, Path], data_type: str):

    in_dir_path = Path(in_dir)
    assert in_dir_path.is_dir(), 'Not a valid directory: ' + in_dir

    train_list_path, test_list_path = train_test_dict[data_type]

    out_dir_path = Path(raw_data_base) / f'chestXray14_PA_{data_type}'
    out_dir_path.mkdir(parents=True, exist_ok=True)

    out_train_path = out_dir_path / 'imagesTr'
    out_train_path.mkdir(parents=True, exist_ok=True)


    out_test_path = out_dir_path / 'imagesTs'
    out_test_path.mkdir(parents=True, exist_ok=True)

    out_test_labels_path = out_dir_path / 'labelsTs'
    out_test_labels_path.mkdir(parents=True, exist_ok=True)

    data_augs = {
        'scaling': {'scale_range': [0.97, 1.03]}
    }

    generate_dataset_json(out_dir_path / DATASET_JSON_FILE, out_train_path, out_test_path, ('png-xray',),
                          out_dir_path.name,
                          dataset_description='Images from the NIH Chest X-ray dataset; limited to posteroanterior '
                                              f'views of {data_type} adult patients (over 18), with the test set only '
                                              'including patients which had a bounding box provided.',
                          data_augs=data_augs)

# CHANGE THESE TO MATCH YOUR DATA!!!
organise_xray_data('/vol/biodata/data/chest_xray/ChestXray-NIHCC/images', 'male')
organise_xray_data('/vol/biodata/data/chest_xray/ChestXray-NIHCC/images', 'female')
