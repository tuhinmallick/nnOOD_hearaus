import numpy as np
from collections import OrderedDict
from multiprocessing import Pool
from PIL import Image
from pathlib import Path
import SimpleITK as sitk
import cv2
from nnad.preprocessing.preprocessing import GenericPreprocessor
from nnad.configuration import default_num_processes
from nnad.utils.file_operations import save_pickle
from nnad.utils.miscellaneous import make_default_mask


class GenericPreprocessor_heraus(GenericPreprocessor):
    """
    For RGB images with a value range of [0, 255]. This preprocessor overwrites the default normalization scheme by
    normalizing intensity values through a simple division by 255 which rescales them to [0, 1]
    NOTE THAT THIS INHERITS FROM PreprocessorFor2D, SO ITS WRITTEN FOR 2D ONLY! WHEN CREATING A PREPROCESSOR FOR 3D
    DATA, USE GenericPreprocessor AS PARENT!
    """
    def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, ratio, (dw, dh)


    def find_roi(self,sample_identifier: str, input_folder: Path):
        # import pdb;pdb.set_trace()
        for i in range(len(self.modalities)):
            mod = self.modalities[i]
            suffix = 'png' if 'png' in mod else 'nii.gz'
            file_path = input_folder / f'{sample_identifier}_{i:04d}.{suffix}'

            sitk_image = sitk.ReadImage(file_path.__str__())
            image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
            gradient = 5
            img = image_array.copy()
            if len(img.shape) == 3:
                img = img[:, :, 0]
            upper_part = img[:60, :]
            medians = np.median(upper_part, axis=0)

            first_part = medians[gradient:]
            second_part = medians[:-gradient]
            differences = first_part - second_part

            first_element = np.argmin(differences[: int(0.66 * img.shape[1])])
            starting_point = np.argmax(differences[first_element : first_element + 300])
            try:
                ending_point = np.argmin(differences[first_element + starting_point + 200 :])
            except Exception as e:
                import pdb;pdb.set_trace()
                print(sample_identifier)

            starting_col = first_element + starting_point - 2 * gradient
            ending_col = first_element + starting_point + ending_point + 200 + 4 * gradient
            return starting_col, ending_col
    
    def _run_internal(self, target_spacing, sample_identifier, sample_properties, output_folder_stage: Path,
                      input_folder, force_separate_z):
        # import pdb;pdb.set_trace()
        left_margin, right_margin = self.find_roi(sample_identifier, input_folder)
        
        sample_data = self.load_and_combine(left_margin, right_margin,sample_identifier, input_folder)

        sample_data = sample_data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        
        resampled_data, sample_properties = self.resample_and_normalise(sample_data, target_spacing, sample_properties,
                                                                        force_separate_z)
        
        
        output_file_path = output_folder_stage / f'{sample_identifier}.npz'
        output_properties_path = output_folder_stage / f'{sample_identifier}.pkl'

        # Pass empty array as default, as np.save/load converts None to array of None, which isn't nice
        fg_mask = get_object_mask(resampled_data) if self.make_foreground_masks else make_default_mask()

        print('Saving ', output_file_path)
        np.savez_compressed(output_file_path, data=resampled_data, mask=fg_mask)
        save_pickle(sample_properties, output_properties_path)

    
    def run(self, target_spacings, input_folder: Path, sample_identifiers, sample_properties: OrderedDict,
            output_folder: Path, data_identifier, num_proc=default_num_processes, force_separate_z=None):

        print('Initialising to run preprocessing')
        print('Input folder: ', input_folder)
        print('Output folder: ', output_folder)

        num_stages = len(target_spacings)
        if not isinstance(num_proc, (list, tuple, np.ndarray)):
            num_proc = [num_proc] * num_stages

        assert len(num_proc) == num_stages

        for i in range(num_stages):
            output_folder_stage = output_folder / f'{data_identifier}_stage{i}'
            output_folder_stage.mkdir(parents=True, exist_ok=True)

            spacing = target_spacings[i]
            all_args = map(lambda s_i: [spacing, s_i, sample_properties[s_i], output_folder_stage, input_folder,
                                        force_separate_z],
                           sample_identifiers)
            # import pdb;pdb.set_trace()
            for id in sample_properties:
                self._run_internal(spacing, id,sample_properties[id],output_folder_stage, input_folder,force_separate_z)
            # with Pool(num_proc[i]) as pool:
            #     pool.starmap(self._run_internal, all_args)