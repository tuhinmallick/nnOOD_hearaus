
from nnad.experiment_planning.experiment_planner import ExperimentPlanner
from nnad.paths import *


class ExperimentPlanner_heraus(ExperimentPlanner):
    """
    used by tutorial nnunet.tutorials.custom_preprocessing
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnood_hearaus"
        self.plans_fname = join(self.preprocessed_output_folder, "nnood_hearaus" + "_plans.pkl")

        # The custom preprocessor class we intend to use is GenericPreprocessor_heraus. It must be located
        # in nnunet.preprocessing (any file and submodule) and will be found by its name. Make sure to always define
        # unique names!
        self.preprocessor_name = 'GenericPreprocessor_heraus'