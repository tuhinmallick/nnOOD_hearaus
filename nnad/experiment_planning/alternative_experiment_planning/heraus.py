
from nnad.experiment_planning.experiment_planner import ExperimentPlanner
from nnad.paths import *
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

class ExperimentPlanner_heraus(ExperimentPlanner):
    """
    used by tutorial nnunet.tutorials.custom_preprocessing
    """
    def __init__(self, raw_data_path: Path, preprocessed_data_path: Path, num_processes: int, num_disable_skip: int):
        super().__init__(raw_data_path, preprocessed_data_path, num_processes, num_disable_skip)
        self.data_identifier = "nnood_hearaus"
        self.plans_fname = join(self.preprocessed_data_path, "nnood_hearaus" + "_plans.pkl")

        # The custom preprocessor class we intend to use is GenericPreprocessor_heraus. It must be located
        # in nnunet.preprocessing (any file and submodule) and will be found by its name. Make sure to always define
        # unique names!
        self.preprocessor_name = 'GenericPreprocessor_heraus'