from setuptools import setup

setup(
    name='nnad',
    version='',
    packages=['nnad', 'nnad.data', 'nnad.data.dataset_conversion', 'nnad.utils', 'nnad.training',
              'nnad.training.dataloading', 'nnad.training.loss_functions', 'nnad.training.network_training',
              'nnad.training.data_augmentation', 'nnad.inference', 'nnad.evaluation', 'nnad.preprocessing',
              'nnad.experiment_planning', 'nnad.network_architecture', 'nnad.self_supervised_task',
              'nnad.self_supervised_task.patch_transforms'],
    url='',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=[
        'numpy',
        'nibabel',
        'SimpleITK',
        'tqdm',
        'opencv-python',
        'pandas',
        'torch>=1.10.0',
        'matplotlib',
        'sklearn',
        'scikit-learn>=1.0.1',
        'batchgenerators>=0.23',
        'scikit-image>=0.19.0',
        'argparse',
        'scipy',
        'unittest2',
        'pie-torch'
    ]
)
