from setuptools import setup

setup(
    name="micro_diffusion",
    py_modules=["micro_diffusion"],
    install_requires=[
        'accelerate',
        'diffusers',
        'timm',
        'open_clip_torch<=2.24.0',
        'easydict',
        'einops',
        'torchmetrics',
        'mosaicml[tensorboard, wandb]<=0.24.1',
        'pandas',
        'fastparquet',
        'omegaconf', 
        'datasets', 
        'hydra-core',
        'beautifulsoup4'
    ],
)