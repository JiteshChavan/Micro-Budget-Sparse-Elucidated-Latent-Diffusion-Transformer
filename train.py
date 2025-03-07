import time
import hydra

import torch

import torch.distributed as dist

if dist.is_initialized():
    dist.destroy_process_group()


"""composer train.py --config-path ./configs --config-name textcaps_overfit.yaml exp_name=textcaps_overfit model.train_mask_ratio=0.0"""

from omegaconf import DictConfig, OmegaConf
from composer.core import Precision
from composer.utils import dist, reproducibility
from composer.algorithms import GradientClipping
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from micro_diffusion.models.utils import get_text_encoder_embedding_format

# 3-5 % speedup
torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None)
def train(cfg : DictConfig) -> None:
    """Train a sparse latent elucidated diffusion transformer model using the provided configuration.
    
    Args:
        cfg (DictConfig): Configuration object loaded from yaml file.
    """
    if not cfg:
        raise ValueError('Config not specified. Please provide --config-path and --config-name, respectively.')
    
    reproducibility.seed_all(cfg['seed'])

    assert cfg.model.latents_precomputed, "VAE latents are required for training latent diffusion transformer"
    model = hydra.utils.instantiate(cfg.model)

    # Setup optimizer with special handling for MoE prameters
    # model.named_parameters() returns name, p tuple
    moe_params = [p[1] for p in model.dit.named_parameters() if 'moe' in p[0].lower()]
    rest_params = [p[1] for p in model.dit.named_parameters() if 'moe' not in p[0].lower()]

    if len(moe_params) > 0:
        print ('Reducing learning rate of MoE parameters by 1/2')
        opt_dict = dict(cfg.optimizer)
        opt_name = opt_dict['_target_'].split('.')[-1]
        del opt_dict['_target_']
        optimizer = getattr (torch.optim, opt_name) (
                params = [{'params': rest_params}, {'params':moe_params, 'lr':cfg.optimizer.lr/2}], **opt_dict
        )
    else:
        # no moe parameters, no need of multiple param groups
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.dit.parameters())
    
    # Convert ListConfig betas to native list to avoid ValueError when saving optimizer sate
    for p in optimizer.param_groups:
        p['betas'] = list(p['betas'])
    
    # Set up data loaders
    if "text_enoder_name" in cfg.model:
        cap_seq_size, cap_emb_dim = get_text_encoder_embedding_format(cfg.model.text_encoder_name)
    else:
        #default
        cap_seq_size, cap_emb_dim = 77, 1024
    
    train_loader = hydra.utils.instantiate (
        cfg.dataset.train,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.dataset.train_batch_size // dist.get_world_size(), # one batch multiple GPUs
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim,
        cap_drop_prob=cfg.dataset.cap_drop_prob
    )

    print (f"Found {len(train_loader.dataset)*dist.get_world_size()} images in the training dataset")
    time.sleep(3)

    eval_loader = hydra.utils.instantiate(
        cfg.dataset.eval,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.dataset.eval_batch_size // dist.get_world_size(),
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim
    )

    print (f"Found {len(eval_loader.dataset)*dist.get_world_size()} images in eval dataset")
    time.sleep(3)

    # Initialize Training Components
    logger, callbacks, algorithms = [], [], []
    
    # Set up loggers
    for log, log_conf in cfg.logger.items():
        if '_target_' in log_conf:
            if log == 'wandb':
                # we don't do this.
                wandb_logger = hydra.utils.instantiate(log_conf, _partial_=True)
                logger.append(wandb_logger(init_kwargs={'config': OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)}))
            else:
                #Tensorboard
                logger.append(hydra.utils.instantiate(log_conf))

    # Configure Algorithms
    if 'algorithms' in cfg:
        for alg_name, alg_conf in cfg.algorithms.items():
            if alg_name == 'low_precision_layernorm':
                apply_low_precision_layernorm (model=model.dit, precision=Precision(alg_conf['precision']), optimizers=optimizer)
            elif alg_name == "gradient_clipping":
                algorithms.append(GradientClipping(clipping_type='norm', clipping_threshold=alg_conf['clip_norm']))
            else:
                print (f'Algorithm {alg_name} not supported.')
    
    # configure callbacks
    if 'callbacks' in cfg:
        for _, call_conf in cfg.callbacks.items():

            if '_target_' in call_conf:
                print (f'Instantiating callbacks : {call_conf._target_}')
                callback_instance = hydra.utils.instantiate(call_conf)
                
                if _ == "image_monitor" and "caption_latents_path" in call_conf:
                    # do something
                    latents = torch.load(call_conf.caption_latents_path)
                    callback_instance.latent_prompts  = latents
                    print ("LOADED LATENT PROMPTS FOR INFERENCE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    
                callbacks.append(callback_instance)
    
            
            
    scheduler = hydra.utils.instantiate(cfg.scheduler)

    # disable online evals if using torch.compile
    if cfg.misc.compile:
        cfg.trainer.eval_interval = 0
    
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        train_dataloader = train_loader,
        eval_dataloader = eval_loader,
        optimizers = optimizer,
        model = model,
        loggers = logger,
        algorithms = algorithms,
        schedulers = scheduler,
        callbacks = callbacks,
        precision = 'amp_bf16' if cfg.model['dtype'] == 'bfloat16' else 'amp_fp16', #fp16 by default
        python_log_level='debug',
        compile_config={} if cfg.misc.compile else None # enables torch.compile (~15% speedup)
    )

    # Ensure models are on correct device
    device = next(model.dit.parameters()).device
    model.vae.to(device)
    if model.text_encoder is not None:
        model.text_encoder.to(device)

    return trainer.fit()

if __name__ == '__main__':
    train()
