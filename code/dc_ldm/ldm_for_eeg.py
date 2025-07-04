import numpy as np
import wandb
import torch
import psutil
import gc
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_eeg import eeg_encoder, classify_network, mapping 
from EEGPT import LitEEGPTCausal, EEGPT2DD
from PIL import Image

from pympler import muppy, summary

def print_mem_usage():
    print("Memory Usage:")
    print(f"CPU RAM: {psutil.virtual_memory().used/1024**3:.2f}GB / {psutil.virtual_memory().total/1024**3:.2f}GB")
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"GPU RAM: {torch.cuda.memory_reserved()/1024**3:.2f}GB / {total_mem/1024**3:.2f}GB")
    # all_objects = muppy.get_objects()
    # sum_obj = summary.summarize(all_objects)
    # summary.print_(sum_obj)

def create_model_from_config(config, num_voxels, global_pool):
    model = eeg_encoder(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels=440, cond_dim=1280, global_pool=True, clip_tune = True, cls_tune = False):
        super().__init__()
        # prepare pretrained fmri mae 
        modelencoder = LitEEGPTCausal(metafile['checkpoint_path']).half()
        model = EEGPT2DD(modelencoder).half()
        self.mae = model
        if clip_tune:
            self.mapping = mapping()
        if cls_tune:
            self.cls_net = classify_network()

        self.fmri_seq_len = 128
        self.fmri_latent_dim = 1024
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

        # self.image_embedder = FrozenImageEmbedder()

    # def forward(self, x):
    #     # n, c, w = x.shape
    #     latent_crossattn = self.mae(x)
    #     if self.global_pool == False:
    #         latent_crossattn = self.channel_mapper(latent_crossattn)
    #     latent_crossattn = self.dim_mapper(latent_crossattn)
    #     out = latent_crossattn
    #     return out

    def forward(self, x):
        # n, c, w = x.shape
        latent_crossattn = self.mae(x)
        latent_return = latent_crossattn
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out, latent_return

    # def recon(self, x):
    #     recon = self.decoder(x)
    #     return recon

    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        # image_embeds = self.image_embedder(image_inputs)
        target_emb = self.mapping(x)
        # similarity_matrix = nn.functional.cosine_similarity(target_emb.unsqueeze(1), image_embeds.unsqueeze(0), dim=2)
        # loss = clip_loss(similarity_matrix)
        loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return loss
    


class eLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune = True, cls_tune = False):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.ckp_path = os.path.join(pretrain_root, 'models/v1-5-pruned.ckpt')
        self.config_path = os.path.join(pretrain_root, 'models/config15.yaml') 
        
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        print("\n=== Model Input/Output Specifications ===")
        print(f"Expected EEG input shape: (batch_size, {num_voxels})")
        print(f"EEG input dtype: torch.float32")
        print(f"Image output shape: (batch_size, 3, {config.model.params.image_size}, {config.model.params.image_size})") 
        print(f"Image output dtype: torch.float32 (normalized to [-1, 1])")
        print("=======================================\n")

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)
        # pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
       
        # m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune = clip_tune,cls_tune = cls_tune)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model.to(device).half()
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def training_step(self, batch, batch_idx):
        # Implement training step logic
        batch = {k: v.to(self.device) for k, v in batch.items()}
        loss = self.model.training_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        # Implement validation step logic  
        batch = {k: v.to(self.device) for k, v in batch.items()}
        loss = self.model.validation_step(batch, batch_idx)
        return loss

    def finetune(self, train_loader, test_loader, num_epochs, lr,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        self.model.eval_avg = config.eval_avg

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        for epoch in range(num_epochs):

            self.model.train()
            train_loss = 0.0
            
            # Training loop
            for batch_idx, batch in enumerate(train_loader):
                print(f"\n### BATCH {batch_idx}")
                optimizer.zero_grad()
                print_mem_usage()
                loss = self.training_step(batch, None)
                print("FORWARD PASS OK")
                print_mem_usage()
                loss.backward()
                print("BACKWARD PASS OK")
                print_mem_usage()
                optimizer.step()
                train_loss += loss.item()
                
                # Cleanup after every 10 batches
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                print()
            
            # Validation loop
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    loss = self.validation_step(batch, None)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(test_loader)
            
            # Save checkpoint if improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': config,
                    'state': torch.random.get_rng_state()
                }, os.path.join(output_path, 'checkpoint.pth'))
            
            print(f'Epoch {epoch}: Val Loss {avg_val_loss:.4f}')
            
            # Final cleanup for epoch
            torch.cuda.empty_cache()
            gc.collect()
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                
                # Memory monitoring
                if count % 10 == 0:
                    print(f"\nMemory usage before sample {count}:")
                    print(f"CPU RAM: {psutil.virtual_memory().used/1024**3:.2f}GB / {psutil.virtual_memory().total/1024**3:.2f}GB")
                    print(f"GPU RAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
                
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w')
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                # Store sample and immediately clean up
                sample = torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)
                all_samples.append(sample)
                
                if output_path is not None:
                    samples_t = (255. * sample.numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
                
                # Explicit cleanup
                del latent, gt_image, c, re_latent, samples_ddim, x_samples_ddim, sample
                if 'samples_t' in locals():
                    del samples_t
                torch.cuda.empty_cache()
                gc.collect()
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
