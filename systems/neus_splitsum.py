import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy

def calc_albedo_mult(pred, target, fg_mask):
    albedo_scaling = []
    for j in range(3):
        x_hat = (pred[..., j][fg_mask] ) 
        x = target[..., j][fg_mask]
        scale = x_hat.dot(x) / x_hat.dot(x_hat)
        albedo_scaling.append(scale)
    albedo_scaling = torch.tensor(albedo_scaling).float().to(pred.device)
    # print(albedo_scaling)
    return albedo_scaling


@systems.register('neus-split-system')
class NeuSSplitSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.albedo_mult = None
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        # import pdb; pdb.set_trace()
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
            if self.dataset.has_albedo:
                albedo = self.dataset.all_albedo[index, y, x].view(-1, self.dataset.all_albedo.shape[-1]).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
            if self.dataset.has_albedo:
                albedo = self.dataset.all_albedo[index].view(-1, self.dataset.all_albedo.shape[-1]).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
            if self.dataset.has_albedo:
                albedo = albedo * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        })
        if self.dataset.has_albedo:
            batch.update({
                'albedo': albedo,
            })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / (out['num_samples_full'].sum().item()+1)))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
        # import pdb; pdb.set_trace()
        # print(self.model.texture.ao_layer.output_layer.bias)
        if torch.isnan(loss):
            raise Exception(f'Loss is NaN')

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        # print(batch_idx)
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        extras_comp = []
        for name, value in sorted(out.items(), key=lambda x: x[0].lower()):
            if "extras_comp" in name:
                if "ao" in name:
                    value = value.mean(dim=-1, keepdim=True)
                if value.shape[-1] == 1:
                    extras_comp.append({'type': 'grayscale', 'img': value.view(H, W), 'kwargs': {'data_range': (0, 1), 'cmap': None}})
                else:
                    extras_comp.append({'type': 'rgb', 'img': value.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}})
        if self.config.get("save_vis", True):
            self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {}},
            ] + ([
                {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ] if self.config.model.learned_background else []) + [
                {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
                {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
            ]+extras_comp)
        
        val_out = {
            'psnr': psnr,
            'index': batch['index']
        }      
        # if self.dataset.has_albedo:
        #     # import pdb; pdb.set_trace()
        #     albedo_psnr = self.criterions['psnr'](out['extras_comp_albedo_full'].to(batch['albedo']), batch['albedo'])
        #     val_out.update({'albedo_psnr': albedo_psnr})
        if self.dataset.has_albedo:
            albedo_target = batch['albedo']
            albedo_pred = out['extras_comp_albedo_full'].to(albedo_target)
            fg_mask = batch['fg_mask'] > 0.9
            # import pdb; pdb.set_trace()
            if batch_idx == 0: self.albedo_mult = calc_albedo_mult(albedo_pred, albedo_target, fg_mask)
            albedo_pred[fg_mask] = (albedo_pred[fg_mask] * self.albedo_mult).clamp(0., 1.)
            albedo_psnr = self.criterions['psnr'](albedo_pred, albedo_target)
            val_out.update({'albedo_psnr': albedo_psnr})
        return val_out
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.config.model.texture.get("use_ao", False): self.model.update_ao_mesh(self.trainer.optimizers)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                    if self.dataset.has_albedo:
                        out_set[step_out['index'].item()].update({'albedo_psnr': step_out['albedo_psnr']})
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
                        if self.dataset.has_albedo:
                            out_set[index[0].item()].update({'albedo_psnr': step_out['albedo_psnr'][oi]})
            # log_dict = {}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            # log_dict['val/psnr'] = psnr
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)      
            if self.dataset.has_albedo:  
                albedo_psnr = torch.mean(torch.stack([o['albedo_psnr'] for o in out_set.values()]))
                self.log('val/albedo_psnr', albedo_psnr, prog_bar=True, rank_zero_only=True)  
                # log_dict['val/albedo_psnr'] = albedo_psnr
            # import pdb; pdb.set_trace()
            envmap = self.export_envmap(save_img=self.config.get("save_vis", True))['specular_0.0'].clip(0.,1.)
            gt_envmap = self.dataset.envmap.to(envmap)
            envmap_mult = calc_albedo_mult(envmap, gt_envmap, torch.ones_like(envmap)[...,0].bool())
            envmap = (envmap * envmap_mult).clip(0., 1.)
            envmap_psnr = self.criterions['psnr'](envmap, gt_envmap)
            # log_dict['val/envmap_psnr'] = envmap_psnr
            self.log('val/envmap_psnr', envmap_psnr, prog_bar=True, rank_zero_only=True)     
            # self.log(log_dict, prog_bar=True, rank_zero_only=True)     
            

    def test_step(self, batch, batch_idx):
        # print(batch_idx)
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        extras_comp = []
        for name, value in sorted(out.items(), key=lambda x: x[0].lower()):
            if "extras_comp" in name:
                if "ao" in name:
                    value = value.mean(dim=-1, keepdim=True)
                if value.shape[-1] == 1:
                    extras_comp.append({'type': 'grayscale', 'img': value.view(H, W), 'kwargs': {'data_range': (0, 1), 'cmap': None}})
                else:
                    extras_comp.append({'type': 'rgb', 'img': value.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}})
        # if self.config.get("save_vis", True):
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None}},
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ]+extras_comp)
        test_out = {
            'psnr': psnr,
            'index': batch['index']
        }      
        if self.dataset.has_albedo:
            albedo_target = batch['albedo']
            albedo_pred = out['extras_comp_albedo_full'].to(albedo_target)
            fg_mask = batch['fg_mask'] > 0.9
            # import pdb; pdb.set_trace()
            if batch_idx == 0: self.albedo_mult = calc_albedo_mult(albedo_pred, albedo_target, fg_mask)
            albedo_pred[fg_mask] = (albedo_pred[fg_mask] * self.albedo_mult).clamp(0., 1.)
            albedo_psnr = self.criterions['psnr'](albedo_pred, albedo_target)
            test_out.update({'albedo_psnr': albedo_psnr})
        return test_out
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                    if self.dataset.has_albedo:
                        out_set[step_out['index'].item()].update({'albedo_psnr': step_out['albedo_psnr']})
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
                        if self.dataset.has_albedo:
                            out_set[index[0].item()].update({'albedo_psnr': step_out['albedo_psnr'][oi]})
                            
            # log_dict = {}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            # log_dict['test/psnr'] = psnr
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    
            if self.dataset.has_albedo:  
                albedo_psnr = torch.mean(torch.stack([o['albedo_psnr'] for o in out_set.values()]))
                self.log('test/albedo_psnr', albedo_psnr, prog_bar=True, rank_zero_only=True)    
                # log_dict['test/albedo_psnr'] = albedo_psnr 
            if self.config.get("save_vis", True):
                self.save_img_sequence(
                    f"it{self.global_step}-test",
                    f"it{self.global_step}-test",
                    '(\d+)\.png',
                    save_format='mp4',
                    fps=30
                )
            
            self.export()
            # self.export_envmap()
            envmap = self.export_envmap(save_img=True)['specular_0.0'].clip(0.,1.)
            gt_envmap = self.dataset.envmap.to(envmap)
            envmap_mult = calc_albedo_mult(envmap, gt_envmap, torch.ones_like(envmap)[...,0].bool())
            envmap = (envmap * envmap_mult).clip(0., 1.)
            envmap_psnr = self.criterions['psnr'](envmap, gt_envmap)
            # log_dict['test/envmap_psnr'] = envmap_psnr 
            # self.log(log_dict, prog_bar=True, rank_zero_only=True)  
            self.log('test/envmap_psnr', envmap_psnr, prog_bar=True, rank_zero_only=True) 
    
    def export(self):
        meshes = self.model.export(self.config.export)
        for name, mesh in meshes.items():
            if self.config.get("save_vis", True):
                self.save_mesh(
                    f"it{self.global_step}-{name}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.ply",
                    **mesh
                )  
        
    def export_envmap(self, save_img=True):
        envmap_dict = self.model.export_envmap(self.config.export_envmap)
        envmaps = []
        for envmap_name, envmap in sorted(envmap_dict.items(), key=lambda x: x[0]):
            envmaps.append({'type': 'rgb', 'img': envmap, 'kwargs': {'data_format': 'HWC', 'data_range': (0,1)}})
        if save_img:
            self.save_image_grid(f"it{self.global_step}-envmaps.png", envmaps)
        return envmap_dict
