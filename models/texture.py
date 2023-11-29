import torch
import torch.nn as nn
import trimesh
import imageio
# from trimesh.ray.ray_pyembree import RayMeshIntersector
from pyoptix.raycast_gpu import OptixRayMeshIntersector

import torch.nn.functional as F
import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step
from models.utils_illum import *
import copy


@models.register("volume-radiance")
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get("n_dir_dims", 3)
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        network = get_mlp(
            self.n_input_dims, self.n_output_dims, self.config.mlp_network_config
        )
        self.encoding = encoding
        self.network = network

    def forward(self, features, dirs, *args):
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat(
            [features.view(-1, features.shape[-1]), dirs_embd]
            + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        color = (
            self.network(network_inp)
            .view(*features.shape[:-1], self.n_output_dims)
            .float()
        )
        if "color_activation" in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register("volume-color")
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(
            self.n_input_dims, self.n_output_dims, self.config.mlp_network_config
        )
        self.network = network

    def forward(self, features, *args):
        network_inp = features.view(-1, features.shape[-1])
        color = (
            self.network(network_inp)
            .view(*features.shape[:-1], self.n_output_dims)
            .float()
        )
        if "color_activation" in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}


@models.register("volume-splitsum")
class VolumeSplitsum(nn.Module):
    def __init__(self, config):
        super(VolumeSplitsum, self).__init__()
        self.config = config
        n_pos_feats = self.config.input_feature_dim
        self.dir_encoding = get_encoding(3, self.config.dir_encoding_config)
        n_dir_feats = self.dir_encoding.n_output_dims
        self.rough_encoding = get_encoding(1, self.config.rough_encoding_config)
        n_rough_feats = self.rough_encoding.n_output_dims
        self.use_transparency = config.get("use_transparency", False)
        if self.use_transparency: self.n_mat_feats = 6
        else: self.n_mat_feats = 5
        self.material_layer = get_mlp(
            n_pos_feats, self.n_mat_feats, self.config.material_network_config
        )  # pos features -> albedo, roughness, metallic
        self.use_ao = config.get("use_ao", True)
        self.learnt_ao = config.get("learnt_ao", True)
        self.illum_loss_rand = config.get("illum_loss_rand", True)
        self.average_ao = config.get("average_ao", False)
        self.weighted_ao_loss = config.get("weighted_ao_loss", False)
        self.start_ao_ones = config.get("start_ao_ones", False)
        self.debug = config.get("debug", False)
        self.ao_weighted_local_loss = config.get("ao_weighted_local_loss", "none")
        self.ao_sample_dist = config.get("ao_sample_dist", 0.05)
        self.mipmap_illum = config.get("mipmap_illum", False)
        if self.average_ao: self.ao_n_channels = 2
        else: self.ao_n_channels = 6
        # self.ao_layer = None
        if self.learnt_ao:
            self.ao_layer = get_mlp(
                        n_pos_feats, self.ao_n_channels, self.config.ao_network_config
                    )  # pos features -> ao
            if self.start_ao_ones: self.ao_layer.requires_grad_(False)
        # import pdb; pdb.set_trace()
        
        if self.mipmap_illum:
            from models.diff_env_mipmap import create_trainable_env_rnd
            self.mipmap = create_trainable_env_rnd(512, scale=0.0, bias=0.5)
        else:
            self.illum_layer = get_mlp(
                n_dir_feats + n_rough_feats, 3, self.config.illum_network_config
            )  # roughness features + directional features (n or r) -> (Ls or Ld)
        self.use_local = config.get("use_local", True)
        if self.use_local:
            self.local_layer = get_mlp(
                n_pos_feats + n_dir_feats, 3, self.config.local_network_config
            )  # pos features + directional features (v) -> Ll
        self.ao_n_samples = config.get("ao_n_samples", 64)
        self.ind_n_samples = config.get("ind_n_samples", 64)
        self.diff_percentage = config.get("diff_percentage", 0.5)
        # if config.indirect_light:

        imageio.plugins.freeimage.download()
        BRDF_map = imageio.imread(self.config.brdf_map_path, format="HDR-FI")
        BRDF_map = torch.Tensor(BRDF_map).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer("BRDF_map", BRDF_map)

        env_map = imageio.imread(self.config.env_map_path, format="HDR-FI")
        env_map = torch.Tensor(env_map).permute(2,0,1).unsqueeze(0) * 3
        self.register_buffer('env_map', env_map)

        if self.config.get("use_gt_mesh", False):
            mesh = trimesh.load(self.config.mesh_path)
            # self.intersector = RayMeshIntersector(mesh)
            self.intersector = OptixRayMeshIntersector(mesh)
            if self.use_ao and self.learnt_ao:
                # self.ao_layer = get_mlp(
                #     n_pos_feats, self.ao_n_channels, self.config.ao_network_config
                # )  # pos features -> ao
                self.ao_layer.requires_grad_(True)
        else:
            self.intersector = None
        self.geometry = None
        self.local_gamma = config.get("local_gamma", 1.0)
        self.local_weight = 1.
        if self.debug: import pdb; pdb.set_trace()

    def sample_BRDF_map(self, NdV, roughness):
        sample_coords = torch.stack([NdV, 1 - roughness], axis=-1).unsqueeze(0)
        sample_coords = sample_coords * 2 - 1
        return (
            F.grid_sample(
                self.BRDF_map,
                sample_coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=None,
            )
            .squeeze(-1)
            .squeeze(0)[:2]
            .T
        )

    def encode_specular(self, R, roughness):
        enc_inputs = [self.dir_encoding(R), self.rough_encoding(roughness)]
        return torch.cat(enc_inputs, axis=-1)

    def get_specular_global(self, R, roughness):
        if self.mipmap_illum:
            spec_global = self.mipmap.get_specular(R, roughness)
        else:
            enc_spec = self.encode_specular(R, roughness)
            spec_global = self.illum_layer(enc_spec)
        return spec_global
    
    # def get_specular_global2(self, R, roughness):
    #     enc_spec = self.encode_specular(R, roughness)
    #     return self.illum_layer.forward(enc_spec)

    def get_diffuse_global(self, N):
        if self.mipmap_illum:
            diffuse = self.mipmap.get_diffuse(N)
        else:
            diffuse = (
                self.get_specular_global(N, torch.ones(N.shape[0], 1, device=N.device))
            )
        return diffuse * torch.pi

    def get_local_radiance(self, pos_features, view_dir):
        enc_inputs = [pos_features, self.dir_encoding(view_dir)]
        local = self.local_layer(torch.cat(enc_inputs, axis=-1))
        
        return local

    # def get_indirect(self, pos_features, view_dir):
    #     enc_inputs = [pos_features, self.dir_encoding(view_dir)]
    #     indirect = self.indirect_layer(torch.cat(enc_inputs, axis=-1))
    #     return indirect

    def query_envmap(self, L):
        return self.get_specular_global(L, torch.zeros_like(L)[..., 0:1])
    
    def query_envmap_nograd(self, L):
        return query_illum(self.env_map, L)

    def query_indirect(self, targets, origins):
        sdf, sdf_grad, feature = self.geometry(
            targets, with_grad=True, with_feature=True
        )
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        t_dirs = targets - origins
        t_dirs = F.normalize(t_dirs, p=2, dim=-1)
        Lb = F.relu(self(feature, t_dirs, normal)["raw_color"])
        return Lb

    def update_ao_mesh(self, mesh, geometry=None, optimizer=None):
        self.geometry = geometry
        self.local_weight *= self.local_gamma
        if self.config.get("use_gt_mesh", False):
            return
        
        # if self.use_ao and self.learnt_ao and not self.ao_layer.output_layer.weight.requires_grad:
        if self.use_ao and self.learnt_ao:
            # self.ao_layer = get_mlp(
            #     self.config.input_feature_dim, self.ao_n_channels, self.config.ao_network_config
            # ).to(self.BRDF_map.device)  # pos features -> ao
            # import pdb; pdb.set_trace()
            self.ao_layer.requires_grad_(True)
            # ao_param_group = optimizer[0].param_groups[1].copy()
            # ao_param_group['params'] =  list(self.ao_layer.parameters())
            # ao_param_group['name'] =  "ao_layer"
            # optimizer[0].param_groups.append(ao_param_group)
            
            
            # optimizer[0]
        mesh = trimesh.Trimesh(vertices=mesh["v_pos"], faces=mesh["t_pos_idx"])
        # self.intersector = RayMeshIntersector(mesh)
        self.intersector = OptixRayMeshIntersector(mesh)
        
        # if self.geometry is None:
        #     self.indirect_layer = get_mlp(
        #         self.config.input_feature_dim + self.dir_encoding.n_output_dims,
        #         3,
        #         self.config.indirect_network_config,
        #     ).to(
        #         self.BRDF_map.device
        #     )  # pos features + directional features (v) -> Ll
        # import pdb; pdb.set_trace()
        

    def forward(self, features, view_dir, N, *args):
        out = {}
        if self.mipmap_illum: self.mipmap.build_mips()
        R = reflect(N, view_dir)
        NdV = dot(N, -view_dir)
        normals_to_flip = NdV < 0
        N = torch.where(normals_to_flip, -N, N)
        NdV = saturate(torch.where(normals_to_flip, -NdV, NdV))

        materials = self.material_layer(features)
        roughness = torch.sigmoid(materials[..., 0:1])
        metallic = torch.sigmoid(materials[..., 1:2])
        albedo = torch.sigmoid(materials[..., 2:5])
        
        if self.learnt_ao:
            # import pdb; pdb.set_trace()
            if self.start_ao_ones and self.intersector is None and self.training:
                ao = torch.ones((N.shape[0],self.ao_n_channels)).to(N)
            else:
                ao = torch.sigmoid(self.ao_layer(features))
        else:
            if N.shape[0] > 0 and self.intersector is not None:
                ao = self.get_ao_mc(args[0], N, roughness)
            else:
                ao = torch.ones((N.shape[0],self.ao_n_channels)).to(N)
        
        # if self.use_ao and self.intersector is not None:
        #     if self.learnt_ao:
        #         ao = torch.sigmoid(self.ao_layer(features))
        #     else:
        #         if N.shape[0] > 0:
        #             # ao = torch.zeros_like(albedo)[...,:2]
        #             ao = self.get_ao_mc(args[0], N, roughness)
        #             # ao = torch.ones_like(ao)
        #         else:
        #             ao = torch.ones((0,2)).to(N)
        # else:
        #     ao = torch.ones_like(albedo)
            
            
            
        # import pdb; pdb.set_trace()
        # if N.shape[0] > 0:
        #     with torch.no_grad():
        #         ao = self.get_ao_mc(args[0], N, roughness)
        # else:
        #     ao = torch.ones((0,2)).to(N)
        # import pdb; pdb.set_trace()
        # print(ao.shape, albedo.shape)
        
        F0 = mix(0.04, albedo, metallic)
        Fr = FresnelSchlickRoughness(NdV, F0, roughness)
        Ks = Fr
        Kd = 1.0 - Ks
        Kd = Kd * ( 1 - metallic )
        
        if self.use_transparency:
            transparency = torch.sigmoid(materials[..., -1:])
            # transparency*= 1e-4
            # Kd *= 1-transparency
            # Kt = 1-transparency + transparency * (1 - torch.pow(NdV, 0.8))
            Kd = Kd * ( 1 - transparency )
            Kt = 1 - transparency * torch.pow(NdV, 0.8)
            out.update({"Kt": Kt,
                        "transparency": transparency})
            Ks = Ks * Kt
            # Ks *= Kt

        diffuse_irradiance_global = self.get_diffuse_global(N)
        diffuse = Kd * albedo * diffuse_irradiance_global / torch.pi

        specular_irradiance_global = self.get_specular_global(R, roughness)
        # specular_irradiance_global = torch.ones_like(diffuse_irradiance_global)
        envBRDF = self.sample_BRDF_map(NdV, roughness)
        specular = specular_irradiance_global * (Fr * envBRDF[:, 0:1] + envBRDF[:, 1:2])

        
        # if self.geometry is not None:
            # local_radiance = F.relu(local_radiance)
            # indirect = self.get_indirect(features, view_dir)
            # raw_color += local_radiance
            # out["indirect"] = local_radiance

        raw_color = (diffuse * ao[..., :self.ao_n_channels//2] + specular * ao[..., -self.ao_n_channels//2:])
        # import pdb; pdb.set_trace()
        # raw_color = diffuse  + specular
        if self.use_local:
            local_radiance = self.get_local_radiance(features, view_dir)
            raw_color = raw_color + local_radiance
            out["local"] = local_radiance
        # import pdb; pdb.set_trace()
        color = saturate(linear_to_srgb(raw_color))

        out.update(
            {
                "raw_color": raw_color,
                "color": color,
                "ao": ao,
                "roughness": roughness,
                "metallic": metallic,
                "albedo": albedo,
                "KdA": Kd * albedo,
                "view_dir": view_dir,
                "F0": F0,
                "N": N,
                "R": R,
            }
        )
        # import pdb; pdb.set_trace()
        return out

    def regularizations(self, out):
        # import pdb; pdb.set_trace()
        reg = {}
        if not self.mipmap_illum:
            if self.illum_loss_rand:
                illum_loss = self.get_illum_loss_rand(8129, 8129, device=self.BRDF_map.device)
            else:
                illum_loss = self.get_illum_loss_trainsamples(8129, 8129, out["R"], out["N"], out["roughness"])
            reg["illum"] = illum_loss

        if self.use_ao and self.learnt_ao:
            ao_loss = self.get_ao_loss(
                out["pos"], out["N"], out["roughness"], out["ao"], out["weights"]
            )
            reg["ao"] = ao_loss


        reg["met"] = self.get_metallic_loss(out["metallic"], out["weights"])
        return reg

    def get_illum_loss(self, L_samples, N_samples, roughness_samples, device):
        ##########################
        # MAIN ASSUMPTION: FG = NoL, v = n
        L_samples = torch.cat([L_samples, N_samples], axis=0)
        L_samples = L_samples.unsqueeze(0)
        N_samples = N_samples.unsqueeze(1)
        roughness_samples = roughness_samples.unsqueeze(1)

        Li_samples_clean = self.query_envmap(L_samples)
        Li_samples_rough = self.get_specular_global(N_samples, roughness_samples)

        NoL = saturate(dot(N_samples, L_samples))
        a = torch.clip(roughness_samples.double() ** 2, 1e-8, 1.0)
        W = NoL / ((NoL + 1) * (a**2 - 1) / 2 + 1) ** 2

        Li_estim_rough = (Li_samples_clean * W).sum(dim=1) / W.sum(dim=1)
        return F.smooth_l1_loss(Li_estim_rough.float(), Li_samples_rough.squeeze(1))

    def get_illum_loss_rand(self, num_samples_light, num_samples_reg, device):
        num_diff = int(self.diff_percentage * num_samples_reg)
        L_samples = sample_sphere(num_samples_light, device=device)
        N_samples = sample_sphere(num_samples_reg, device=device)
        roughness_samples = torch.cat(
            [
                torch.rand(num_samples_reg - num_diff, 1, device=device),
                torch.ones(num_diff, 1, device=device),
            ],
            dim=0,
        )
        return self.get_illum_loss(
            L_samples, N_samples, roughness_samples, device=device
        )

    def get_ao_loss(self, pos_samples, N, roughness, ao, weights):
        if self.intersector is None:
            return torch.tensor(0.0).to(N)
        device = N.device
        num_ao_samples = int(pos_samples.shape[0] * 0.1)
        sample_idx = torch.randperm(
            pos_samples.shape[0], dtype=torch.long, device=device
        )[:num_ao_samples]
        pos_samples, N, roughness, ao, weights = (
            pos_samples[sample_idx],
            N[sample_idx],
            roughness[sample_idx],
            ao[sample_idx],
            weights[sample_idx],
        )
        ao_gt = self.get_ao_mc(pos_samples, N, roughness)
        loss = F.smooth_l1_loss(ao, ao_gt.detach(), reduction="none")
        if self.weighted_ao_loss:
            loss = (loss * weights.unsqueeze(-1) / weights.sum()) * num_ao_samples
        return loss.mean()

    def get_metallic_loss(self, metallic, weights):
        loss = F.smooth_l1_loss(metallic, torch.zeros_like(metallic), reduction="none")
        if self.weighted_ao_loss:
            loss = (loss * weights.unsqueeze(-1) / weights.sum()) * metallic.shape[0]
        return loss.mean()

    def get_ao_mc(self, pos_samples, N, roughness):
        ao_diff = self.get_ao_diff(pos_samples, N)
        ao_spec = self.get_ao_spec(pos_samples, N, roughness)
        return torch.cat([ao_diff, ao_spec], dim=-1)

    @torch.no_grad()
    def get_ao_diff(self, pos_samples, N):
        L = sample_cos_hemisphere(n_samples=self.ao_n_samples, N=N.unsqueeze(1))
        Lg = self.query_envmap(L)
        Vg = self.eval_vis_mesh(pos_samples, L)
        ao_gt = (Lg * Vg).sum(dim=1) / (Lg.sum(dim=1) + 1e-8)
        if self.average_ao: ao_gt = ao_gt.mean(dim=-1, keepdim=True)
        return ao_gt

    @torch.no_grad()
    def get_ao_spec(self, pos_samples, N, roughness):
        L = sample_GGX(
            n_samples=self.ao_n_samples,
            roughness=roughness.unsqueeze(1),
            N=N.unsqueeze(1),
        )
        Lg = self.query_envmap(L)
        Vg = self.eval_vis_mesh(pos_samples, L)
        NoL = saturate(dot(N.unsqueeze(1), L))
        LgNoL = Lg * NoL
        ao_gt = (Vg * LgNoL).sum(dim=1) / (LgNoL.sum(dim=1) + 1e-8)
        if self.average_ao: ao_gt = ao_gt.mean(dim=-1, keepdim=True)
        return ao_gt

    def eval_vis_mesh(self, pos, d):
        d_shape = d.shape
        pos_expanded = pos.unsqueeze(1).expand(-1, d.shape[1], -1).contiguous()
        delta = self.ao_sample_dist
        pos_expanded = pos_expanded + delta * d
        # intersect = self.intersector.intersects_any(
        #     pos_expanded.reshape(-1, 3).detach().cpu(), d.reshape(-1, 3).detach().cpu()
        # )
        intersect = self.intersector.query_intersection(
            pos_expanded.reshape(-1, 3), d.reshape(-1, 3)
        )[...,-1]
        return 1 - intersect.unsqueeze(-1).expand(-1, 3).reshape(d_shape)

