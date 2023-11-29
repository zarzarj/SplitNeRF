import torch
import torch.nn.functional as F
import numpy as np
import functools


# // Input Ve: view direction
# // Input alpha_x, alpha_y: roughness parameters
# // Input U1, U2: uniform random numbers
# // Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
# vec3 sampleGGXVNDF(vec3 Ve, float alpha_x, float alpha_y, float U1, float U2)
# {
# // Section 3.2: transforming the view direction to the hemisphere configuration
# vec3 Vh = normalize(vec3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
# // Section 4.1: orthonormal basis (with special case if cross product is zero)
# float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
# vec3 T1 = lensq > 0 ? vec3(-Vh.y, Vh.x, 0) * inversesqrt(lensq) : vec3(1,0,0);
# vec3 T2 = cross(Vh, T1);
# // Section 4.2: parameterization of the projected area
# float r = sqrt(U1);
# float phi = 2.0 * M_PI * U2;
# float t1 = r * cos(phi);
# float t2 = r * sin(phi);
# float s = 0.5 * (1.0 + Vh.z);
# t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
# // Section 4.3: reprojection onto hemisphere
# vec3 Nh = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2))*Vh;
# // Section 3.4: transforming the normal back to the ellipsoid configuration
# vec3 Ne = normalize(vec3(alpha_x * Nh.x, alpha_y * Nh.y, std::max<float>(0.0, Nh.z)));
# r


def cart2sph(xyz):
    x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]
    xy = x**2 + y**2
    # r = np.sqrt(xy + z**2)
    # for elevation angle defined from XY-plane up
    elev = torch.arctan2(torch.sqrt(xy), z) 
    # elev = torch.arctan2(z, torch.sqrt(xy)) 
    azim = torch.arctan2(y, x)

    ######################################
    # ORIGINAL BALL HAS + PI 
    azim = azim + torch.pi

    elev[elev < 0] += 2 * torch.pi
    azim[azim < 0] += 2 * torch.pi
    # azim = 2 * torch.pi - azim
    return torch.stack([azim, elev], axis=-1)

def sph2cart(n):
    elev = n[0]
    az = n[1]
    x = np.sin(elev) * np.cos(az)
    y = np.sin(elev) * np.sin(az)
    z = np.cos(elev)
    return np.stack([x, y, z])

def get_sph_sample_coords(elev_res=512, az_res=1024):
    az = -np.linspace(-np.pi, np.pi, az_res)
    elev = np.linspace(0, np.pi, elev_res)
    angles = np.stack([elev[:,None].repeat(az_res, axis=1).reshape(-1), az[None,:].repeat(elev_res, axis=0).reshape(-1)])
    
    sample_coords = torch.tensor(sph2cart(angles).T).cuda().float()
    return sample_coords

def sample_sphere(n_samples, device):
    phi = 2 * torch.pi * torch.rand(n_samples, device=device)
    theta = torch.arccos(1 - 2 * torch.rand(n_samples, device=device))
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    L = torch.stack([x, y, z], axis=-1)
    return L

def query_env_map(env_map, L):
    L_coords = cart2sph(L)
    L_coords = L_coords / (torch.pi)
    L_coords[...,0] = 1 - L_coords[...,0] / 2
    L_coords = L_coords * 2 - 1
    L_env = F.grid_sample(env_map, L_coords.unsqueeze(0),
                                    mode='bilinear', padding_mode='border',
                                    align_corners=None).squeeze(0).permute(1,2,0)
    return L_env

def calc_GGX(NoH, roughness):
    a = (roughness * roughness)
    return a**2 / (torch.pi * ((NoH**2) * (a**2 - 1) + 1)**2 + 1e-10)

def pdf_GGX(NoH, VoH, roughness):
    #pdf = D * NoH / (4 * VoH)
    D = calc_GGX(NoH, roughness)
    return D * NoH / (4 * VoH)

def sample_GGX(N, roughness, n_samples=32):
    #N N, 3
    # roughness N, 1
    # import pdb; pdb.set_trace()
    a = (roughness * roughness)
    samples = torch.rand(N.shape[0], n_samples, 2, device=N.device)
    # N = N.unsqueeze(1).repeat(1,num_samples,1)
    phi = 2 * torch.pi * samples[...,0:1]
    cosTheta = torch.sqrt((1 - samples[...,1:2]) / (1 + (a*a - 1) * samples[...,1:2]))
    sinTheta = torch.sqrt(1 - cosTheta * cosTheta)
    x = sinTheta * torch.cos(phi)
    y = sinTheta * torch.sin(phi)
    z = cosTheta
    
    upVector = torch.zeros_like(N)
    close_up = torch.abs(N[...,2]) > 0.999
    upVector[~close_up] = torch.tensor([0.,0.,1.], device=N.device)
    upVector[close_up] = torch.tensor([1.,0.,0.], device=N.device)
    tangentX = F.normalize(torch.cross(upVector,N, dim=-1), dim=-1)
    tangentY = torch.cross(N, tangentX, dim=-1)
    return tangentX * x + tangentY * y + N * z



def G_sub(roughness, NoV):
    k = roughness**2 / 2
    G = NoV / (NoV * (1 - k) + k)
    G[(roughness == 0).squeeze()] = 1
    # G = torch.where(k == 0, torch.ones_like(G), G)
    return G
    # return NoV 

def G_smith(roughness, NoV, NoL):
    return G_sub(roughness, NoV) * G_sub(roughness, NoL)


def sample_hemisphere(n_samples, N):
    samples = torch.rand(N.shape[0], n_samples, 2, device=N.device)
    theta = torch.arccos(samples[...,0:1])
    phi =  2 * torch.pi * samples[...,1:2]
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    upVector = torch.zeros_like(N)
    close_up = torch.abs(N[...,2]) > 0.999
    upVector[~close_up] = torch.tensor([0.,0.,1.], device=N.device)
    upVector[close_up] = torch.tensor([1.,0.,0.], device=N.device)
    tangentX = F.normalize(torch.cross(upVector,N, dim=-1), dim=-1)
    tangentY = torch.cross(N, tangentX, dim=-1)
    # import pdb; pdb.set_trace()
    L = tangentX * x + tangentY * y + N * z
    return L

def sample_cos_hemisphere(n_samples, N):
    samples = torch.rand(N.shape[0], n_samples, 2, device=N.device)
    theta = torch.arccos(samples[...,0:1].sqrt())
    phi =  2 * torch.pi * samples[...,1:2]
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    upVector = torch.zeros_like(N)
    close_up = torch.abs(N[...,2]) > 0.999
    upVector[~close_up] = torch.tensor([0.,0.,1.], device=N.device)
    upVector[close_up] = torch.tensor([1.,0.,0.], device=N.device)
    tangentX = F.normalize(torch.cross(upVector,N, dim=-1), dim=-1)
    tangentY = torch.cross(N, tangentX, dim=-1)
    # import pdb; pdb.set_trace()
    L = tangentX * x + tangentY * y + N * z
    return L

def query_illum(env_map, L, eval_vis_fn=None, pos=None):
    return query_illum_fn(functools.partial(query_env_map, env_map), L, eval_vis_fn=eval_vis_fn, pos=pos)
    # L_env = query_env_map(env_map, L)
    # if eval_vis_fn is not None:
    #     # import pdb; pdb.set_trace()
    #     # L_env = L_env * ((L_env * eval_vis_fn(pos, L)).sum(dim=1, keepdim=True) / L_env.sum(dim=1, keepdim=True)).mean(dim=-1, keepdim=True)
    #     L_env = L_env * eval_vis_fn(pos, L)
    # return L_env
    
def sample_diffuse(env_map, N, n_samples=1024, eval_vis_fn=None, pos=None):
    return sample_diffuse_fn(functools.partial(query_env_map, env_map),
                             N, n_samples=n_samples, eval_vis_fn=eval_vis_fn, pos=pos)


def sample_specular(env_map, F0, N, V, roughness, n_samples=1024, eval_vis_fn=None, pos=None):
    return  sample_specular_fn(functools.partial(query_env_map, env_map), 
                               F0, N, V, roughness, n_samples=n_samples,
                               eval_vis_fn=eval_vis_fn, pos=pos)


def query_illum_fn(env_query_fn, L, eval_vis_fn=None, pos=None):
    L_env = env_query_fn(L)
    if eval_vis_fn is not None:
        # L_env = L_env * ((L_env * eval_vis_fn(pos, L)).sum(dim=1, keepdim=True) / L_env.sum(dim=1, keepdim=True)).mean(dim=-1, keepdim=True)
        L_env = L_env * eval_vis_fn(pos, L)
    return L_env

def sample_diffuse_fn(env_query_fn, N, n_samples=1024, eval_vis_fn=None, pos=None):
    N = N.unsqueeze(1)
    L = sample_cos_hemisphere(n_samples, N)
    sample_color = query_illum_fn(env_query_fn, L, eval_vis_fn, pos)
    #Uniform pdf over cos-weighted hemisphere = costheta/pi
    diffuse = sample_color * torch.pi
    return diffuse.mean(axis=1)


def sample_specular_fn(env_query_fn, F0, N, V, roughness, n_samples=1024, eval_vis_fn=None, pos=None):
    N = N.unsqueeze(1)
    V = V.unsqueeze(1)
    F0 = F0.unsqueeze(1)
    roughness = roughness.unsqueeze(1)
    # roughness = roughness.clamp(0.01,1.)

    H = sample_GGX(N, roughness, n_samples=n_samples)
    L = reflect(H, V)
    NoV = saturate(dot(N, -V))
    NoL = saturate(dot(N, L))
    NoH = saturate(dot(N, H))
    VoH = saturate(dot(-V, H))
    sample_color = query_illum_fn(env_query_fn, L, eval_vis_fn, pos)

    G = G_smith(roughness, NoV, NoL)
    Fc = torch.pow( 1 - VoH, 5.)
    Fr = (1 - Fc) * F0 + Fc
    # Incident light = SampleColor * NoL
    # Microfacet specular = D*G*F / (4*NoL*NoV)
    # pdf = D * NoH / (4 * VoH)
    specular = sample_color * Fr * G * VoH / (NoH * NoV)
    valid_idx = torch.logical_and(NoL > 0, NoV.repeat(1,NoL.shape[1],1) > 0)
    specular[~valid_idx[...,0]] = 0
    # assert(torch.all(valid_idx.sum(axis=1) > 0))
    # if torch.any(specular.isnan()): import pdb; pdb.set_trace()
    return  specular.mean(axis=1)


def saturate(x):
    return torch.clip(x, 0.0, 1.0)

def linear_to_srgb(x):
    x = saturate(x)

    switch_val = torch.tensor(0.0031308)
    return torch.where(
        torch.ge(x, switch_val),
        1.055 * torch.pow(torch.maximum(x, switch_val), 1.0 / 2.4) - 0.055,
        x * 12.92,
    )
    
def srgb_to_linear(x):
    x = saturate(x)

    switch_val = torch.tensor(0.04045)
    # import pdb; pdb.set_trace()
    return torch.where(
        torch.ge(x, switch_val),
        torch.pow((torch.maximum(x, switch_val) + 0.055) / 1.055, 2.4),
        x / 12.92,
    )
    
def FresnelSchlickRoughness(NdV, F0, roughness):
    return F0 + (torch.maximum(1.0 - roughness, F0) - F0) * torch.pow(torch.maximum(1.0 - NdV, torch.tensor(0.0)), 5.0)

def mix(x, y, a):
    return x * (1 - a) + y * a

def reflect(N, D):
    # import pdb; pdb.set_trace()
    return D - 2 * dot(D, N) * N

def dot(X, Y):
    return torch.sum(X * Y, axis=-1, keepdims=True)