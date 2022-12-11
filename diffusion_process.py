#%%

import torch
import numpy as np
from torchvision import transforms 
from torchvision.utils import save_image
from model import Unet
import torch.nn.functional as F

#%%
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(dim=28,channels=1,dim_mults=(1, 2, 4,))
model = model.to(device)
model.load_state_dict(torch.load('./save_model/best_model_state_dict.pt'))

IMG_SIZE = 28
T = 3200
betas = linear_beta_schedule(timesteps=T)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def Diffusion_Process(index):
    # Sample noise
    result = torch.tensor([])
    img_size = IMG_SIZE
    img = torch.randn((index, 1, img_size, img_size), device=device)
    num_images = 8
    stepsize = int(T/num_images)

    # Transform
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ])


    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            image = img.detach().cpu()
            for k in range(len(image)):
                image_temp = image[k, :, :, :] 
                image_temp = reverse_transforms(image_temp)
                result = torch.cat((result, image_temp), 0)

    return torch.reshape(result, (-1, 3, 28, 28))


images = Diffusion_Process(8)
save_image(images, '310704009.png')