#%%

import os
import torch
import numpy as np
from PIL import Image
import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from model import Unet

# %%

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

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

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def show_tensor_image(image):
    matplotlib.use('agg')
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

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
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 8
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i/stepsize+1)
            show_tensor_image(img.detach().cpu())
    plt.savefig('./save/' + str(epoch) + '_pic') 

#%%
channel = 1
data_transforms = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ])


x = np.zeros((len(os.listdir("mnist")), channel, 28, 28), dtype=np.float32)

for i, filename in enumerate(os.listdir("mnist")):
    img = Image.open(os.path.join("mnist/",filename)).convert('L')
    img = data_transforms(img)
    x[i, :, :, :] = img

#%%

savepic_folder = Path("./save")
savepic_folder.mkdir(exist_ok = True)
savemodel_folder = Path("./save_model")
savemodel_folder.mkdir(exist_ok = True)
saveresult_folder = Path("./310704009")
saveresult_folder.mkdir(exist_ok = True)

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 28
BATCH_SIZE = 256
dataloader = DataLoader(x, batch_size=BATCH_SIZE, shuffle=True)

model = Unet(dim=28,channels=1,dim_mults=(1, 2, 4,))
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001, cycle_momentum=False)

epochs = 200 
new_loss = 999
for epoch in range(epochs):
    for step, batch in tqdm.tqdm(enumerate(dataloader)):
        batch_size = batch.shape[0]
        optimizer.zero_grad()

        t = torch.randint(0, T, (batch_size,), device=device).long()
        loss = get_loss(model, batch, t)

        if new_loss > loss:
            new_loss = loss
            torch.save(model.state_dict(), ('./save_model/best_model_state_dict.pt'))

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            torch.save(model.state_dict(), ('./save_model/'+ str(epoch) +'_model_state_dict.pt'))
            sample_plot_image()

# %%
