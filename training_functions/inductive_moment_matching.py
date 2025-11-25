import torch
from torch import nn
from . import shared_functions

def kernel(X,Y,w):
    return torch.exp(w * torch.clamp(torch.exp2(X - Y),min=0.0001) / (4*32*32))

def weighting_function(time):
    weighting=1
    b=2
    epsilon=1e-2
    time_paper=1-time
    a=1-time_paper
    lambda_value=2*(torch.log(a) - torch.log(time_paper))
    derivative_lamda=2*(-1/(1-time_paper+epsilon)-(1/time_paper+epsilon))
    elbo=-derivative_lamda*0.5*nn.functional.sigmoid(b-lambda_value)
    FM_training=1/(torch.square(a)+torch.square(time_paper)+epsilon)
    return (weighting+a+elbo+FM_training).detach()


def IMMLoss(time,images_r,images_t,number_groups,group_size):
    images_r_flattened=images_r.flatten(start_dim=1)
    images_t_flattened=images_t.flatten(start_dim=1)
    images_r_flattened=images_r_flattened.reshape(number_groups,group_size,images_r_flattened.size(1))
    images_t_flattened=images_t_flattened.reshape(number_groups,group_size,images_t_flattened.size(1))
    w=1/torch.abs(time)
    result_kernel_rt=kernel(images_r_flattened,images_t_flattened,w)
    result_kernel_rr=kernel(images_r_flattened,images_t_flattened,w)
    result_kernel_tt=kernel(images_t_flattened,images_t_flattened,w)
    group_loss=result_kernel_rr+result_kernel_tt-2*result_kernel_rt
    return torch.mean(group_loss*weighting_function(time))


def get_v(noise,images):
    return noise-images

def MF_loss(output,input_image,v,jvp,t,s):
    target=input_image+v - (s - t) *jvp
    loss=torch.square(output-target)
    w=1/(loss+0.0001)
    return loss*w.detach


def training_step_IMM(real_images,labels,model,autoencoder, optimizer,number_groups,group_size,r,t,s, device):
    batch_size=number_groups*group_size
    real_images.to(device)
    labels.to(device)
    with torch.no_grad():
        real_images = autoencoder.encode(real_images).latent_dist.sample()
    noise = torch.randn(batch_size,4,32,32).to(device)
    r_unfolded=r.repeat_interleave(group_size, dim=0).to(device)
    t_unfolded=t.repeat_interleave(group_size, dim=0).to(device)
    s_unfolded=s.repeat_interleave(group_size, dim=0).to(device)
    r_images=shared_functions.interpolate(noise,real_images,r_unfolded)
    t_images=shared_functions.interpolate(noise,real_images,t_unfolded)
    s_images_big_jump=model(t_images,t_unfolded,s_unfolded)
    s_images_small_jump=model(r_images,r_unfolded,s_unfolded)
    loss=IMMLoss(s,s_images_big_jump,s_images_small_jump.detach(),number_groups,group_size)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
