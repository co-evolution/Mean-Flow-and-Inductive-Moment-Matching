import torch
from . import mean_flow
from . import inductive_moment_matching
from . import shared_functions

def combination_MF_IMM(real_images,labels,model,autoencoder, optimizer,number_groups,group_size,r,t,s,IMM_loss_factor,device):
    batch_size=number_groups*group_size
    real_images.to(device)
    labels.to(device)
    with torch.no_grad():
        real_images = autoencoder.encode(real_images).latent_dist.sample()
    noise = torch.randn(batch_size,4,32,32).to(device)
    r_unfolded=r.repeat_interleave(group_size).to(device)
    t_unfolded=t.repeat_interleave(group_size).to(device)
    s_unfolded=s.repeat_interleave(group_size).to(device)
    point_velocity=mean_flow.get_v(noise,real_images)
    vector = (point_velocity,torch.zeros_like(t),torch.zeros_like(s))
    r_images=shared_functions.interpolate(noise,real_images,r_unfolded)
    t_images=shared_functions.interpolate(noise,real_images,t_unfolded)

    s_images_big_jump,jvp_big_jump=torch.func.jvp(model,(t_images,t_unfolded,s_unfolded),vector)
    s_images_small_jump,jvp_small_jump=torch.func.jvp((model,(r_images,r_unfolded,s_unfolded),vector))

    w=1/torch.abs(s)
    loss=IMM_loss_factor*inductive_moment_matching.IMMLoss(s,s_images_big_jump,s_images_small_jump.detach(),number_groups,group_size,w)+mean_flow.MF_loss(s_images_big_jump,real_images,point_velocity,jvp_big_jump,t_unfolded,s_unfolded)+mean_flow.MF_loss(s_images_small_jump,real_images,point_velocity,jvp_small_jump,r_unfolded,s_unfolded)

    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()
    return loss

