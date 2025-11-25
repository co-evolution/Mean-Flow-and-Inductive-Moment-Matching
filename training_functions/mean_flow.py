import torch
from . import shared_functions
def get_v(noise,images):
    return noise-images

def MF_loss(output,input_image,v,jvp,t,s):
    target=input_image-(v - (s - t) *jvp)
    loss=(output-target)**2
    w=1/(loss+0.01)
    return (loss*w.detach()).mean()

def training_step_MF(real_images,labels,model,autoencoder,optimizer,batch_size,t,s,device):
    noise = torch.randn(batch_size,4,32,32).to(device)
    real_images.to(device)
    labels.to(device)
    with torch.no_grad():
        real_images = autoencoder.encode(real_images).latent_dist.sample()
    print("images encoded")
    train_inputs=shared_functions.interpolate(noise,real_images,t)
    point_velocity=get_v(noise,real_images)
    vector = (point_velocity,torch.zeros_like(t),torch.zeros_like(s))
    print("velocity calculated")
    model_output,jvp=torch.func.jvp(model,(train_inputs,t,s),vector)
    print("model run")
    loss=MF_loss(model_output,real_images,point_velocity,jvp,t,s)
    optimizer.zero_grad()
    loss.backward()
    print("backwardpass finished")
    optimizer.step()
    return loss.item(),jvp
