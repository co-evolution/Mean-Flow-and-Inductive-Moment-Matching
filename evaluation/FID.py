import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def evaluate_FID(model,autoencoder,test_dataloader, batch_size, evaluation_number_samples,device):
    fid = FrechetInceptionDistance(normalize=True).to(device)
    model.to(device)
    autoencoder.to(device)
    t = torch.zeros(batch_size)
    s = torch.ones(batch_size)
    for i in range (evaluation_number_samples//batch_size):
        noise = torch.randn(batch_size,4,32,32).to(device)
        latents=model(noise,t,s)
        latents = latents.half()
        fake_images=(autoencoder.decode(latents)+1)/2
        print(fake_images.min().item(), fake_images.max().item())
        fid.update(fake_images,real=False)

    for batch in test_dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            real_images = batch[0]
        else:
            real_images = batch
        fid.update(real_images.to(device), real=True)


    fid_score = float(fid.compute())
    return fid_score


