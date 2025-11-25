import torch
from diffusers import AutoencoderKL
from torch.amp import autocast
from training_functions import mean_flow
from evaluation import FID
from dataset import kaggle_dataloader
from models.attention_U_Net import UNet
from training_functions import shared_functions

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")

auto_encoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)


if __name__ == "__main__":
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
    else:
        autocast_dtype=torch.float16

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    num_epochs=1
    batch_size=2
    model=UNet(False)
    model.to(device)
    auto_encoder.to(device)
    train_loader=kaggle_dataloader.get_train_loader(batch_size,2,batch_size*2,2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0001)
    print("starting training")
    number_batches_trained=0
    for images,labels in train_loader:
        t,s=shared_functions.get_time_sample_MF(batch_size)
        with autocast(device_type=device,dtype=autocast_dtype):
            loss,jvp=mean_flow.training_step_MF(images,labels,model,auto_encoder,optimizer,batch_size,t,s,device)
        print(loss)
        number_batches_trained+=1
        if (number_batches_trained>10):
            break

    test_loader=kaggle_dataloader.get_test_loader(batch_size=10,num_worker=2)
    print(FID.evaluate_FID(model,auto_encoder,test_loader,10,100,device))





