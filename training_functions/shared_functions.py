import torch

def get_time_sample_MF(batch_size):
    mean=torch.ones(batch_size)*-0.4
    std=torch.tensor([1])
    value1=torch.nn.functional.sigmoid(torch.normal(mean,std))
    value2=torch.nn.functional.sigmoid(torch.normal(mean,std))
    r=torch.max(value1,value2).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    t=torch.min(value1,value2).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    return r,t

def get_time_sample_IMM(batch_size):
    mean = torch.ones(batch_size) * -0.4
    std = torch.tensor([1])
    
    value1 = torch.nn.functional.sigmoid(torch.normal(mean, std))
    value2 = torch.nn.functional.sigmoid(torch.normal(mean, std))
    value3 = torch.nn.functional.sigmoid(torch.normal(mean, std))
    
    temp = torch.stack([value1, value2, value3], dim=0) 
    
    r, _ = temp.max(dim=0)  
    t, _ = temp.median(dim=0)  
    s, _ = temp.min(dim=0)  
    
    r = r.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    s = s.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    
    return r, t, s

def interpolate(noise,images,time):
    new_image=noise*time+images*(1-time)
    return new_image

