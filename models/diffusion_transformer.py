import torch
from torch import nn


class image_embeddings_learnable_frequencies(nn.Module):
    def __init__(self,number_patches,embedding_size):
        super().__init__()
        self.dimensions = int(number_patches**0.5)
        self.register_buffer("patch_numbers", torch.arange(1, number_patches + 1, dtype=torch.float32))
        self.fourrier_values_b = nn.Parameter(torch.randn(2,2,embedding_size//4))
        horizontal=self.patch_numbers%(self.dimensions)+1
        vertical=torch.ceil(self.patch_numbers//self.dimensions)+1
        self.register_buffer("horizontal", horizontal)
        self.register_buffer("vertical", vertical)

    def forward(self):
        cos_values_horizontal=torch.cos(torch.outer(self.horizontal,self.fourrier_values_b[0][0]))
        sin_values_horizontal=torch.sin(torch.outer(self.horizontal,self.fourrier_values_b[0][1]))
        cos_values_vertical=torch.cos(torch.outer(self.vertical,self.fourrier_values_b[1][0]))
        sin_values_vertical=torch.sin(torch.outer(self.vertical,self.fourrier_values_b[1][1]))

        return torch.cat((cos_values_horizontal,sin_values_horizontal,cos_values_vertical,sin_values_vertical),dim=1)


class image_attention(nn.Module):
    def __init__(self,patch_size,number_patches,embedding_dimensions):
        super(image_attention,self).__init__()
        self.patch_size=patch_size
        self.number_pixels_per_patch=patch_size*patch_size
        self.query_embedding = nn.Linear(self.number_pixels_per_patch+embedding_dimensions, self.number_pixels_per_patch)
        self.key_embedding = nn.Linear(self.number_pixels_per_patch+embedding_dimensions, self.number_pixels_per_patch)
        self.value_embedding = nn.Linear(self.number_pixels_per_patch+embedding_dimensions, self.number_pixels_per_patch)
        self.pos_embed = image_embeddings_learnable_frequencies(number_patches,embedding_dimensions)

    def forward(self,x):
        b, c, h, w = x.shape

        patches = nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)            
        L = patches.shape[-1]                                    

        patches = patches.view(b, c, self.number_pixels_per_patch, L).permute(0, 1, 3, 2)  
        
        positional_embedding=self.pos_embed()

        positional_embedding=positional_embedding.unsqueeze(0).unsqueeze(0).expand(b, c, L, -1)

        patches = torch.cat((patches,positional_embedding), dim=-1)   

        q = self.query_embedding(patches)  
        k = self.key_embedding(patches)
        v = self.value_embedding(patches)

        head_dim = q.size(-1)
    
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  
        
        scores = scores - scores.amax(dim=-1, keepdim=True)
        attn_weights = torch.softmax(scores, dim=-1)
        
        attn_out = torch.matmul(attn_weights, v)  
        
        attn_out = attn_out.permute(0, 1, 3, 2).contiguous().view(b, c * self.number_pixels_per_patch, L)

        new_image = nn.functional.fold(attn_out, output_size=(h, w), kernel_size=self.patch_size, stride=self.patch_size)  
        
        return new_image

class time_embedding(nn.Module):
    def __init__(self):
        super(time_embedding,self).__init__()
        self.size=64
        self.fourrier_values_b = nn.Parameter(torch.randn(1,2,self.size))
        self.module=nn.Sequential(nn.Linear(self.size*4,256),
                                  nn.BatchNorm1d(256,affine=False, track_running_stats=False),
                                  nn.SiLU(),
                                  nn.Linear(256,256),
                                  nn.BatchNorm1d(256,affine=False, track_running_stats=False),
                                  nn.SiLU(),
                                  nn.Linear(256,256),
                                  nn.BatchNorm1d(256,affine=False, track_running_stats=False),
                                  nn.SiLU())
        self.skip_embedding_noise=nn.Linear(256,9)
        self.skip_embedding_network_output=nn.Linear(256,9)

        nn.init.zeros_(self.skip_embedding_noise.weight)
        nn.init.ones_(self.skip_embedding_noise.bias)  

        nn.init.zeros_(self.skip_embedding_network_output.weight)
        nn.init.constant_(self.skip_embedding_network_output.bias, 0.01)

    def forward(self,input_time,out_time):
        input_time=input_time.squeeze()
        out_time=out_time.squeeze()
        times = torch.stack([input_time, out_time], dim=1).unsqueeze(-1)
        sin_values=torch.sin(self.fourrier_values_b*times)
        cos_values=torch.cos(self.fourrier_values_b*times)
        fourrier_output=torch.concat((sin_values,cos_values),dim=1).flatten(start_dim=1)
        embedding=self.module(fourrier_output)
        skip_weights_noise=self.skip_embedding_noise(embedding)
        skip_weights_network_output=self.skip_embedding_network_output(embedding)
        return embedding,skip_weights_noise,skip_weights_network_output


class UNet_Module_with_attention(nn.Module):
    def __init__(self, in_channels, out_channels,patch_size,number_patches,embedding_dimensions):
        super(UNet_Module_with_attention,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.film_embedding_1=nn.Linear(256,in_channels*2)
        self.film_embedding_2=nn.Linear(256,out_channels*2)
        self.skip_embedding=nn.Linear(256,1)
        self.conv1=nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn_1=nn.BatchNorm2d(in_channels,affine=False, track_running_stats=False)
        self.attention=image_attention(patch_size,number_patches,embedding_dimensions)
        self.conv2=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn_2=nn.BatchNorm2d(out_channels,affine=False, track_running_stats=False)

    def forward(self,x,time_embedding):
        film_embedding_1=self.film_embedding_1(time_embedding)
        film_embedding_2=self.film_embedding_2(time_embedding)
        scale1,bias1=film_embedding_1.split(self.in_channels, dim=-1)
        scale1 = scale1.unsqueeze(-1).unsqueeze(-1)
        bias1  = bias1.unsqueeze(-1).unsqueeze(-1)
        output=self.conv1(x)*scale1+bias1
        output=self.bn_1(output)
        output=nn.functional.silu(output)
        output=self.attention(output)
        scale2, bias2 = film_embedding_2.split(self.in_channels, dim=-1)
        scale2 = scale2.unsqueeze(-1).unsqueeze(-1)
        bias2  = bias2.unsqueeze(-1).unsqueeze(-1)
        output=self.conv2(output)*scale2+bias2
        output=self.bn_2(output)
        output=nn.functional.silu(output)
        return output+x*self.skip_embedding(time_embedding).unsqueeze(-1).unsqueeze(-1)

class UNet_Module(nn.Module):
    def __init__(self, in_channels, out_channels,stride,upsample_factor):
        super(UNet_Module,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.film_embedding_1=nn.Linear(256,in_channels*2)
        self.film_embedding_2=nn.Linear(256,out_channels*2)
        self.upsample_factor=upsample_factor
        self.conv1=nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn_1=nn.BatchNorm2d(in_channels,affine=False, track_running_stats=False)
        self.conv2=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=stride)
        self.bn_2=nn.BatchNorm2d(out_channels,affine=False, track_running_stats=False)



    def forward(self,x,time_embedding):
        film_embedding_1=self.film_embedding_1(time_embedding)
        film_embedding_2=self.film_embedding_2(time_embedding)
        scale1 = film_embedding_1[:,:self.in_channels]
        bias1 = film_embedding_1[:,self.in_channels:]
        scale1 = scale1.unsqueeze(-1).unsqueeze(-1)
        bias1  = bias1.unsqueeze(-1).unsqueeze(-1)
        output=self.conv1(x)*scale1+bias1
        output=self.bn_1(output)
        output=nn.functional.silu(output)
        output = nn.functional.interpolate(output, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
        scale2 = film_embedding_2[:,:self.out_channels]
        bias2 = film_embedding_2[:,self.out_channels:]
        scale2 = scale2.unsqueeze(-1).unsqueeze(-1)
        bias2  = bias2.unsqueeze(-1).unsqueeze(-1)
        output=self.conv2(output)*scale2+bias2
        output=self.bn_2(output)
        output=nn.functional.silu(output)
        return output
    
    
class UNet(nn.Module):
    def __init__(self,use_time_scaling=True):
        super(UNet,self).__init__()

        self.multiplier_time_scaling=0
        if (use_time_scaling):
            self.multiplier_time_scaling=1
            

        self.time_embedding=time_embedding()

        self.conv_input = nn.Conv2d(4, 4, kernel_size=3, padding=1)

        self.module1_down = UNet_Module_with_attention(4, 4, 4,64,32)
        self.module2_down = UNet_Module(4, 8, 2, 1)
        self.module3_down = UNet_Module_with_attention(8, 8, 4,16,16)
        self.module4_down = UNet_Module(8, 16, 2, 1)
        self.module5_down = UNet_Module_with_attention(16, 16, 2,16,16)
        self.module6_down = UNet_Module(16, 32, 2, 1)

        self.module_in_between1 = UNet_Module(32, 64, 1, 1)
        self.module_in_between2 = UNet_Module_with_attention(64, 64, 2,4,4)
        self.module_in_between3 = UNet_Module(64, 64, 1, 1)
        self.module_in_between4 = UNet_Module_with_attention(64, 64, 2,4,4)  
        self.module_in_between5 = UNet_Module(64, 32, 1, 1)
      
        self.module6_up = UNet_Module(32, 16, 1, 2)
        self.module5_up = UNet_Module_with_attention(16, 16, 2,16,16)
        self.module4_up = UNet_Module(16, 8, 1, 2)
        self.module3_up = UNet_Module_with_attention(8, 8, 4,16,16)
        self.module2_up = UNet_Module(8, 4, 1, 2)
        self.module1_up = UNet_Module_with_attention(4, 4, 4,64,16)

        self.conv_output = nn.Conv2d(4, 4, kernel_size=3, padding=1)

        self.bn=nn.BatchNorm2d(4,affine=False, track_running_stats=False)


    def forward(self, input_image, t, s):
        time_embedding,skip_weights_noise,skip_weights_network_output=self.time_embedding(t,s)

        x=torch.nn.functional.silu(self.conv_input(input_image))

        x1 = self.module1_down(x,time_embedding)                          
        x2 = self.module2_down(x1,time_embedding)                         
        x3 = self.module3_down(x2,time_embedding)                          
        x4 = self.module4_down(x3,time_embedding)                          
        x5 = self.module5_down(x4,time_embedding)  
        x6 = self.module6_down(x5,time_embedding)                       

        x7 = self.module_in_between1(x6,time_embedding)
        x8 = self.module_in_between2(x7,time_embedding)
        x9 = self.module_in_between3(x8,time_embedding)
        x10 = self.module_in_between4(x9,time_embedding)*skip_weights_network_output[:,0:1, None, None]+x7*skip_weights_noise[:,0:1, None, None]
        x11 = self.module_in_between5(x10,time_embedding)*skip_weights_network_output[:,1:2, None, None]+x6*skip_weights_noise[:,1:2, None, None]

        y6 = self.module6_up(x11,time_embedding)*skip_weights_network_output[:,2:3, None, None] + x5*skip_weights_noise[:,2:3, None, None]
        y5 = self.module5_up(y6,time_embedding)*skip_weights_network_output[:,3:4, None, None] + x4*skip_weights_noise[:,3:4, None, None]
        y4 = self.module4_up(y5,time_embedding)*skip_weights_network_output[:,4:5, None, None] + x3*skip_weights_noise[:,4:5, None, None]
        y3 = self.module3_up(y4,time_embedding)*skip_weights_network_output[:,5:6, None, None] + x2*skip_weights_noise[:,5:6, None, None]
        y2 = self.module2_up(y3,time_embedding)*skip_weights_network_output[:,6:7, None, None] + x1*skip_weights_noise[:,6:7, None, None]
        y1 = self.module1_up(y2,time_embedding)*skip_weights_network_output[:,7:8, None, None] + x*skip_weights_noise[:,7:8, None, None]

        out = self.conv_output(self.bn(y1))*((-1-t)*self.multiplier_time_scaling+skip_weights_network_output[:,8:9, None, None]*(0.01+1-self.multiplier_time_scaling))+input_image*(skip_weights_noise[:,8:9, None, None]*(0.01+1-self.multiplier_time_scaling)+1)
        return out
