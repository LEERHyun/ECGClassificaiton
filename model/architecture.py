import torch
import torch.nn as nn
import torch.nn.functional as F
#from . import arch_util
import numbers
from einops import rearrange
import torchsummary

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Restormer Module
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##########################################################################

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


"""class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias"""

class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    
    def forward(self, x):
        # x: (N, C, L)
        mu = x.mean(1, keepdim=True)  # (N, 1, L)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1) * y + self.bias.view(1, -1, 1)
        
    
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv1d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv1d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.sg = SimpleGate()
        self.project_out = nn.Conv1d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        #x = self.dwconv(x)
        #x = self.sg(x)
        x = self.project_out(x)
        return x


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv1d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv1d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,l = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) l -> b head c l', head=self.num_heads)
        k = rearrange(k, 'b (head c) l -> b head c l', head=self.num_heads)
        v = rearrange(v, 'b (head c) l -> b head c l', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c l -> b (head c) l', head=self.num_heads, l=l)

        out = self.project_out(out)
        return out

## TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm1d(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm1d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#NAFNet Module
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#SimpleGate Module
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

#LayerNorm1D

    
#NAFBlock Architecture
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv1d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True) #
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv1d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv1d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm1d(c)
        self.norm2 = LayerNorm1d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    
class DoubleConv1D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down1D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is (batch, channels, length)
        diffL = x2.size()[2] - x1.size()[2]
        
        x1 = F.pad(x1, [diffL // 2, diffL - diffL // 2])
        
        # concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, n_channels=1, n_classes=5):
        """
        1D U-Net for ECG signal classification
        
        Args:
            n_channels: Number of input channels (1 for single-lead ECG)
            n_classes: Number of output classes (Normal, APB, PVC, LBBB, RBBB)
        """
        super(UNet1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv1D(n_channels, 16)
        self.down1 = Down1D(16, 32)
        self.down2 = Down1D(32, 64)
        self.down3 = Down1D(64, 128)
        
        # Bottleneck
        self.down4 = Down1D(128, 256)
        
        # Decoder
        self.up1 = Up1D(256, 128)
        self.up2 = Up1D(128, 64)
        self.up3 = Up1D(64, 32)
        self.up4 = Up1D(32, 16)
        
        # Output layer
        self.outc = nn.Conv1d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 16 channels
        x2 = self.down1(x1)   # 32 channels
        x3 = self.down2(x2)   # 64 channels
        x4 = self.down3(x3)   # 128 channels
        x5 = self.down4(x4)   # 256 channels (bottleneck)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # 128 channels
        x = self.up2(x, x3)   # 64 channels
        x = self.up3(x, x2)   # 32 channels
        x = self.up4(x, x1)   # 16 channels
        
        # Output
        logits = self.outc(x)
        return logits
#----------------------------------------------------------------------------------------
#HybridModel
#----------------------------------------------------------------------------------------
class HybridNAFNet(nn.Module):

    def __init__(self, 
                 img_channel=1,
                 out_channel = 6,
                   width=4, 
                   ffn_expansion_factor =2.66,
                   bias = False,
                   heads = [1,2,4,8]
                   ):
        super().__init__()
        self.intro = nn.Conv1d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)# 1 =>4
        self.ending = nn.Conv1d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True) # 4=>1

        #1->4
        self.intro = nn.Conv1d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)  
        #Level 1 Encoder
        chan = width
        self.encoder_level1 = nn.Sequential(TransformerBlock(dim=chan, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias),
                                            NAFBlock(chan))
        self.down1_2 = nn.Conv1d(chan, 2*chan, 2, 2) ## DownSample Level 1 to Level 2
        
        chan = chan*2
        
        #Level 2 Encoder
        self.encoder_level2 = nn.Sequential(TransformerBlock(dim=chan, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias),
                                            NAFBlock(chan))
        self.down2_3 = nn.Conv1d(chan, 2*chan, 2, 2) ## From Level 2 to Level 3
        
        chan = chan*2
        
        #Level 3 Encoder
        self.encoder_level3 = nn.Sequential(TransformerBlock(dim=chan, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias),
                                            NAFBlock(chan))
        self.down3_4 = nn.Conv1d(chan, 2*chan, 2, 2) ## From Level 3 to Level 4
        
        chan = chan*2
        
        #Middle Block
        self.middle = nn.Sequential(TransformerBlock(dim=chan, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias),
                                            NAFBlock(chan))
        
        #Level 3 Decoder        
        self.up4_3 = nn.ConvTranspose1d(chan, chan//2, kernel_size = 2,stride=2, bias=bias) ## From Level 4 to Level 3
        chan = chan//2
        
        self.decoder_level3 = nn.Sequential(TransformerBlock(dim=chan, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias),
                                            NAFBlock(chan))

        #Level 2 Decoder
        
        self.up3_2 = nn.ConvTranspose1d(chan, chan//2, kernel_size = 2,stride=2, bias=bias) ## From Level 3 to Level 2
        
        chan = chan//2        
        
        self.decoder_level2 = nn.Sequential(TransformerBlock(dim=chan, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias),
                                            NAFBlock(chan)) 
        
        #Level 1 Decoder
        self.up2_1 = nn.ConvTranspose1d(chan, chan//2, kernel_size = 2,stride=2, bias=bias) ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        chan = chan//2
        
        self.decoder_level1 = nn.Sequential(TransformerBlock(dim=chan, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias),
                                            NAFBlock(chan))

        #Ending Width -> 6 classes
        self.ending = nn.Conv1d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True) # 4=>6    
        
        self.padder_size = 2 **len(heads)


    def forward(self, inp):
        
        B, C, L = inp.shape
        inp = self.check_image_size(inp)        
        inp_enc_level1 = self.intro(inp) # 32->3
        
        #Level 1 Encoder
        out_enc_level1 = self.encoder_level1(inp_enc_level1) #Skip Connection Value Level 1
        inp_enc_level2 = self.down1_2(out_enc_level1) #Dowsample 1->2
        
        #Level 2 Encoder
        out_enc_level2 = self.encoder_level2(inp_enc_level2) #Skip Connection Value Level 2
        inp_enc_level3 = self.down2_3(out_enc_level2) #Downsample 2->3
        
        #Level 3 Encoder
        out_enc_level3 = self.encoder_level3(inp_enc_level3) #Skip Connection Value Level 3
        inp_enc_level4 = self.down3_4(out_enc_level3) #Downsample 3->4
        
        #Middle Block
        latent = self.middle(inp_enc_level4) 
        
        #Level 3 Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = inp_dec_level3 + out_enc_level3 #Skip connection Level 3
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        #Level 2 Decoder
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = inp_dec_level2 + out_enc_level2 #Skip connection Level 2
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        #Level 2 Decoder
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = inp_dec_level1 + out_enc_level1 #Skip connection Level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # Level1 -> 6 channel
        ending_out = self.ending(out_dec_level1)
        
        output = ending_out           
        return output[:,:, :L]

    def check_image_size(self, x):
        _, _, l = x.size()
        mod_pad_l = (self.padder_size - l % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_l))
        return x



if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    img_channel = 1
    width = 4

    #Calculate Model Complexity---------------------------------------------------------------------------------------------------------------------------------------------------------
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    custom = HybridNAFNet(img_channel=1,out_channel=6,width=4)
    custom.to(device)
    #custom = HybridNAFNet()
    torchsummary.summary(custom,(1,1024))

    

    custom.to(device)
    #Model Summary
    flops, params = get_model_complexity_info(custom, (1,1024), verbose=False, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational Complexity: ', flops))
    print('{:<30}  {:<8}'.format('Parameters: ', params))

    #unet = UNet1D(n_channels=1,n_classes=6)
    #unet.to(device)
    #torchsummary.summary(unet,(1,1024))
    #flops, params = get_model_complexity_info(unet, (1,1024), verbose=False, print_per_layer_stat=False)
    #print('{:<30}  {:<8}'.format('Computational Complexity: ', flops))
    #print('{:<30}  {:<8}'.format('Parameters: ', params))
