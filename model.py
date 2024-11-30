import torch
import torch.nn as nn #引入torch.nn模块
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode #引入多步参数LIF神经元和多步LIF神经元
from timm.models.layers import to_2tuple, trunc_normal_, DropPath 
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model 

__all__ = ['Spikingformer'] #定义一个列表

#MLP类是一个多层感知机（Multi-Layer Perceptron）的实现，它是一种前馈神经网络，包含一个输入层，一个或多个隐藏层，以及一个输出层。每一层都是全连接的。
class MLP(nn.Module): #定义一个MLP类，继承nn.Module
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.): #初始化函数，输入特征，隐藏特征，输出特征，drop
        super().__init__() #调用父类的初始化函数
        out_features = out_features or in_features #如果输出特征为空，则输出特征等于输入特征
        hidden_features = hidden_features or in_features #如果隐藏特征为空，则隐藏特征等于输入特征
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy') #定义一个多步LIF神经元
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1) #定义一个卷积层
        self.mlp1_bn = nn.BatchNorm2d(hidden_features) #定义一个BatchNorm2d层
        #批量归一化的主要思想是对每个批次的数据进行归一化处理，使其均值为0，方差为1，从而使得网络在训练过程中对输入数据的分布不敏感。

        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy') #定义一个多步LIF神经元
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1) #定义一个卷积层
        self.mlp2_bn = nn.BatchNorm2d(out_features) #定义一个BatchNorm2d层，可以加速训练过程

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):#定义前向传播函数
        T, B, C, H, W = x.shape #获取输入数据的形状
#T:序列长度，B:batch size，C:通道数，H:高度，W:宽度
        x = self.mlp1_lif(x) #通过第一个隐藏层
        x = self.mlp1_conv(x.flatten(0, 1)) #用来把数据展平，然后通过卷积层
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W) #通过BatchNorm2d层

        x = self.mlp2_lif(x) #通过第二个隐藏层
        x = self.mlp2_conv(x.flatten(0, 1)) #用来把数据展平，然后通过卷积层
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x
#在这个MLP类中，定义了两个隐藏层，每个隐藏层包含一个LIF（Leaky Integrate-and-Fire）神经元，一个卷积层和一个批量归一化层。LIF神经元是一种脉冲神经元，它模拟了生物神经元的行为。
#卷积层用于提取输入数据的特征，批量归一化层用于加速训练过程并提高模型的性能。
#在forward方法中，首先将输入数据通过第一个隐藏层，然后将结果通过第二个隐藏层，最后返回输出结果。这个过程模拟了神经网络的前向传播过程

class SpikingSelfAttention(nn.Module): #定义一个SpikingSelfAttention类，继承nn.Module 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1): #初始化函数，输入维度，头数，是否有偏置，qk_scale，attn_drop，proj_drop，sr_ratio
    #dim:输入维度，num_heads:头数，qkv_bias:是否有偏置，qk_scale:qk缩放，attn_drop:注意力机制的dropout，proj_drop:投影的dropout，sr_ratio:缩放比例
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}." #断言，dim应该可以被num_heads整除

        self.dim = dim #输入维度
        self.num_heads = num_heads #头数

        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy') #定义一个多步LIF神经元
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False) #定义一个卷积层
        self.q_bn = nn.BatchNorm1d(dim) #定义一个BatchNorm1d层

        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy') #定义一个多步LIF神经元
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False) 
        self.k_bn = nn.BatchNorm1d(dim)

        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
    #在这个SpikingSelfAttention类中，定义了一个多头注意力机制，包含了查询、键、值的卷积层和批量归一化层，以及一个投影的卷积层和批量归一化层。每一个分别用来提取输入数据的特征。


    def forward(self, x): #定义前向传播函数
        T, B, C, H, W = x.shape #获取输入数据的形状
        x = self.proj_lif(x) #通过投影层

        x = x.flatten(3) #展平
        T, B, C, N = x.shape #获取输入数据的形状
        x_for_qkv = x.flatten(0, 1) #展平
 
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
#上面在前向传播函数中，首先通过投影层，然后将输入数据展平，接着通过查询、键、值的卷积层和批量归一化层，最后通过多步LIF神经元，得到查询、键、值。
        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
#这里的qkv分别是注意力机制中的结构，q是查询，k是键，v是值
        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x
        #在这个SpikingSelfAttention类中，定义了一个多头注意力机制，包含了查询、键、值的卷积层和批量归一化层，以及一个投影的卷积层和批量归一化层。每一个分别用来提取输入数据的特征。


class SpikingTransformer(nn.Module): #定义一个SpikingTransformer类，继承nn.Module
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., #初始化函数，输入维度，头数，mlp_ratio，是否有偏置，qk_scale，drop，attn_drop
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,  
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio) #定义一个多头注意力机制
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop) #定义一个多层感知机

    def forward(self, x): #定义前向传播函数
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikingTokenizer(nn.Module): #定义一个SpikingTokenizer类，继承nn.Module
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256): #初始化函数，图像尺寸，patch尺寸，输入通道数，嵌入维度
        super().__init__() #调用父类的初始化函数
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)

        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)

        self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj2_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x): #定义前向传播函数
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1)) #用来把数据展平，然后通过卷积层
        x = self.proj_bn(x).reshape(T, B, -1, H, W) #通过BatchNorm2d层

        x = self.proj1_lif(x).flatten(0, 1)
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H, W) #通过BatchNorm2d层

        x = self.proj2_lif(x).flatten(0, 1)
        x = self.proj2_conv(x)
        x = self.proj2_bn(x)
        x = self.proj2_mp(x).reshape(T, B, -1, int(H / 2), int(W / 2))

        x = self.proj3_lif(x).flatten(0, 1)
        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_mp(x).reshape(T, B, -1, int(H / 4), int(W / 4))

        x = self.proj4_lif(x).flatten(0, 1)
        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, int(H / 4), int(W / 4))

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class vit_snn(nn.Module): #定义一个vit_snn类，继承nn.Module
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 ): #初始化函数，图像尺寸，patch尺寸，输入通道数，类别数，嵌入维度，头数，mlp_ratio，是否有偏置，qk_scale，drop_rate，attn_drop_rate，drop_path_rate，norm_layer，depths，sr_ratios，T，pretrained_cfg
        super().__init__()
        self.num_classes = num_classes #类别数
        self.depths = depths #深度
        self.T = T #时间步长
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule，随机深度衰减规则

        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims) #定义一个SpikingTokenizer类
        num_patches = patch_embed.num_patches #patch数量
        block = nn.ModuleList([SpikingTransformer( #定义一个SpikingTransformer类
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)]) #循环depths次

        setattr(self, f"patch_embed", patch_embed) #设置patch_embed属性
        setattr(self, f"block", block) #设置block属性

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity() #定义一个线性层
        self.apply(self._init_weights) #初始化权重

    def _init_weights(self, m): #初始化权重
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: #如果是线性层并且有偏置
                nn.init.constant_(m.bias, 0) #初始化偏置为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0 )

    def forward_features(self, x): #定义前向传播函数
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed") #获取patch_embed属性

        x, (H, W) = patch_embed(x) #通过patch_embed
        for blk in block: #循环block
            x = blk(x) #通过block
        return x.flatten(3).mean(3)     # B, C, H, W -> B, C
 
    def forward(self, x): #定义前向传播函数
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1) #将输入数据展开
        x = self.forward_features(x) #通过forward_features
        x = self.head(x.mean(0)) #通过线性层
        return x


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__': #测试代码
    input = torch.randn(2, 3, 32, 32).cuda() #生成一个随机张量
    model = create_model( #创建一个模型
        'Spikingformer', #模型名称
        pretrained=False, #是否预训练
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=100, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4, 
    ).cuda() # 创建一个模型上面进行了一些参数的设置


    # print the output
    model.eval()
    y = model(input)
    print(y.shape) #打印输出的形状
    print('Test Good!')











