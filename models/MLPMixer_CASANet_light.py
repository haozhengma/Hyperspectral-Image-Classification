import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchEmbeddings(nn.Module):

    def __init__(
        self,
        patch_size: int,
        hidden_dim: int,
        channels: int
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"Input shape: {x.shape}")
        x = self.proj[0](x)  # 仅卷积操作
        # print(f"After Conv2d: {x.shape}")
        x = self.proj[1](x)  # 仅 Rearrange 操作
        # print(f"After Rearrange: {x.shape}")
        return x


class GlobalAveragePooling(nn.Module):

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ##print(x.shape)
        return x.mean(dim=self.dim)


class Classifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.model.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ##print(x.shape)
        return self.model(x)


class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(x.shape)
        return self.model(x)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        #print(x.shape)
        return x


# Modality Specific Part + Modality Correlative Part(MSI&SAR Patches)
class MixerBlock_s1s2(nn.Module):

    def __init__(
        self,
        num_patches: int, # 36
        num_channels: int, # 128
        tokens_hidden_dim: int, # 64
        channels_hidden_dim: int # 512
    ):
        super().__init__()
        self.cablock = CABlock()
        self.token_mixing_img_s1 = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img_s1 = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channels_hidden_dim)
        )
        self.token_mixing_img_s2 = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img_s2 = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channels_hidden_dim)
        )

    def forward(self, x1, x2):

        # Specific: SAR Patches
        x1_token_img = x1 + self.token_mixing_img_s1(x1)
        x1_img = x1_token_img + self.channel_mixing_img_s1(x1_token_img)

        # Specific: MSI Patches
        x2_token_img = x2 + self.token_mixing_img_s2(x2)
        x2_img = x2_token_img + self.channel_mixing_img_s2(x2_token_img)

        # Correlative: MSI&SAR Patches
        att = self.cablock(x1, x2)
        #print(x1_img.shape)
        #print(x2_img.shape)
        #print(att.shape)
        return x1_img, x2_img, att


# Modality Correlative Part(Channel Concatenation Patches)
class MixerBlock_multimodal(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channels_hidden_dim)
        )
        self.proj = nn.Sequential(
            Rearrange("b (h w) c -> b c h w",h=3),  # yancheng(r=5) h=3
            nn.Conv2d(num_channels * 2, num_channels, 1),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x_self, x_last):
        x_token_img = x_self + self.token_mixing_img(x_self)
        x1 = x_token_img + self.channel_mixing_img(x_token_img)
        #print(x1.shape)
        x2 = torch.concatenate((x1, x_last), 2)
        #print(x2.shape)
        x2 = self.proj(x2)
        return x2


# basic attention
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x1, x2):
        # [batch_size, num_patches + 1, total_embed_dim]
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv1 = self.qkv(x1).reshape(B1, N1, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(x2).reshape(B2, N2, 3, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  # make torchscript happy (cannot use tensor as tuple)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x1 = (attn1 @ v2).transpose(1, 2).reshape(B1, N1, C1)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        x2 = (attn2 @ v1).transpose(1, 2).reshape(B2, N2, C2)
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)
        #print(x1.shape)
        #print(x2.shape)
        return x1, x2


# cross attention module
class CABlock(nn.Module):
    def __init__(
        # self,
        # dim=128,
        # hidden_dim=128,
        # num_patches=36,
        # tokens_hidden_dim=512,
        # norm_layer=nn.LayerNorm
        self,
        dim=128,
        hidden_dim=128,  # 修改为实际使用的 dim
        num_patches=int,
        tokens_hidden_dim=64,  # 修改为实际使用的 tokens_hidden_dim#######################################################64
        norm_layer=nn.LayerNorm
                 ):
        super(CABlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=hidden_dim)
        self.norm2 = norm_layer(dim)
        # self.mlp = MLPBlock(input_dim=num_patches, hidden_dim=tokens_hidden_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim),
            # Rearrange("b p c -> b c p"),
            MLPBlock(dim, tokens_hidden_dim),
            # Rearrange("b c p -> b p c")
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(dim),
            # Rearrange("b p c -> b c p"),
            MLPBlock(dim, tokens_hidden_dim),
            # Rearrange("b c p -> b p c")
        )

    def forward(self, x1, x2):
        x1att, x2att = self.attn(self.norm1(x1), self.norm2(x2))
        x1 = x2att + x1
        x2 = x1att + x2

        x1 = x1 + self.block1(x1)
        x2 = x2 + self.block2(x2)
        att = torch.cat([x1, x2], 2)
        #print(att.shape)
        return att


# CASA Block
class CASA_Block(nn.Module):

    def __init__(
        self,
        # num_classes: int,
        # image_size: int = 128,
        # channels: int = 128,
        # patch_size: int = 9,
        # num_layers: int = 4,
        # hidden_dim: int = 128,#128
        # tokens_hidden_dim: int = 64,#64
        # channels_hidden_dim: int = 512#512
        num_classes=int,
        image_size=int,
        channels=int,  # 减少通道数
        patch_size=int,
        num_layers: int = 2,  # 减少层数
        hidden_dim: int = 128,
        tokens_hidden_dim: int = 32,  # 减少 tokens 隐藏维度#######################64-->32
        channels_hidden_dim: int = 256  # 减少通道隐藏维度#######################512-->256
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        # num_patches = 9
        # #print('num of patches', num_patches)
        self.embed_img = PatchEmbeddings(patch_size, hidden_dim, channels)
        self.embed_sv = PatchEmbeddings(patch_size, hidden_dim, channels)
        # self.embed_fuse = PatchEmbeddings(patch_size, hidden_dim, channels*2)

        self.Mixer_s1s2_1 = MixerBlock_s1s2(
                num_patches=num_patches,  # 36
                num_channels=hidden_dim,  # 128
                tokens_hidden_dim=tokens_hidden_dim,  # 64
                channels_hidden_dim=channels_hidden_dim  # 512
            )
        self.Mixer_s1s2_2 = MixerBlock_s1s2(
            num_patches=num_patches,
            num_channels=hidden_dim,
            tokens_hidden_dim=tokens_hidden_dim,
            channels_hidden_dim=channels_hidden_dim
            )
        # self.Mixer_s1s2_3 = MixerBlock_s1s2(
        #     num_patches=num_patches,
        #     num_channels=hidden_dim,
        #     tokens_hidden_dim=tokens_hidden_dim,
        #     channels_hidden_dim=channels_hidden_dim
        #     )
        # self.Mixer_s1s2_4 = MixerBlock_s1s2(
        #     num_patches=num_patches,
        #     num_channels=hidden_dim,
        #     tokens_hidden_dim=tokens_hidden_dim,
        #     channels_hidden_dim=channels_hidden_dim
        #     )
        self.Mixer_1 = MixerBlock_multimodal(
                num_patches=num_patches,
                num_channels=hidden_dim*2,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        self.Mixer_2 = MixerBlock_multimodal(
                num_patches=num_patches,
                num_channels=hidden_dim*2,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        # self.Mixer_3 = MixerBlock_multimodal(
        #         num_patches=num_patches,
        #         num_channels=hidden_dim*2,
        #         tokens_hidden_dim=tokens_hidden_dim,
        #         channels_hidden_dim=channels_hidden_dim
        #     )
        # self.Mixer_4 = MixerBlock_multimodal(
        #         num_patches=num_patches,
        #         num_channels=hidden_dim*2,
        #         tokens_hidden_dim=tokens_hidden_dim,
        #         channels_hidden_dim=channels_hidden_dim
        #     )

        self.norm = nn.LayerNorm(hidden_dim*4)

        self.pool = GlobalAveragePooling(dim=1)

        self.classifier = Classifier(hidden_dim*4, num_classes)

    def forward(self, img: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        s1 = self.embed_img(img)           # [b, p, c]
        s2 = self.embed_sv(sv)           # [b, p, c]
        ##print('after embed', s1.shape, s2.shape)

        s1_1, s2_1, att_1 = self.Mixer_s1s2_1(s1, s2)
        mm_1 = self.Mixer_1(torch.concatenate((s1, s2), 2), att_1)
        #print(mm_1.shape)

        s1_2, s2_2, att_2 = self.Mixer_s1s2_2(s1_1, s2_1)
        mm_2 = self.Mixer_2(mm_1, att_2)

        # s1_3, s2_3, att_3 = self.Mixer_s1s2_3(s1_2, s2_2)
        # mm_3 = self.Mixer_3(mm_2, att_3)

        # s1_4, s2_4, att_4 = self.Mixer_s1s2_4(s1_3, s2_3)#####################################去掉了两次
        # mm_4 = self.Mixer_4(mm_3, att_4)

        x = torch.concatenate((s1_2, s2_2, mm_2), 2)
        # x = torch.concatenate((s1_4, s2_4, mm_4), 2)
        x = self.norm(x)
        x = self.pool(x)            # [b, c]

        x = self.classifier(x)      # [b, num_classes]
        #print(x.shape)
        return x


if __name__ == "__main__":

    # pm = PatchEmbeddings(patch_size=9, hidden_dim=128, channels=128)
    x1 = torch.randn(32, 64, 7, 7)
    x2 = torch.randn(32, 64, 7, 7)
    model = CASA_Block(num_classes=18, image_size=9, channels=64, patch_size=3)
    print(model)
    #print('model1 parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    output = model(x1, x2)
    # #print(output)




