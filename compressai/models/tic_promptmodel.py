import math
from click import prompt
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import RSTB, MultistageMaskedConv2d, RSTB_PromptModel
from timm.models.layers import trunc_normal_


from .utils import conv, deconv, update_registered_buffers, Demultiplexer, Multiplexer

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class Mask_Gen_Shallow(nn.Module):

    def __init__(self, out_channel=1, mid_channel=64):
        super(Mask_Gen_Shallow, self).__init__()

        #mid_channel*2*mid_channel*2
        self.down1 = nn.Sequential(
            nn.Conv2d(3, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True))
        self.down1_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        #mid_channel*mid_channel
        self.down2 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel*2, mid_channel*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel*2),
            nn.ReLU(inplace=True),)
        self.down2_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        #32*32
        self.down3 = nn.Sequential(
            nn.Conv2d(mid_channel*2, mid_channel*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel*4, mid_channel*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel*4),
            nn.ReLU(inplace=True),)

        self.upsample2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.up2 = nn.Sequential(
            nn.Conv2d(mid_channel*4+mid_channel*2, mid_channel*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel*2, mid_channel*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel*2, mid_channel*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel*2),
            nn.ReLU(inplace=True),)

        self.upsample1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.up1 = nn.Sequential(
            nn.Conv2d(mid_channel*2+mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            )

        self.classifier = nn.Sequential(
                nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=2, padding=1),
                # nn.Sigmoid(),
            )
    # mid_channel*2

    def forward(self, img):
        #128*128
        down1 = self.down1(img)
        down1_pool = self.down1_pool(down1)

        #64*64
        down2 = self.down2(down1_pool)
        down2_pool = self.down2_pool(down2)

        #32*32
        down3 = self.down3(down2_pool)
        
        up2 = self.upsample2(down3)

        up2 = torch.cat((down2,up2), 1)
        up2 = self.up2(up2)

        up1 = self.upsample1(up2)
        up1 = torch.cat((down1,up1), 1)
        up1 = self.up1(up1)

        prob = self.classifier(up1)

        return prob


class TIC_PromptModel_first2(nn.Module):
    """
    Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    """
    def __init__(self, N=128, M=192, prompt_config=None, input_resolution=(256,256)):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False
        self.prompt_config = prompt_config

        if prompt_config is not None:
            architect = prompt_config.ARCHITECT
        else:
            architect = None

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.mask_down = prompt_config.MASK_DOWNSAMPLE
        encblock = RSTB_PromptModel

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a0_prompt = Mask_Gen_Shallow(out_channel= N, mid_channel=32)
        self.g_a0_prompt_layers = nn.ModuleList([conv(N, N, kernel_size=3, stride=self.mask_down) for _ in range(depths[0])])
        self.g_a1 = encblock(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a2_prompt = conv(N, N, kernel_size=3, stride=2)
        self.g_a2_prompt_layers = nn.ModuleList([conv(N, N, kernel_size=3, stride=self.mask_down) for _ in range(depths[1])])
        self.g_a3 = encblock(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config
        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= None 
        )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= None
        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
        )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)
        

        decoder_blocks = [RSTB_PromptModel if prompt_config.MODEL_DECODER and nn+1 in prompt_config.DECODER_BLOCK else RSTB for nn in range(4)]

        self.g_s0 = decoder_blocks[0](dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config=  prompt_config if architect in ['both', 'decoder'] and 1 in prompt_config.DECODER_BLOCK else None
        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = decoder_blocks[1](dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config=  prompt_config if architect in ['both', 'decoder'] and 2 in prompt_config.DECODER_BLOCK else None
        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = decoder_blocks[2](dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'decoder'] and 3 in prompt_config.DECODER_BLOCK else None
        )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = decoder_blocks[3](dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'decoder'] and 4 in prompt_config.DECODER_BLOCK else None
        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        if prompt_config.MODEL_DECODER:
            self.g_s0_prompt = conv(M, M, kernel_size=3, stride=1)
            if 1 in prompt_config.DECODER_BLOCK:
                self.g_s0_prompt_layers = nn.ModuleList([conv(M, M, kernel_size=3, stride=self.mask_down) for _ in range(depths[2])])
            self.g_s2_prompt = deconv(M, N, kernel_size=3, stride=2)
            if 2 in prompt_config.DECODER_BLOCK:
                self.g_s2_prompt_layers = nn.ModuleList([conv(N, N, kernel_size=3, stride=self.mask_down) for _ in range(depths[3])])
            self.g_s4_prompt = deconv(N, N, kernel_size=3, stride=2)
            if 3 in prompt_config.DECODER_BLOCK:
                self.g_s4_prompt_layers = nn.ModuleList([conv(N, N, kernel_size=3, stride=self.mask_down) for _ in range(depths[4])])
            self.g_s6_prompt = deconv(N, N, kernel_size=3, stride=2)
            if 4 in prompt_config.DECODER_BLOCK:
                self.g_s6_prompt_layers = nn.ModuleList([conv(N, N, kernel_size=3, stride=self.mask_down) for _ in range(depths[5])])

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.apply(self._init_weights)   

    def g_a(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
        m = self.g_a0_prompt(x)
        x = self.g_a0(x)
        m_layers = [prompt_layer(m) for prompt_layer in self.g_a0_prompt_layers]
        x, attn = self.g_a1(x, m_layers, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)
        x = self.g_a2(x)
        m = self.g_a2_prompt(m)
        m_layers = [prompt_layer(m) for prompt_layer in self.g_a2_prompt_layers]
        x, attn = self.g_a3(x, m_layers, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)
        x = self.g_a4(x)
        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)
        x = self.g_a6(x)
        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        return x, attns

    def g_s(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        if self.prompt_config.MODEL_DECODER:
            m = self.g_s0_prompt(x)
            if 1 in self.prompt_config.DECODER_BLOCK:
                m_layers = [prompt_layer(m) for prompt_layer in self.g_s0_prompt_layers]
                x, attn = self.g_s0(x, m_layers, (x_size[0]//16, x_size[1]//16))
            else:
                x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
            attns.append(attn)
            x = self.g_s1(x)
            m = self.g_s2_prompt(m)
            if 2 in self.prompt_config.DECODER_BLOCK:
                m_layers = [prompt_layer(m) for prompt_layer in self.g_s2_prompt_layers]
                x, attn = self.g_s2(x, m_layers, (x_size[0]//8, x_size[1]//8))
            else:
                x, attn = self.g_s2(x,  (x_size[0]//8, x_size[1]//8))
            attns.append(attn)
            x = self.g_s3(x)
            m = self.g_s4_prompt(m)
            if 3 in self.prompt_config.DECODER_BLOCK:
                m_layers = [prompt_layer(m) for prompt_layer in self.g_s4_prompt_layers]
                x, attn = self.g_s4(x, m_layers, (x_size[0]//4, x_size[1]//4))
            else:
                x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
            attns.append(attn)
            x = self.g_s5(x)
            m = self.g_s6_prompt(m)
            if 4 in self.prompt_config.DECODER_BLOCK:
                m_layers = [prompt_layer(m) for prompt_layer in self.g_s6_prompt_layers]
                x, attn = self.g_s6(x, m_layers, (x_size[0]//2, x_size[1]//2))
            else:
                x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
            attns.append(attn)
            x = self.g_s7(x)
        else:
            x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
            attns.append(attn)
            x = self.g_s1(x)
            x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
            attns.append(attn)
            x = self.g_s3(x)
            x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
            attns.append(attn)
            x = self.g_s5(x)
            x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
            attns.append(attn)
            x = self.g_s7(x)
        return x, attns

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat, x_size)

        scales_hat, means_hat = params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat, attns_s = self.g_s(y_hat, x_size)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "attn_a": attns_a,
            "attn_s": attns_s,
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat, attns_s = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}