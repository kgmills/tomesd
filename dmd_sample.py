import torch, tomesd
import random
from diffusers import PixArtAlphaPipeline, DDPMScheduler, PixArtTransformer2DModel


def compute_blk_sum(config_list):
    cost = 0

    for config_int in config_list:
        if config_int == 0:
            cost += 1
        elif config_int == 1:
            cost += 0.5
        elif config_int == 2 or config_int == 3:
            cost += 0.25
        elif config_int == 4:
            cost += 0
        else:
            raise ValueError("Not a right value")
    return cost


pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    scheduler=DDPMScheduler.from_pretrained("PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512", subfolder="scheduler"), 
    transformer=PixArtTransformer2DModel.from_pretrained("PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512", subfolder="transformer", torch_dtype=torch.float16),
    torch_dtype=torch.float16, 
    ).to("cuda")


sample_tsx_list = random.choices([0, 1, 2, 3, 4], k=28)
print(f"Config: {sample_tsx_list} with cost {compute_blk_sum(sample_tsx_list)}")
tomesd.apply_patch(pipe, ratio=0.5, max_downsample=8, merge_attn=True, merge_crossattn=True, merge_mlp=True, tsx_list=sample_tsx_list)

image = pipe(prompt="A lion riding a bike in Paris", 
             num_inference_steps=1, 
             guidance_scale=1, 
             timesteps=[400]).images

image[0].save("pixart_dmd_512_sample_rand.png")

