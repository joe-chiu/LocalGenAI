# script for in-painting, tested on SDXL and SD1.5 fine tunes (Dreamshaper)
# SDXL uses a different pipeline

import random
import argparse

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import torch

def main(args):
    if args.seed is None:
        args.seed = random.randint(0, 2**30)

    InpaintPipeline = StableDiffusionXLInpaintPipeline if "stable-diffusion-xl" in args.model else StableDiffusionInpaintPipeline

    pipe = InpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16, 
        variant="fp16",
        use_safetensors=True,
        local_files_only=True
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True)
 
    while True:
        init_image = load_image(args.init_image)
        mask_image = load_image(args.mask)
        images = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            guidance_scale=args.guidance_scale,
            # clip_skip=args.clip_skip
        ).images
        images[0].save(args.output, "JPEG")

        if args.interactive is False:
            break;

        # caller can use this output to end waiting
        print("GENERATION_DONE")
        # wait and set up next task
        print("next prompt:")
        prompt = input();
        args.prompt = prompt
        print("next output:")
        output = input();
        args.output = output
        print("init image:")
        image1 = input();
        args.init_image = image1
        print("mask image:")
        image2 = input();
        args.mask = image2
        args.seed += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Lykon/dreamshaper-8-inpainting", help="model name")
    parser.add_argument("--seed", type=int, default=None, help="random seed for generating consistent images per prompt")
    # parser.add_argument("--clip-skip", type=int, default=None, help="number of layers to be skipped for CLIP")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="num inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--prompt", type=str, default="a painting of a beautiful graceful woman with long hair", help="prompt")
    parser.add_argument("--negative-prompt", type=str, default=None, help="negative prompt")
    parser.add_argument("--init-image", type=str, default=None, help="path to initial image")
    parser.add_argument("--mask", type=str, default=None, help="mask of the region to inpaint on the initial image")
    parser.add_argument("--output", type=str, default="output.jpg", help="output image name")
    parser.add_argument("--interactive", action="store_true", help="interactive mode to do multiple generations")

    args = parser.parse_args()

    main(args)