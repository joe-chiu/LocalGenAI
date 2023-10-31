# script for text-to-image, tested on SDXL and SD1.5 fine tunes (Dreamshaper)

# 1. add an --interactive command line, which would load the pipeline
# 2. generate if prompt are specified via the command line
# 3. if not or when first generation is complete, waiting for new input over new prompt
# take in the input over console input and as a single json string

import random
import argparse
import sys

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

def main(args):
    if args.seed is None:
        args.seed = random.randint(0, 2**30)

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None,
        local_files_only=True
    ).to("cuda")
        
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True)
 
    while True:
        images = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
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
        print("next output:")
        output = input();
        args.prompt = prompt
        args.output = output
        args.seed += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="model name")
    parser.add_argument("--seed", type=int, default=None, help="random seed for generating consistent images per prompt")
    # parser.add_argument("--clip-skip", type=int, default=None, help="number of layers to be skipped for CLIP")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="num inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--prompt", type=str, default="a painting of a beautiful graceful woman with long hair", help="prompt")
    parser.add_argument("--negative-prompt", type=str, default="disfigured hands, ugly, deformed, signature, watermark, stamp", help="negative prompt")
    parser.add_argument("--output", type=str, default="output.jpg", help="output image name")
    parser.add_argument("--interactive", action="store_true", help="interactive mode to do multiple generations")

    args = parser.parse_args()

    main(args)