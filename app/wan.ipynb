# Install required packages
!pip install -q diffusers torch pillow gradio

import torch
import PIL.Image
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
import gradio as gr

# Load model once
model_id = "AlekseyCalvin/WanVACE_1.3B_nf4_umT5fp8_Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
pipe.to("cuda")

def prepare_video_and_mask(first_img: PIL.Image.Image, last_img: PIL.Image.Image, height: int, width: int, num_frames: int):
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = []
    frames.append(first_img)
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask

def generate_video(prompt, negative_prompt, first_frame, last_frame, height, width, num_frames, num_steps, guidance_scale, seed):
    # Load default images if not provided
    if first_frame is None:
        first_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png")
    if last_frame is None:
        last_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png")
    
    video_frames, mask = prepare_video_and_mask(first_frame, last_frame, height, width, num_frames)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    output = pipe(
        video=video_frames,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]
    
    video_path = "output.mp4"
    export_to_video(output, video_path, fps=16)
    return video_path

# Default values
default_prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
default_negative = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# Gradio UI
iface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(label="Prompt", value=default_prompt, lines=3),
        gr.Textbox(label="Negative Prompt", value=default_negative, lines=3),
        gr.Image(label="First Frame (upload or use default)", type="pil"),
        gr.Image(label="Last Frame (upload or use default)", type="pil"),
        gr.Slider(label="Height", minimum=256, maximum=1024, step=64, value=512),
        gr.Slider(label="Width", minimum=256, maximum=1024, step=64, value=512),
        gr.Slider(label="Number of Frames", minimum=1, maximum=100, step=1, value=81),
        gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=5, value=30),
        gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, step=0.5, value=5.0),
        gr.Number(label="Seed", value=42),
    ],
    outputs=gr.Video(label="Generated Video"),
    title="WanVACE Video Generator",
    description="Generate videos using WanVACE model. Upload first and last frames or use defaults."
)

iface.launch(share=True)
