#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch
from diffusers import SanaPipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_repair_utils import add_pretrained_args, load_rgb, preprocess_pair, pretrained_kwargs, restore_output_size, write_json
from sana_artifact_repair_channel_concat import route2_lora_path, tensor_stats
from sana_focus_wan_crossattn import (
    build_focus_wan_model,
    decode_vae_latents,
    encode_image_tokens,
    encode_vae_latents,
    expand_sana_patch_embedding_for_focus_wan,
    focus_prompt,
    load_focus_wan_config,
    load_focus_wan_condition_state_dict,
    load_image_encoder_and_processor,
    num_condition_images,
    validate_focus_wan_checkpoint,
)


def parse_args():
    p=argparse.ArgumentParser(description='Infer Focus Fusion SANA Wan-style latent concat + image cross-attention.')
    p.add_argument('--condition_mode', choices=('single','dual'), required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', default=None)
    p.add_argument('--image_a', required=True)
    p.add_argument('--image_b', default=None)
    p.add_argument('--focus_a', default=None)
    p.add_argument('--focus_b', default=None)
    p.add_argument('--output', required=True)
    p.add_argument('--prompt', default=None)
    p.add_argument('--negative_prompt', default='')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--guidance_scale', type=float, default=1.0)
    p.add_argument('--strength', type=float, default=0.3)
    p.add_argument('--init_mode', choices=('noise','a','focus_composite'), default=None)
    p.add_argument('--img2img_schedule_mode', choices=('pipeline_full','sliced'), default='sliced')
    p.add_argument('--max_pixels', type=int, default=1048576)
    p.add_argument('--size_divisor', type=int, default=32)
    p.add_argument('--downscale_if_exceeds_max_pixels', action='store_true')
    p.add_argument('--restore_to_original_size', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--dtype', choices=('fp32','bf16','fp16','auto'), default='auto')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--debug_dir', default=None)
    p.add_argument('--image_cross_attention_scale_a', type=float, default=None)
    p.add_argument('--image_cross_attention_scale_b', type=float, default=None)
    add_pretrained_args(p)
    return p.parse_args()


def select_dtype(v):
    if v=='fp32': return torch.float32
    if v=='bf16': return torch.bfloat16
    if v=='fp16': return torch.float16
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def encode_latent(pipe, image, h, w, device):
    pixels=pipe.image_processor.preprocess(image, height=h, width=w).to(device, torch.float32)
    return encode_vae_latents(pipe.vae, pixels)


@torch.no_grad()
def decode_pil(pipe, latents):
    return pipe.image_processor.postprocess(decode_vae_latents(pipe.vae, latents), output_type='pil')[0]


def load_pipeline(args, dtype):
    cfg, ckpt_dir = load_focus_wan_config(args.checkpoint)
    if cfg.get('condition_mode') != args.condition_mode:
        raise ValueError(f"checkpoint condition_mode={cfg.get('condition_mode')} is incompatible with requested condition_mode={args.condition_mode}")
    model_id=args.model or cfg['base_model']
    pipe=SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **pretrained_kwargs(args)).to('cuda')
    pipe.vae.to(dtype=torch.float32); pipe.vae.requires_grad_(False).eval(); pipe.text_encoder.requires_grad_(False).eval(); pipe.transformer.requires_grad_(False).eval()
    expand_sana_patch_embedding_for_focus_wan(pipe.transformer, args.condition_mode)
    image_args=argparse.Namespace(image_encoder_model=cfg['image_encoder'], image_encoder_subfolder=None, image_encoder_revision=None, image_encoder_local_files_only=args.local_files_only)
    image_encoder, image_processor=load_image_encoder_and_processor(image_args, dtype=dtype, device='cuda')
    model=build_focus_wan_model(pipe.transformer, args.condition_mode, int(cfg['image_encoder_hidden_size']), cfg.get('image_gate_init',1e-3), cfg.get('share_image_projector',False)).to('cuda', dtype=dtype).eval()
    validate_focus_wan_checkpoint(cfg, args.condition_mode, model.transformer)
    load_focus_wan_condition_state_dict(model, ckpt_dir/'focus_wan_condition.safetensors')
    lora_dir=ckpt_dir/'transformer_lora'
    if lora_dir.exists(): pipe.load_lora_weights(lora_dir)
    pipe.transformer=model
    return pipe, image_encoder, image_processor, cfg, ckpt_dir


def timesteps_for_init(pipe, steps, strength, init_mode, device):
    pipe.scheduler.set_timesteps(steps, device=device)
    ts=pipe.scheduler.timesteps
    if init_mode=='noise': return ts,0,steps
    eff=max(1, min(steps, int(steps*strength)))
    start=max(steps-eff,0)
    if hasattr(pipe.scheduler,'set_begin_index'): pipe.scheduler.set_begin_index(start)
    return ts[start:], start, eff


@torch.no_grad()
def generate(pipe, image_encoder, image_processor, cfg, args, a_img, b_img=None):
    if args.condition_mode=='dual' and b_img is None: raise ValueError('dual mode requires --image_b')
    init_mode=args.init_mode or 'a'
    if init_mode=='focus_composite':
        if not args.focus_a or not args.focus_b: raise ValueError('focus_composite requires focus maps; first Focus WAN version does not use focus maps by default.')
    device=torch.device('cuda')
    gen=torch.Generator(device=device).manual_seed(args.seed)
    prompt=args.prompt or focus_prompt(args.condition_mode)
    do_cfg=args.guidance_scale>1.0
    pe, pm, ne, nm=pipe.encode_prompt(prompt, do_cfg, negative_prompt=args.negative_prompt, num_images_per_prompt=1, device=device, clean_caption=False, max_sequence_length=300)
    if do_cfg: pe=torch.cat([ne,pe],0); pm=torch.cat([nm,pm],0)
    prepared, size_info=preprocess_pair(a_img, b_img or a_img, args.max_pixels, args.size_divisor, args.downscale_if_exceeds_max_pixels)
    w,h=size_info['canvas_size']
    z_a=encode_latent(pipe, prepared['src'], h, w, device)
    z_b=encode_latent(pipe, prepared['ref'], h, w, device) if args.condition_mode=='dual' else None
    tok_a=encode_image_tokens(image_encoder, image_processor, [prepared['src']], device, pipe.transformer.dtype)
    tok_b=encode_image_tokens(image_encoder, image_processor, [prepared['ref']], device, pipe.transformer.dtype) if args.condition_mode=='dual' else None
    ts,start,eff=timesteps_for_init(pipe,args.steps,args.strength,init_mode,device)
    noise=torch.randn(z_a.shape,generator=gen,device=device,dtype=z_a.dtype)
    if init_mode=='a':
        latent_timestep=pipe.scheduler.timesteps[start].expand(z_a.shape[0])
        latents=pipe.scheduler.add_noise(original_samples=z_a, noise=noise, timesteps=latent_timestep)
    else:
        latents=noise
    scales=[cfg.get('image_cross_attention_scale_a',1.0) if args.image_cross_attention_scale_a is None else args.image_cross_attention_scale_a]
    if args.condition_mode=='dual': scales.append(cfg.get('image_cross_attention_scale_b',1.0) if args.image_cross_attention_scale_b is None else args.image_cross_attention_scale_b)
    for t in ts:
        latent_in=torch.cat([latents,latents],0) if do_cfg else latents
        a_in=torch.cat([z_a,z_a],0) if do_cfg else z_a
        if args.condition_mode=='dual':
            b_in=torch.cat([z_b,z_b],0) if do_cfg else z_b
            model_input=torch.cat([latent_in,a_in,b_in],1)
            image_inputs=[torch.cat([tok_a,tok_a],0) if do_cfg else tok_a, torch.cat([tok_b,tok_b],0) if do_cfg else tok_b]
        else:
            model_input=torch.cat([latent_in,a_in],1)
            image_inputs=[torch.cat([tok_a,tok_a],0) if do_cfg else tok_a]
        tin=t.expand(latent_in.shape[0])*pipe.transformer.config.timestep_scale
        pred=pipe.transformer(hidden_states=model_input.to(pipe.transformer.dtype), encoder_hidden_states=pe.to(pipe.transformer.dtype), encoder_attention_mask=pm, encoder_hidden_states_images=image_inputs, image_cross_attention_scales=scales, timestep=tin, return_dict=False)[0].float()
        if do_cfg:
            u,c=pred.chunk(2); pred=u+args.guidance_scale*(c-u)
        latents=pipe.scheduler.step(pred,t,latents,return_dict=False)[0]
    out=decode_pil(pipe,latents)
    stats={'condition_mode':args.condition_mode,'checkpoint':str(args.checkpoint),'steps':args.steps,'strength':args.strength,'init_mode':init_mode,'schedule mode':args.img2img_schedule_mode,'latent shapes':{'a':list(z_a.shape),'b':list(z_b.shape) if z_b is not None else None,'final':list(latents.shape)},'vision token shapes':{'a':list(tok_a.shape),'b':list(tok_b.shape) if tok_b is not None else None},'image gate values':pipe.transformer.image_cross_attention_adapter.gate_values()[0],'output finite status':bool(torch.isfinite(latents).all().cpu()),'effective_steps':eff,'status':'success'}
    return out, stats, prepared, size_info


def main():
    args=parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required.')
    dtype=select_dtype(args.dtype)
    pipe,enc,proc,cfg,ckpt=load_pipeline(args,dtype)
    a=load_rgb(args.image_a); b=load_rgb(args.image_b) if args.image_b else None
    image,stats,prepared,size_info=generate(pipe,enc,proc,cfg,args,a,b)
    if args.debug_dir:
        d=Path(args.debug_dir); d.mkdir(parents=True,exist_ok=True)
        a.save(d/'raw_a.png'); prepared['src'].save(d/'resized_a.png')
        if b is not None: b.save(d/'raw_b.png'); prepared['ref'].save(d/'resized_b.png')
        image.save(d/'final_output.png'); write_json(d/'stats.json',stats)
    image=restore_output_size(image,size_info,args.restore_to_original_size)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); image.save(out)
    write_json(out.with_suffix(out.suffix+'.stats.json'),stats)
    print(f'[FOCUS_WAN] output={out}',flush=True)

if __name__=='__main__': main()
