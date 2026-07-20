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
    validate_focus_wan_checkpoint(cfg, args.condition_mode, model.transformer, cfg.get('share_image_projector', False))
    load_focus_wan_condition_state_dict(model, ckpt_dir/'focus_wan_condition.safetensors')
    lora_dir=ckpt_dir/'transformer_lora'
    if lora_dir.exists(): pipe.load_lora_weights(lora_dir)
    pipe.transformer=model
    return pipe, image_encoder, image_processor, cfg, ckpt_dir


def _as_float(value):
    if hasattr(value, 'detach'):
        return float(value.detach().float().cpu())
    return float(value)


def _as_plain_number(value):
    value = _as_float(value)
    return int(value) if value.is_integer() else value


def sigma_for_timestep(scheduler, timestep, device):
    if not hasattr(scheduler, 'sigmas'):
        raise RuntimeError('Scheduler does not expose sigmas; cannot verify img2img noise schedule consistency.')
    scheduler_timesteps = scheduler.timesteps.to(device)
    scheduler_sigmas = scheduler.sigmas.to(device)
    timestep_tensor = timestep.to(device) if hasattr(timestep, 'to') else torch.tensor(timestep, device=device)
    matches = torch.nonzero(torch.isclose(scheduler_timesteps.float(), timestep_tensor.float()), as_tuple=False).flatten()
    if matches.numel() == 0:
        raise RuntimeError(f'Could not find timestep={_as_float(timestep)} in scheduler.timesteps.')
    return scheduler_sigmas[int(matches[0].item())]


def prepare_timesteps_for_init(pipe, steps, strength, init_mode, schedule_mode, device):
    if steps <= 0:
        raise ValueError('--steps must be positive.')
    pipe.scheduler.set_timesteps(steps, device=device)
    full_timesteps = pipe.scheduler.timesteps
    order = int(getattr(pipe.scheduler, 'order', 1))

    if schedule_mode == 'pipeline_full':
        if init_mode != 'noise':
            raise ValueError(
                "img2img_schedule_mode='pipeline_full' requires init_mode='noise'. "
                "Use img2img_schedule_mode='sliced' for A or focus-composite initialization."
            )
        if strength != 1.0:
            print(
                "[FOCUS_WAN][WARNING] strength is ignored for pipeline_full noise initialization; effective strength is 1.0.",
                flush=True,
            )
        if hasattr(pipe.scheduler, 'set_begin_index'):
            pipe.scheduler.set_begin_index(0)
        first_timestep = full_timesteps[0]
        init_sigma = getattr(pipe.scheduler, 'init_noise_sigma', None)
        if init_sigma is None:
            init_sigma = sigma_for_timestep(pipe.scheduler, first_timestep, device)
        return {
            'timesteps': full_timesteps,
            't_start_index': 0,
            'effective_steps': len(full_timesteps),
            'requested_strength': float(strength),
            'effective_strength': 1.0,
            'initial_noise_timestep': first_timestep,
            'first_denoise_timestep': first_timestep,
            'initial_sigma': init_sigma,
            'schedule_consistent': True,
            'scheduler_begin_index': 0,
            'scheduler_order': order,
        }

    if schedule_mode != 'sliced':
        raise ValueError(f'Unsupported img2img_schedule_mode={schedule_mode!r}')

    if init_mode in ('a', 'focus_composite'):
        if not (0.0 < strength <= 1.0):
            raise ValueError("sliced img2img with init_mode='a' or 'focus_composite' requires 0 < strength <= 1.")
        init_steps = min(max(int(round(steps * strength)), 1), steps)
        t_start = max(steps - init_steps, 0)
        timesteps = full_timesteps[t_start:]
        if len(timesteps) == 0:
            raise RuntimeError('Sliced timestep schedule is empty.')
        if hasattr(pipe.scheduler, 'set_begin_index'):
            pipe.scheduler.set_begin_index(t_start * order)
        first_timestep = timesteps[0]
        initial_timestep = first_timestep
        init_sigma = sigma_for_timestep(pipe.scheduler, first_timestep, device)
        schedule_consistent = torch.isclose(first_timestep.float(), initial_timestep.float()).all().item()
        if not schedule_consistent:
            raise RuntimeError('Initial latent noise timestep does not match the first denoising timestep.')
        return {
            'timesteps': timesteps,
            't_start_index': t_start,
            'effective_steps': len(timesteps),
            'requested_strength': float(strength),
            'effective_strength': float(strength),
            'initial_noise_timestep': initial_timestep,
            'first_denoise_timestep': first_timestep,
            'initial_sigma': init_sigma,
            'schedule_consistent': bool(schedule_consistent),
            'scheduler_begin_index': t_start * order,
            'scheduler_order': order,
        }

    if init_mode == 'noise':
        if hasattr(pipe.scheduler, 'set_begin_index'):
            pipe.scheduler.set_begin_index(0)
        first_timestep = full_timesteps[0]
        init_sigma = getattr(pipe.scheduler, 'init_noise_sigma', None)
        if init_sigma is None:
            init_sigma = sigma_for_timestep(pipe.scheduler, first_timestep, device)
        return {
            'timesteps': full_timesteps,
            't_start_index': 0,
            'effective_steps': len(full_timesteps),
            'requested_strength': float(strength),
            'effective_strength': 1.0,
            'initial_noise_timestep': first_timestep,
            'first_denoise_timestep': first_timestep,
            'initial_sigma': init_sigma,
            'schedule_consistent': True,
            'scheduler_begin_index': 0,
            'scheduler_order': order,
        }
    raise ValueError(f'Unsupported init_mode={init_mode!r}')


def timesteps_for_init(pipe, steps, strength, init_mode, schedule_mode, device):
    plan = prepare_timesteps_for_init(pipe, steps, strength, init_mode, schedule_mode, device)
    return plan['timesteps'], plan['t_start_index'], plan['effective_steps'], plan['initial_noise_timestep']

def load_focus_mask(path, size):
    from PIL import Image
    import numpy as np
    mask = Image.open(path).convert('L').resize(size, Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(mask, dtype='float32') / 255.0).unsqueeze(0)


def make_focus_composite(prepared, focus_a_path, focus_b_path, debug_dir=None):
    import numpy as np
    from PIL import Image
    size = prepared['src'].size
    focus_a = load_focus_mask(focus_a_path, size)
    focus_b = load_focus_mask(focus_b_path, size)
    eps = 1e-6
    denom = focus_a + focus_b
    weight_a = (focus_a / (denom + eps)).clamp(0, 1)
    invalid = denom < eps
    a = torch.from_numpy(np.asarray(prepared['src'], dtype='float32') / 255.0).permute(2, 0, 1)
    b = torch.from_numpy(np.asarray(prepared['ref'], dtype='float32') / 255.0).permute(2, 0, 1)
    composite = weight_a * a + (1.0 - weight_a) * b
    composite = torch.where(invalid.expand_as(composite), a, composite).clamp(0, 1)
    if debug_dir:
        d = Path(debug_dir); d.mkdir(parents=True, exist_ok=True)
        Image.fromarray((focus_a.squeeze(0).numpy() * 255).astype('uint8')).save(d / 'focus_a.png')
        Image.fromarray((focus_b.squeeze(0).numpy() * 255).astype('uint8')).save(d / 'focus_b.png')
        Image.fromarray((weight_a.squeeze(0).numpy() * 255).astype('uint8')).save(d / 'focus_weight_a.png')
    image = Image.fromarray((composite.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    if debug_dir:
        image.save(Path(debug_dir) / 'focus_composite.png')
    return image


@torch.no_grad()
def generate(pipe, image_encoder, image_processor, cfg, args, a_img, b_img=None):
    if args.condition_mode=='dual' and b_img is None: raise ValueError('dual mode requires --image_b')
    init_mode=args.init_mode or 'a'
    if init_mode == 'focus_composite':
        if args.condition_mode != 'dual':
            raise ValueError('focus_composite is only valid for condition_mode=dual.')
        if not args.focus_a or not args.focus_b:
            raise ValueError('focus_composite requires --focus_a and --focus_b.')
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
    schedule_plan=prepare_timesteps_for_init(pipe,args.steps,args.strength,init_mode,args.img2img_schedule_mode,device)
    ts=schedule_plan['timesteps']
    noise=torch.randn(z_a.shape,generator=gen,device=device,dtype=z_a.dtype)
    if init_mode == 'a':
        latent_timestep = schedule_plan['initial_noise_timestep'].expand(z_a.shape[0])
        latents = pipe.scheduler.add_noise(original_samples=z_a, noise=noise, timesteps=latent_timestep)
    elif init_mode == 'focus_composite':
        composite = make_focus_composite(prepared, args.focus_a, args.focus_b, args.debug_dir)
        z_composite = encode_latent(pipe, composite, h, w, device)
        latent_timestep = schedule_plan['initial_noise_timestep'].expand(z_composite.shape[0])
        latents = pipe.scheduler.add_noise(original_samples=z_composite, noise=noise, timesteps=latent_timestep)
    else:
        init_sigma = schedule_plan['initial_sigma']
        if hasattr(init_sigma, 'to'):
            init_sigma = init_sigma.to(device=device, dtype=noise.dtype)
        latents = noise * init_sigma
    if _as_float(schedule_plan['initial_noise_timestep']) != _as_float(schedule_plan['first_denoise_timestep']):
        raise RuntimeError('Initial latent noise timestep does not match the first denoising timestep.')
    scales=[cfg.get('image_cross_attention_scale_a',1.0) if args.image_cross_attention_scale_a is None else args.image_cross_attention_scale_a]
    if args.condition_mode=='dual': scales.append(cfg.get('image_cross_attention_scale_b',1.0) if args.image_cross_attention_scale_b is None else args.image_cross_attention_scale_b)
    for t in ts:
        latent_in=torch.cat([latents,latents],0) if do_cfg else latents
        a_in=torch.cat([z_a,z_a],0) if do_cfg else z_a
        if args.condition_mode=='dual':
            b_in=torch.cat([z_b,z_b],0) if do_cfg else z_b
            assert latent_in.shape == a_in.shape == b_in.shape
            model_input=torch.cat([latent_in,a_in,b_in],1)
            assert model_input.shape[1] == 3 * cfg['latent_channels']
            image_inputs=[torch.cat([tok_a,tok_a],0) if do_cfg else tok_a, torch.cat([tok_b,tok_b],0) if do_cfg else tok_b]
            assert len(image_inputs) == 2
        else:
            assert latent_in.shape == a_in.shape
            model_input=torch.cat([latent_in,a_in],1)
            assert model_input.shape[1] == 2 * cfg['latent_channels']
            image_inputs=[torch.cat([tok_a,tok_a],0) if do_cfg else tok_a]
            assert len(image_inputs) == 1
        tin=t.expand(latent_in.shape[0])*pipe.transformer.config.timestep_scale
        pred=pipe.transformer(hidden_states=model_input.to(pipe.transformer.dtype), encoder_hidden_states=pe.to(pipe.transformer.dtype), encoder_attention_mask=pm, encoder_hidden_states_images=image_inputs, image_cross_attention_scales=scales, timestep=tin, return_dict=False)[0].float()
        if do_cfg:
            u,c=pred.chunk(2); pred=u+args.guidance_scale*(c-u)
        latents=pipe.scheduler.step(pred,t,latents,return_dict=False)[0]
    out=decode_pil(pipe,latents)
    stats={
        'condition_mode':args.condition_mode,
        'checkpoint':str(args.checkpoint),
        'steps':args.steps,
        'strength':args.strength,
        'init_mode':init_mode,
        'img2img_schedule_mode':args.img2img_schedule_mode,
        'schedule mode':args.img2img_schedule_mode,
        'requested_steps':args.steps,
        'effective_steps':int(schedule_plan['effective_steps']),
        'requested_strength':float(schedule_plan['requested_strength']),
        'effective_strength':float(schedule_plan['effective_strength']),
        't_start_index':int(schedule_plan['t_start_index']),
        'initial_noise_timestep':_as_plain_number(schedule_plan['initial_noise_timestep']),
        'first_denoise_timestep':_as_plain_number(schedule_plan['first_denoise_timestep']),
        'initial_sigma':_as_float(schedule_plan['initial_sigma']),
        'schedule_consistent':bool(schedule_plan['schedule_consistent']),
        'scheduler_begin_index':int(schedule_plan['scheduler_begin_index']),
        'scheduler_order':int(schedule_plan['scheduler_order']),
        'latent shapes':{'a':list(z_a.shape),'b':list(z_b.shape) if z_b is not None else None,'final':list(latents.shape)},
        'vision token shapes':{'a':list(tok_a.shape),'b':list(tok_b.shape) if tok_b is not None else None},
        'image gate values':pipe.transformer.image_cross_attention_adapter.gate_values()[0],
        'output finite status':bool(torch.isfinite(latents).all().cpu()),
        'status':'success',
    }
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
