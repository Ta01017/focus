#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))

from artifact_repair_utils import add_pretrained_args, load_metadata, load_rgb, resolve_dataset_path, write_json
from infer_sana_focus_wan_crossattn import generate, load_pipeline, select_dtype


def parse_args():
    p=argparse.ArgumentParser(description='Batch infer Focus Fusion SANA WAN cross-attention.')
    p.add_argument('--condition_mode', choices=('single','dual'), required=True)
    p.add_argument('--metadata_path', '--dataset_metadata_path', dest='metadata_path', required=True)
    p.add_argument('--dataset_base_path', default='.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', default=None)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--target_key', default='image')
    p.add_argument('--edit_key', default='edit_image')
    p.add_argument('--prompt_key', default='prompt')
    p.add_argument('--start_index', type=int, default=0)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--continue_on_error', action='store_true')
    p.add_argument('--save_debug', action='store_true')
    p.add_argument('--save_comparison', action='store_true')
    p.add_argument('--prompt', default=None)
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
    add_pretrained_args(p)
    return p.parse_args()


def concat_images(items):
    h=max(x.height for x in items); w=sum(x.width for x in items)
    canvas=Image.new('RGB',(w,h),'white'); x=0
    for im in items:
        canvas.paste(im,(x,0)); x+=im.width
    return canvas


def main():
    args=parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required.')
    dtype=select_dtype(args.dtype)
    pipe,enc,proc,cfg,ckpt=load_pipeline(args,dtype)
    records=load_metadata(args.metadata_path)
    end=None if args.max_samples is None else args.start_index+args.max_samples
    selected=list(enumerate(records[args.start_index:end], start=args.start_index))
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; failed=[]
    for index, sample in selected:
        output=outdir/f'{index:06d}.png'
        try:
            edits=sample.get(args.edit_key)
            need=1 if args.condition_mode=='single' else 2
            if not isinstance(edits,list) or len(edits)<need: raise ValueError(f'edit_image length insufficient for {args.condition_mode}')
            a_path=resolve_dataset_path(edits[0], args.dataset_base_path, record_index=index, field_name=f"{args.edit_key}[0]")
            b_path=resolve_dataset_path(edits[1], args.dataset_base_path, record_index=index, field_name=f"{args.edit_key}[1]") if len(edits)>1 else None
            gt_path=resolve_dataset_path(sample[args.target_key], args.dataset_base_path, record_index=index, field_name=args.target_key)
            focus_a_path=resolve_dataset_path(edits[2], args.dataset_base_path, record_index=index, field_name=f"{args.edit_key}[2]") if len(edits)>2 else None
            focus_b_path=resolve_dataset_path(edits[3], args.dataset_base_path, record_index=index, field_name=f"{args.edit_key}[3]") if len(edits)>3 else None
            a=load_rgb(a_path); b=load_rgb(b_path) if b_path else None
            run_args=argparse.Namespace(**vars(args)); run_args.image_a=str(a_path); run_args.image_b=str(b_path) if b_path else None; run_args.focus_a=str(focus_a_path) if focus_a_path else None; run_args.focus_b=str(focus_b_path) if focus_b_path else None; run_args.output=str(output); run_args.seed=args.seed+index; run_args.debug_dir=str(outdir/f'{index:06d}_debug') if args.save_debug else None
            if run_args.prompt is None and sample.get(args.prompt_key): run_args.prompt=sample[args.prompt_key]
            image, stats, prepared, size_info=generate(pipe,enc,proc,cfg,run_args,a,b)
            from artifact_repair_utils import restore_output_size
            image=restore_output_size(image,size_info,args.restore_to_original_size)
            image.save(output)
            comp=None
            if args.save_comparison:
                gt=load_rgb(gt_path).resize(image.size, Image.Resampling.LANCZOS); avis=a.resize(image.size, Image.Resampling.LANCZOS)
                if args.condition_mode=='dual':
                    bvis=b.resize(image.size, Image.Resampling.LANCZOS); comp=outdir/f'{index:06d}_comparison.png'; concat_images([avis,bvis,gt,image]).save(comp)
                else:
                    comp=outdir/f'{index:06d}_comparison.png'; concat_images([avis,gt,image]).save(comp)
            rows.append({'index':index,'src/A path':str(a_path),'B path':str(b_path) if b_path else None,'GT path':str(gt_path),'focus_a path':str(focus_a_path) if focus_a_path else None,'focus_b path':str(focus_b_path) if focus_b_path else None,'output path':str(output),'comparison path':str(comp) if comp else None,'status':'success','error':None,'condition_mode':args.condition_mode,'checkpoint':str(args.checkpoint)})
            print(f'[FOCUS_WAN][BATCH] ok index={index} output={output}',flush=True)
        except Exception as e:
            failed.append(index); rows.append({'index':index,'output path':str(output),'status':'error','error':repr(e),'condition_mode':args.condition_mode,'checkpoint':str(args.checkpoint)})
            print(f'[FOCUS_WAN][BATCH] error index={index}: {e}',flush=True)
            if not args.continue_on_error: raise
    summary={'success count':sum(1 for r in rows if r['status']=='success'),'failed count':len(failed),'failed indices':failed,'results':rows}
    write_json(outdir/'results.json', summary)
    print(f'[FOCUS_WAN][BATCH] results={outdir/"results.json"}',flush=True)

if __name__=='__main__': main()
