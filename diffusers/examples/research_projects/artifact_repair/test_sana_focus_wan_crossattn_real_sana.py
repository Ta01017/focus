import os

import pytest
import torch
from diffusers import SanaPipeline

from sana_focus_wan_crossattn import build_focus_wan_model, encode_image_tokens, expand_sana_patch_embedding_for_focus_wan, load_image_encoder_and_processor


@pytest.mark.skipif(os.environ.get('RUN_REAL_SANA_SMOKE') != '1', reason='Set RUN_REAL_SANA_SMOKE=1 to run with local SANA weights.')
def test_real_sana_single_and_dual_forward_once():
    model_id=os.environ.get('SANA_MODEL','Efficient-Large-Model/Sana_600M_1024px_diffusers')
    image_encoder_model=os.environ.get('IMAGE_ENCODER_MODEL','openai/clip-vit-large-patch14')
    pipe=SanaPipeline.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True).to('cuda')
    args=type('Args',(),{'image_encoder_model':image_encoder_model,'image_encoder_subfolder':None,'image_encoder_revision':None,'image_encoder_local_files_only':True})()
    enc,proc=load_image_encoder_and_processor(args,dtype=torch.float32,device='cuda')
    pe,pm,_,_=pipe.encode_prompt('focus fusion',False,device='cuda',clean_caption=False,max_sequence_length=300)
    for mode in ('single','dual'):
        pipe=SanaPipeline.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True).to('cuda')
        c=expand_sana_patch_embedding_for_focus_wan(pipe.transformer,mode)
        model=build_focus_wan_model(pipe.transformer,mode,int(enc.config.hidden_size)).to('cuda').eval()
        z=torch.randn(1,c,32,32,device='cuda')
        model_input=torch.cat([z,z],1) if mode=='single' else torch.cat([z,z,z],1)
        dummy_image=torch.zeros(224,224,3,dtype=torch.uint8).cpu().numpy()
        tok=encode_image_tokens(enc,proc,[dummy_image],torch.device('cuda'),torch.float32)
        imgs=[tok] if mode=='single' else [tok,tok]
        out=model(model_input,pe,torch.zeros(1,device='cuda'),encoder_attention_mask=pm,encoder_hidden_states_images=imgs,return_dict=False)[0]
        assert out.shape[0]==1
