from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from layers.Embed import PositionalEmbedding


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.patch_num = (configs.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_pretrain:
            self.gpt2_im = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2_time = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        else:
            self.gpt2_im = GPT2Model(GPT2Config())
            self.gpt2_time = GPT2Model(GPT2Config())

        self.gpt2_im.h = self.gpt2_im.h[:configs.gpt_layers]
        self.gpt2_time.h = self.gpt2_time.h[:configs.gpt_layers]
        print("gpt2 = {}".format(self.gpt2_im))

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.gpt2_im.to(device=device)
            self.gpt2_time.to(device=device)

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.temporal_in_layer = nn.Linear(configs.enc_in, configs.d_model)
        self.position_embedding = PositionalEmbedding(768)
        self.metric_in_layer = nn.Linear(configs.seq_len, configs.d_model)

        if self.task_name == 'anomaly_detection':
            self.ln_proj = nn.LayerNorm(configs.d_ff)
            self.out_layer_im = nn.Linear(
                configs.d_ff, 
                configs.c_out_im, 
                bias=True)
            self.out_layer_time = nn.Linear(
                configs.d_ff, 
                configs.c_out_time, 
                bias=True)
            self.out_layer = nn.Linear(
                configs.c_out_time * 2, 
                configs.c_out_time, 
                bias=True)
          
    def freeze_step2(self, configs):
        for i, (name, param) in enumerate(self.gpt2_im.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for i, (name, param) in enumerate(self.gpt2_time.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False
    def freeze_step1(self):
        for i, (name, param) in enumerate(self.gpt2_im.named_parameters()):
            param.requires_grad = False
        for i, (name, param) in enumerate(self.gpt2_time.named_parameters()):
            param.requires_grad = False

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        
        dec_out = self.anomaly_detection_dual(x_enc, x_mark_enc)
        return dec_out  # [B, L, D]
     

  
    def anomaly_detection_dual(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc_im = rearrange(x_enc, 'b l m -> b m l')
        x_enc_time = x_enc

        # intermetric
        enc_out_im = self.metric_in_layer(x_enc_im)
        # enc_out_im = torch.nn.functional.pad(x_enc_im, (0, 768-x_enc_im.shape[-1]))
        outputs_im = self.gpt2_im(inputs_embeds=enc_out_im).last_hidden_state
        outputs_im = outputs_im[:, :, :self.d_ff]
        # outputs = self.ln_proj(outputs)
        dec_out_im = self.out_layer_im(outputs_im)
        dec_out_im = rearrange(dec_out_im, 'b m l -> b l m')

        # time
        #position enc
        # enc_out_time = self.enc_embedding(x_enc_time, None)
        enc_out_time = self.temporal_in_layer(x_enc_time) + self.position_embedding(x_enc_time)
        #padding
        # enc_out_time = torch.nn.functional.pad(x_enc_time, (0, 768-x_enc_time.shape[-1]))
        outputs_time = self.gpt2_time(inputs_embeds=enc_out_time).last_hidden_state
        outputs_time = outputs_time[:, :, :self.d_ff]
        dec_out_time = self.out_layer_time(outputs_time)

        # De-Normalization from Non-stationary Transformer
       
        dec_con = torch.cat((dec_out_time, dec_out_im),dim=-1)
        dec_out = self.out_layer(dec_con)
        dec_out = dec_out * stdev
        dec_out = dec_out + means
        return dec_out

   