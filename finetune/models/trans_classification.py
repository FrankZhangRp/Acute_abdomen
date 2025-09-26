import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Tuple, Optional

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x, slice_positions):
        embedding_table = self.pos_embedding.squeeze(0)
        pos_emb = embedding_table[slice_positions]
        assert pos_emb.shape == x.shape, f"Shape mismatch: pos_emb {pos_emb.shape}, x {x.shape}"
        return x + pos_emb

class Trans25D_Classification(nn.Module):
    def __init__(self, backbone_model, embed_dim=1536, use_n_blocks=4, use_avgpool=True, num_classes=100, num_decoder_layers=2, trans_nhead=8, trans_dim_feedforward_ratio=4):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone_model
        self.use_n_blocks = use_n_blocks
        self.embed_dim = embed_dim
        self.use_avgpool = use_avgpool
        self.output_dim = self.embed_dim * use_n_blocks + int(use_avgpool) * embed_dim
        self.output_dim = int(self.output_dim)
        
        self.dim_reduction = nn.Linear(self.output_dim, embed_dim)
        
        self.pos_encoder = LearnablePositionalEncoding(embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=trans_nhead,
            dim_feedforward=trans_dim_feedforward_ratio*embed_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.autocast_ctx = partial(torch.autocast, enabled=True, dtype=torch.half, device_type="cuda")
    
    def forward(self, images, orig_z):
        batch_features = []
        batch_lengths = []
        batch_positions = []
        
        for i in range(len(orig_z)):
            images_i = images[i][orig_z[i]:]
            if len(images_i) == 0:
                images_i = images[i][:1]
                positions = [0]
            else:
                positions = list(range(orig_z[i], orig_z[i] + len(images_i)))
            
            batch_size = 192
            num_slices = len(images_i)
            slice_features = []
            
            for j in range(0, num_slices, batch_size):
                batch_end = min(j + batch_size, num_slices)
                with torch.no_grad():
                    with self.autocast_ctx():
                        features = self.backbone.get_intermediate_layers(
                            images_i[j:batch_end], 
                            self.use_n_blocks,
                            return_class_token=True
                        )
                        linear_input = create_linear_input(features, self.use_n_blocks, self.use_avgpool)
                        slice_features.append(linear_input)
            
            slice_features_tensor = torch.cat(slice_features, dim=0)
            
            slice_features_reduced = self.dim_reduction(slice_features_tensor)
            
            batch_features.append(slice_features_reduced)
            batch_lengths.append(slice_features_reduced.shape[0])
            batch_positions.append(positions)
        
        max_length = max(batch_lengths)
        padded_features_list = []
        attention_mask_list = []
        padded_positions_list = []
        
        for features, length, positions in zip(batch_features, batch_lengths, batch_positions):
            padding = torch.zeros(max_length - length, self.embed_dim, device=features.device)
            padded = torch.cat([features, padding], dim=0)
            padded_features_list.append(padded)
            
            pos_padding = [0] * (max_length - length)
            padded_positions_list.append(positions + pos_padding)
            
            mask = torch.ones(max_length, device=features.device)
            mask[length:] = 0
            attention_mask_list.append(mask)

        padded_features = torch.stack(padded_features_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)
        padded_positions = torch.tensor(padded_positions_list, device=padded_features.device)
        
        padded_features = self.pos_encoder(padded_features, padded_positions)
        
        cls_tokens = self.cls_token.expand(padded_features.shape[0], -1, -1)
        
        tgt_key_padding_mask = (attention_mask == 0)
        
        decoded_features = self.transformer_decoder(
            cls_tokens,
            padded_features,
            tgt_mask=None,
            memory_key_padding_mask=tgt_key_padding_mask
        )
                
        output = self.classifier(decoded_features.squeeze(1))
        
        return output
    
    def train(self, mode=True):
        self.backbone.train(False)
        self.dim_reduction.train(mode)
        self.transformer_decoder.train(mode)
        self.classifier.train(mode)
    
    def eval(self):
        self.train(mode=False)
    
    def get_optimizer_param_groups(self, lr):
        param_groups = [
            {
                "params": [
                    *self.dim_reduction.parameters(),
                    *self.transformer_decoder.parameters(),
                    *self.classifier.parameters(),
                    *self.pos_encoder.parameters(),
                    self.cls_token
                ],
                "lr": lr
            },
        ]
        return param_groups