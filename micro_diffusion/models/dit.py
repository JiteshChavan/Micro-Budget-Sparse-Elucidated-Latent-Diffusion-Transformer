from typing import List
# this works only if DaVinci is set in pythonpath
# dont do .modules
from modules import CaptionProjection, CrossAttention, MLP, MLPConfig, SelfAttention, T2IFinalLayer, TimeStepEmbedder
from modules import create_norm, get_2d_sincos_pos_embed, get_mask, mask_out_token, modulate, fill_out_masked_tokens
from modules import MLPConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from timm.models.vision_transformer import PatchEmbed


class AttentionBlockPromptEmbedding(nn.Module):
    """Attention block specifically for processing prompt embeddings.
    
    Args:
        n_embd(int) : Input and output dimension
        head_size (int): Channels per attention head
        # TODO: change name, fan_hidden multiplier
        mlp_n_hidden_mult (float): Multiplier for feed-forward network hidden dimension w.r.t input dimension
        # TODO: change name, something to do with layerwise scaling of n_embd
        hidden_base_mult(int): Round feed-forward network fan_hidden up to nearest multiple of this value
        norm_eps (float): Epsilon value for layer normalization (avoid division by 0)
        use_bias (bool): whether to use bias in linear layers
    """

    def __init__ (self, n_embd, head_size, mlp_n_hidden_mult, hidden_base_mult, norm_eps, use_bias):
        super().__init__()
        assert n_embd % head_size == 0, f"n_embd:{n_embd} should be divisble by head_size:{head_size}"

        self.n_embd = n_embd
        self.n_head = n_embd // head_size

        self.ln1 = create_norm ("layernorm", n_embd, eps=norm_eps)
        # Not specifying n_hidden implies qkv will project n_embd into 3xn_embd
        self.attn = SelfAttention (n_embd, self.n_head, qkv_bias=use_bias, norm_eps=norm_eps)

        self.ln2 = create_norm ("layernorm", n_embd, eps=norm_eps)
        self.ffn = FeedForwardNetwork (n_embd, n_hidden=int(n_embd*mlp_n_hidden_mult), hidden_base_mult=hidden_base_mult, use_bias=use_bias)
    
    def forward (self, x, **kwargs):
        x = x + self.attn (self.ln1(x))
        x = x + self.ffn (self.ln2(x))
        return x
    
    def custom_init (self, init_std:float =0.02):
        self.attn.custom_init(init_std=init_std)
        self.ffn.custom_init(init_std=init_std)

class FeedForwardNetwork(nn.Module):
    """Feed-forward block with SiLU activation
        n_embd: input/output channel dimensionality
        n_hidden : hidden layer dimensionality
        hidden_base_mult : make n_hidden nearest next multiple of this hidden_base_mult
        use_bias: Use bias in linear layers or not
    """
    def __init__(self, n_embd, n_hidden, hidden_base_mult, use_bias):
        super().__init__()
        
        self.n_hidden = int (2*n_hidden/3)
        self.n_hidden = hidden_base_mult * ((n_hidden + hidden_base_mult - 1) // hidden_base_mult)
        self.fc1 = nn.Linear (n_embd, self.n_hidden, bias=use_bias)
        self.fc2 = nn.Linear (n_embd, self.n_hidden, bias=use_bias)
        self.fc3 = nn.Linear(self.n_hidden, n_embd, bias=use_bias)

    def forward (self, x):
        return self.fc3 (F.silu(self.fc1(x)) * self.fc2(x))
    
    def custom_init (self, init_std:float):
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
        for linear in (self.fc2, self.fc3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
    
class FeedForwardECMoe (nn.Module):
    """Expert Choice style Mixture of Experts feed forward layer with GELU activation
    
    Args:
        num_experts (int) : number of experts in the layer
        expert_capacity (float): capacity factor determining tokens per expert
        n_embd (int): Input and output dimension
        n_hidden (int): hidden layer channels
        hidden_base_mult (int) : Round hidden dimension upto next nearest multiple of this value
    """

    def __init__(self, num_experts, expert_capacity:float, n_embd, n_hidden, hidden_base_mult):
        self.num_experts = num_experts

        # scaling that restricts or boosts capacity of an expert (eg: 0.5x or 1.5x) we default to 1.0 I think
        self.expert_capacity = expert_capacity
        
        self.n_embd = n_embd
        self.hidden_base_mult = hidden_base_mult
        self.n_hidden = hidden_base_mult * ((n_hidden + hidden_base_mult -1) // hidden_base_mult)

        # to get softmax over num_experts for T tokens
        self.gate = nn.Linear (n_embd, num_experts, bias=False) # bias false makes sense in case model wants to 1 hot on experts

        # each expert goes from n_embd to n_hidden
        self.w1 = nn.Parameter (torch.ones (num_experts, n_embd, n_hidden))
        # non linear activation
        self.gelu = nn.GELU()
        # each expert goes from n_hidden to n_embd
        self.w2 = nn.Parameter (torch.ones (num_experts, n_hidden, n_embd))
    
    def forward (self, x:torch.Tensor):
        # extract shapes
        assert x.dim() == 3
        B, T, C = x.shape

        
        tokens_per_expert = int( self.expert_capacity * T / self.num_experts)

        # get scores, softmax for each token over experts, how appealing is an expert to each of the T tokens
        scores = self.gate (x) # (B, T, E) E is number of experts
        probs = F.softmax (scores, dim=-1) # probs for T tokens across experts

        probs_expert_looking_at_all_tokens = probs.permute(0, 2, 1) # (B, E, T)

        # gather top-tokens-per-expert
        # probs, idices
        expert_specific_token_probs, expert_specific_tokens = torch.topk (probs_expert_looking_at_all_tokens, tokens_per_expert, dim=-1)
        # (B, E, l)       (B, E, l) l is tokens per expert
        # create one hot vectors of T size for the selected tokens, so that we can extract from B,T,C
        # to construct xin for moe
        extract_from_x_one_hot = F.one_hot(expert_specific_tokens, num_classes=T).float() # (B, E, l, T)

        # Goal: (B, E, l C) from x
        xin = torch.einsum ('BElT, BTC -> BElC', extract_from_x_one_hot, x)
        
        # forward
        activation = torch.einsum ('BElC, ECH -> BElH', xin, self.w1) # (B, E, l, H)
        activation = self.gelu(activation)
        activation = torch.einsum ('BElH, EHC -> BElC', activation, self.w2) # (B, E, l, C)

        # scale the activation with gating score probs, so that stronger experts have greater influence on the outputs
        activation = activation * expert_specific_token_probs.unsqueeze(dim=-1)

        # use inner product to combine results of T tokens from all the different experts
        out = torch.einsum ('BElC, BElT -> BTC', activation, extract_from_x_one_hot)
        return out
    
    def custom_init (self, init_std:float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)

class DiTBlock (nn.Module):
    """DiT transformer block comprising Attention and MLP blocks. It supports choosing between dense feed-forward
    and expert choice style Mixture of Experts feed forward blocks.

    Args:
        n_embd (int): Input and Output dimension of the block
        head_size (int) : Channels per attention head
        mlp_n_hidden_mult (float):  Multiplier for feed-forward network hidden dimension w.r.t input dimension
        qkv_n_hidden_mult (float): Multiplier for dimension in qkv layers in attention block
        hidden_base_mult (int): Round hidden dimension upto next nearest multiple of this value
        modulated_sigma_t_embd (int): Dimension of pooled caption embeddings
        norm_eps (float): Epsilon for layer normalization
        
        Setting this flag true, respects depth of blocks and initalizes projection layer weights with respect to block
        indices, to counteract effect of residual additions
        better than GPT2, monolithic init through out network which doesnt respect depth of blocks

        depth_init (bool): Whether to initialize weights of the last layer in MLP/Attention block based on block index
        
        block_index (int): Index of this block in dit model, starts with 0
        num_blocks_for_weight_init (int): total number of blocks in the dit model
        scale_cx_attn_n_hidden (bool): whether to scale cross attention qkv dimension using qkv_n_hidden_mult
        use_bias (bool) : whether to use bias in linear layers
        is_moe (bool) : whether to use mixture of experts for MLP block
        num_experts (int) : Number of experts if using MoE block
        expert_capacity (float) : Capacity factor for each expert if using MoE block
    """

    def __init__ (
            self,
            n_embd:int,
            head_size:int,
            mlp_n_hidden_mult:float,
            qkv_n_hidden_mult:float,
            hidden_base_mult:int,
            modulated_sigma_t_embd:int,
            norm_eps:float,
            depth_init:bool,
            block_index:int,
            num_blocks_for_weight_init:int,
            scale_cx_attn_n_hidden:bool,
            use_bias:bool,
            is_moe:bool,
            num_experts:int,
            expert_capacity:float
    ):
        super().__init__()
        self.n_embd = n_embd

        assert n_embd % head_size == 0, f"n_embd:{n_embd} is not divisible by head_size:{head_size}"

        if qkv_n_hidden_mult == 1:
            qkv_n_hidden = n_embd
        else:
            # round qkv_n_hidden up to be next nearest multiple of 2*head_size
            # qkv_n_hidden % headsize = 0
            qkv_n_hidden = 2*head_size* (( int (qkv_n_hidden_mult *n_embd) + (2*head_size) - 1) // (2*head_size))
        

        self.ln1 = create_norm ("layernorm", n_embd, eps=norm_eps)
        # self attention on token sequence input to DiT
        self.attn = SelfAttention (n_embd, qkv_n_hidden // head_size, qkv_bias=use_bias, n_hidden=qkv_n_hidden, norm_eps=norm_eps)
        self.ln2 = create_norm ("layernorm", dim=n_embd, eps=norm_eps)

        # cross attention TODO: check if its done against time(noise) or caption embeddings
        # TODO: cross attention (IF run on CAPTIONS has to have 0 contribution (use_bias=False) when caption set to 0 for CFG)
        cx_attn_n_hidden = qkv_n_hidden if scale_cx_attn_n_hidden else n_embd
        self.cx_attn = CrossAttention (n_embd=n_embd, n_head=cx_attn_n_hidden//head_size, n_hidden=cx_attn_n_hidden, norm_eps=norm_eps, qkv_bias=use_bias)

        self.ln3 = create_norm ("layernorm", dim=n_embd, eps=norm_eps)

        mlp_n_hidden = int (n_embd * mlp_n_hidden_mult)
        self.mlp = (
            FeedForwardECMoe (num_experts=num_experts, expert_capacity=expert_capacity, 
                              n_embd=n_embd, n_hidden=mlp_n_hidden, 
                              hidden_base_mult= hidden_base_mult) if is_moe else
            FeedForwardNetwork (n_embd=n_embd, n_hidden=mlp_n_hidden, hidden_base_mult=hidden_base_mult, use_bias=use_bias)
        )

        self.AdaLN_modulation = nn.Sequential (
            nn.GELU(approximate="tanh"),
            nn.Linear (modulated_sigma_t_embd, 6*n_embd)
        )

        # Equivalent to NANO_GPT_SCALE_INIT for projecting layers
        # std dev from which projection layers weights are to be initated
        # we hard code rest of the layers as 0.02
        self.weight_init_std = (
            0.02 / (3 * (block_index + 1)) ** 0.5 if depth_init else
            0.02 / (3 * num_blocks_for_weight_init) ** 0.5 # GPT 2 style
        )

    #change name of args after setting up initial aggregation
    # x is forward information stream
    # c is aggregated caption and time to certain abstraction
    # TODO: introspect later t is lets just say time (B, C) for now

    def forward(self, x:torch.Tensor, c:torch.Tensor, t:torch.Tensor):
        # extract 3 gamma, 3 beta from pooled caption embd?
        # each has shape (B, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp =self.AdaLN_modulation (t).split (self.n_embd, dim=-1) # split along 6c so we get 6 splits

        # x (B, T, C) after modulation with scale (B, [1],C) and shift (B, [1], C) //[1] unsqueezed
        # the shape of x is still (B, T, C) it needs to be gated with (B, [1], C) so that the shape will still be (B, T, C)
        # B, T, C = B, 1, C * B, T, C
        x = x + gate_msa.unqueeze(1) * self.attn(modulate( self.ln1(x), shift=shift_msa, scale=scale_msa))
        x = x + self.cx_attn (self.ln2(x), c) # caption condition c already has extracted information from timestep at presetup
        x = x + gate_mlp.unsqueeze(1)*self.mlp(modulate(self.ln3(x), shift=shift_mlp, scale=scale_mlp)) # (B, T, C)

        return x # (B, T, C)
    
    def custom_init(self):
        # reset affine parameters to 1 and 0 respectively
        for norm in (self.ln1, self.ln2, self.ln3):
            norm.reset_parameters()
        
        # initialize layers with HARDCODED calculate correct fan to keep exploding variance in check
        # initalize projecting layers while taking into consideration the effect of residual additions
        self.attn.custom_init(self.weight_init_std)
        self.cx_attn.custom_init(self.weight_init_std)
        self.mlp.custom_init(self.weight_init_std)

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) model that supports conditioning on caption embeddings for text-to-image generation.
    
    Args:
        input_size (int, default: 32): Size of input image (assumed square)
        patch_size (int, default: 2): Size of patches for patch embedding
        in_channels (int, default: 4): Number of input image channels (by default assuming four channel latent space)
        n_embd (int, default: 1152): Embedding Dimension of transformer backbone, i.e., dimension of major transformer layers
        depth (int, default: 28): Number of transformer blocks
        head_size (int, default: 64): Channels per attention head
        n_hidden_base_mult (int, default: 256): Round n_hidden up to nearest multiple of this value in MLP block
        caption_n_embd (int, default: 1024): Number of channels in caption embeddings
        pos_interp_scale (float, default: 1.0): Scale for positional embedding interpolation (1.0 for 256x256, 2.0 for 512x512)
        norm_eps (float, default: 1e-6): Epsilon for layer normalization
        depth_init (bool, default: True): Whether to use depth-dependent initialization in DiT blocks
        qkv_multipliers (List[float], default: [1.0]): Multipliers for QKV projection dimensions in DiT blocks
        ffn_multipliers (List[float], default: [4.0]): Multipliers for FFN hidden dimensions in DiT blocks
        use_patch_mixer (bool, default: True): Whether to use patch mixer layers
        patch_mixer_depth (int, default: 4): Number of patch mixer blocks
        patch_mixer_dim (int, default: 512): Dimension of patch-mixer layers
        patch_mixer_qkv_ratio (float, default: 1.0): Multipliers for QKV projection dimensions in patch-mixer blocks
        patch_mixer_mlp_ratio (float, default: 1.0): Multipliers for FFN hidden dimensions in patch-mixer blocks
        use_bias (bool, default: True): Whether to use bias in linear layers
        num_experts (int, default: 8):  Number of experts if using MoE block
        expert_capacity (int, default: 1): Capacity factor for each expert if using MoE FFN layers
        experts_every_n (int, default: 2): Add MoE FFN layers every n blocks
    """

    def __init__ (
            self,
            input_res :int = 32,
            patch_size:int = 2, #ViT/DiT patch size
            in_channels:int = 4,
            n_embd:int = 1152, # n_embd of transformer backbone
            depth:int = 28, # number of DiT blocks in transformer
            head_size:int = 64, # channels per attention head
            n_hidden_base_mult:int = 256, # round up the n_hidden of backbone to nearest multiple of this
            caption_n_embd:int = 1024,
            pos_interp_scale:float = 1.0,
            norm_eps:float = 1e-6,
            depth_init:bool=True,
            qkv_dim_multipliers:List[float] = [1.0],
            ffn_dim_multipliers:List[float] = [4.0],
            use_patch_mixer: bool = True,
            patch_mixer_depth: int = 4,
            patch_mixer_dim: int=512,
            patch_mixer_qkv_dim_mult:float=1.0,
            patch_mixer_mlp_dim_mult:float=1.0,
            use_bias:bool=True,
            num_experts:int=8,
            expert_capacity:float=1.0, # scale multiplier on tokens_per_expert
            experts_every_n:int = 2
    ):
        super().__init__()
        self.input_res = input_res
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.head_size = head_size
        self.pos_interp_scale = pos_interp_scale
        self.use_patch_mixer = use_patch_mixer

        self.x_embedder = PatchEmbed (input_res, patch_size, in_channels, n_embd, bias=True) # 32x32x4->16x16x1152
        self.t_emebedder = TimeStepEmbedder (n_embd, nn.GELU(approximate="tanh"))

        # get number of patches from x_embedder (32x32->16x16 with patchsize 2)
        num_patches = self.x_embedder.num_patches

        # calculate base size to feed into position embedding function
        self.base_size = input_res // self.patch_size
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, n_embd))

        # non linearity in config if nn.GELU(tanh) by default
        caption_embedder_config = MLPConfig (fan_in=caption_n_embd, fan_h=n_embd, fan_out=n_embd, norm_layer=create_norm("layernorm", n_embd, eps=norm_eps))
        self.caption_embedder = CaptionProjection (caption_embedder_config)

        # refine /pool/aggregate information in caption embedding
        self.caption_embedding_attention = AttentionBlockPromptEmbedding (
            n_embd, head_size, 4.0, n_hidden_base_mult, norm_eps, use_bias=use_bias
        )

        # process the aggregated information in caption embedding
        # non linearity in config is nn.GELU(tanh) by default
        process_pooled_caption_mlp_config = MLPConfig (n_embd, n_embd, n_embd, norm_layer=create_norm("layernorm", n_embd, eps=norm_eps))
        self.process_pooled_caption_embedding = MLP (process_pooled_caption_mlp_config)

        if self.use_patch_mixer:
            # [0 dense, 1 sparse, 2 dense, 3 sparse]
            expert_blocks_idx = [
                i for i in range(patch_mixer_depth) if i % experts_every_n != 0
            ]

            # construct bool flags for use moe using the list above
            is_moe_block = [
                True if i in expert_blocks_idx else False for i in range (patch_mixer_depth)
            ]

            # Patch Mixer
            self.patch_mixer = nn.ModuleList([
                DiTBlock (
                    n_embd=patch_mixer_dim,
                    head_size=head_size,
                    mlp_n_hidden_mult=patch_mixer_mlp_dim_mult,
                    qkv_n_hidden_mult=patch_mixer_qkv_dim_mult,
                    hidden_base_mult=n_hidden_base_mult, # for MLP, attention layers always have 2*head_size base mult
                    modulated_sigma_t_embd=n_embd,
                    norm_eps=norm_eps,
                    depth_init=False,
                    block_index=0, # doesnt affect weight init (intended to count prior inclusive residual additions), since depth_init is false for patch_mixer
                    num_blocks_for_weight_init=depth,
                    scale_cx_attn_n_hidden=False, # dont scale cx_attn_qkv channels for patch mixer
                    use_bias=use_bias,
                    is_moe=is_moe_block[i],
                    num_experts=num_experts,
                    expert_capacity=expert_capacity # float
                ) for i in range(patch_mixer_depth)
            ])

            # projection layers
            if patch_mixer_dim != n_embd:
                # project image to patch_mixer n_embd for processing
                self.project_image_to_patch_mixer_embd = nn.Sequential (
                    create_norm("layernorm", n_embd, eps=norm_eps),
                    nn.Linear(n_embd, patch_mixer_dim, bias=use_bias)
                )

                self.project_patch_mixer_to_backbone_embd = nn.Sequential (
                    create_norm("layernorm", patch_mixer_dim, eps=norm_eps),
                    nn.Linear(patch_mixer_dim, n_embd, bias=use_bias)
                )

                self.project_caption_to_patch_mixer_embd = nn.Sequential (
                    create_norm("layernorm", n_embd, eps=norm_eps),
                    nn.Linear (n_embd, patch_mixer_dim, bias=use_bias)
                )

            else:
                self.project_image_to_patch_mixer_embd = nn.Identity()
                self.project_patch_mixer_to_backbone_embd = nn.Identity()
                self.project_caption_to_patch_mixer_embd = nn.Identity()
            
            # make sure that number of ffn_multipliers is equal to number of qkv attention dim multipliers
            # implicit condition to make sure that number of ffn layers is equal to number of attention layers
            # these lists are used for layerwise scaling
            assert (len(ffn_dim_multipliers) == len(qkv_dim_multipliers))
            # if we have one layerwise scaling multiplier for each layer
            if len(ffn_dim_multipliers) == depth:
                # we proceed with the lists as is
                qkv_fan_h_mults = qkv_dim_multipliers
                mlp_fan_h_mults = ffn_dim_multipliers
            else:
                # each layer doesnt have its own multiplier
                # distribute multipliers equally across splits
                # if we have just 2 multipliers, there will be just 2 splits
                num_splits = len (ffn_dim_multipliers)

                assert (depth % num_splits == 0), f"Depth(number of blocks):{depth} should be divisible by number of splits:{num_splits}"

                depth_per_split = depth // num_splits

                qkv_fan_h_mults = list(np.array([[m]*depth_per_split for m in qkv_dim_multipliers]).reshape(-1))
                mlp_fan_h_mults = list(np.array([[m]*depth_per_split for m in ffn_dim_multipliers]).reshape(-1))

            # now qkv/mlp_fan_h_mults have depth multipliers for layerwise scaling
            
            # don't use MoE in last block
            # list looks like (1, 3, 5.....25)
            expert_blocks_idx = [i for i in range(depth-1) if i % experts_every_n != 0]
            # list looks like (Dense, Sparse, .... Dense, Dense)
            is_moe_block = [ (True if i in expert_blocks_idx else False) for i in range(depth)]

            self.backbone = nn.ModuleList([
                DiTBlock (
                   n_embd=n_embd,
                   head_size=head_size,
                   mlp_n_hidden_mult=mlp_fan_h_mults[i],
                   qkv_n_hidden_mult= qkv_fan_h_mults[i],
                   hidden_base_mult=n_hidden_base_mult, # used only for MLP FFN, attention layers always use basemult of 2xHead_Size
                   modulated_sigma_t_embd=caption_n_embd,
                   norm_eps=norm_eps,
                   depth_init=depth_init,
                   block_index=i,
                   num_blocks_for_weight_init=depth,
                   scale_cx_attn_n_hidden=False,
                   use_bias=use_bias,
                   is_moe=is_moe_block[i],
                   num_experts=num_experts,
                   expert_capacity=expert_capacity
                )for i in range(depth)
            ])

            self.register_buffer ("mask_token", torch.zeros(1, 1, patch_size**2 *self.out_channels))

            self.final_layer = T2IFinalLayer (
                n_embd,
                n_embd,
                patch_size,
                self.out_channels,
                nn.GELU(approximate='tanh'),
                create_norm("layernorm", n_embd, eps=norm_eps)
            )
            self.initialize_weights()


    def forward_without_cfg (
            self,
            x: torch.Tensor,
            sigma_t: torch.Tensor,
            c : torch.Tensor, # caption
            mask_ratio : float = 0,
            **kwargs
    )->dict:
        """Forward pass without classifier free guidance.
        
        Args:
        x: Input tensor of shape (B, C, H, W)
        t: sigma_t noise/time tensor of shape (B,)
        y: Caption embedding tensor of shape (B, 1, seq_len, caption_embd) (B, 1, 77, 1024)
        mask_ratio: Ratio of patches to mask during training (between 0 and 1)

        Returns:
        dict: Dictionary containing:
            - 'sample': Output tensor of shape (batch_size, out_channels, height, width)
            - 'mask': Optional binary mask tensor, if masking was applied else None
        """
        B, C, H, W = x.shape
        self.h = H // self.patch_size # 32/2 = 16
        self.W = W // self.patch_size # 32/2 = 16

        # self.pos_embed is initialized during weight_init 
        x = self.x_embedder(x) + self.pos_embed  # (B, 256, 1152) + (1, 256, 1152) (pos_embed is a register_buffer) 
        
        # sigma_t noise tensor expanded across entire batch -> (B)
        # and then embed using sincos embedding processed by linear-acitvation-linear to yield
        # B, C embedding
        sigma_t = self.t_emebedder(sigma_t.expand(x.shape[0])) #(B, C)

        c = self.caption_embedder (c) # (B, 1, L, 1024) -> (B, 1, L, n_embd=1152), suprious dimension because thats how we wrote the forward pass of openclip encoder

        # Aggregate/refine information in caption embedding using Multi Headed Attention
        c = self.caption_embedding_attention(c.squeeze(dim=1)).unsqueeze(dim=1) # (B, 1, L, n_embd=1152) -> (B, 1, L, n_embd=1152)
        
        # Inject caption information into time/noise
        # pool information of all tokens in caption
        pooled_caption = c.mean(dim=-2).squeeze(dim=1) # (B, 1, L, C) -> (B, 1, C)
        # refine information with mlp
        pooled_caption = self.process_pooled_caption_embedding (pooled_caption) # (B, 1, C)
        
        # modulated noise level and caption information
        modulated_sigma_t = sigma_t + pooled_caption

        
        if self.use_patch_mixer:
            x = self.project_image_to_patch_mixer_embd(x)
            c_mixer = self.project_caption_to_patch_mixer_embd(c)
            # no need to project modulated_sigma_t to patch_mixer_embd since we just get affine parameters of appropriate embd 
            # from it using a adaln layer.
            for block in self.patch_mixer:
                x = block (x, c_mixer, modulated_sigma_t) # (B, T, patch_mixer_embd)
        
        mask = None
        if mask_ratio > 0:
            mask_dict = get_mask(
                x.shape[0], x.shape[1], device=x.device
            )
        idx_keep = mask_dict['idx_keep'] # (B, 0.25T)
        idx_restore = mask_dict['idx_restore'] # (B, T)
        mask = mask_dict['mask'] # (B, T)

        x = mask_out_token(x, idx_keep) # (B, 0.25T, C) if masking ratio is 0.75

        if self.use_patch_mixer:
            # project back to backbone embd from patch_mixer_dim (or patch_mixer_embd)
            # after masking out, to save compute
            x = self.project_patch_mixer_to_backbone_embd (x)
        
        for block in self.backbone:
            x = block(x=x, c=c, t=modulated_sigma_t) # (B, 0.25T, C)
        
        # unmask fill in zero stubs B T,C

        # tokenToImageLayer
        x = self.final_layer (x) # (B, 0.25T, patch_size**2 C) we dont project back to vae latent dim yet

        if mask_ratio > 0:
            x = fill_out_masked_tokens (x, self.mask_token, idx_restore=idx_restore) # fill out stubs with 1, 1, patchsize**2C
        
        x = self.unpatchify(x) # (B, out_channels, H, W)

        return {'sample':x, 'mask':mask} # x (B, 4, H, W) mask: (B, T)
        

        



        