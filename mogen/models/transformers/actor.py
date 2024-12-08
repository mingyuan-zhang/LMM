import torch
from mmcv.runner import BaseModule
from torch import nn
from typing import Optional

from mogen.models.utils.mlp import build_MLP
from mogen.models.utils.position_encoding import (LearnedPositionalEncoding,
                                                  SinusoidalPositionalEncoding)

from ..builder import SUBMODULES


@SUBMODULES.register_module()
class ACTOREncoder(BaseModule):
    """ACTOR Encoder module for motion data.

    Args:
        max_seq_len (Optional[int]): Maximum sequence length for positional encoding.
        njoints (Optional[int]): Number of joints for motion input. Defaults to None.
        nfeats (Optional[int]): Number of features for each joint. Defaults to None.
        input_feats (Optional[int]): Total input feature dimension. Defaults to None.
        latent_dim (Optional[int]): Latent feature dimension.
        condition_dim (Optional[int]): Dimension of condition features. Defaults to None.
        num_heads (Optional[int]): Number of heads in the Transformer encoder.
        ff_size (Optional[int]): Feedforward network size in the Transformer.
        num_layers (Optional[int]): Number of layers in the Transformer encoder.
        activation (Optional[str]): Activation function for the Transformer.
        dropout (Optional[float]): Dropout probability.
        use_condition (Optional[bool]): Whether to use conditioning inputs.
        num_class (Optional[int]): Number of classes for conditional encoding. Defaults to None.
        use_final_proj (Optional[bool]): Whether to apply a final projection layer.
        output_var (Optional[bool]): Whether to output a variance along with mean.
        pos_embedding (Optional[str]): Type of positional encoding ('sinusoidal' or 'learned').
        init_cfg (Optional[dict]): Initialization configuration.
    """

    def __init__(self,
                 max_seq_len: Optional[int] = 16,
                 njoints: Optional[int] = None,
                 nfeats: Optional[int] = None,
                 input_feats: Optional[int] = None,
                 latent_dim: Optional[int] = 256,
                 condition_dim: Optional[int] = None,
                 num_heads: Optional[int] = 4,
                 ff_size: Optional[int] = 1024,
                 num_layers: Optional[int] = 8,
                 activation: Optional[str] = 'gelu',
                 dropout: Optional[float] = 0.1,
                 use_condition: Optional[bool] = False,
                 num_class: Optional[int] = None,
                 use_final_proj: Optional[bool] = False,
                 output_var: Optional[bool] = False,
                 pos_embedding: Optional[str] = 'sinusoidal',
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        
        # If input_feats is not provided, compute it from njoints and nfeats
        self.njoints = njoints
        self.nfeats = nfeats
        if input_feats is None:
            assert self.njoints is not None and self.nfeats is not None
            self.input_feats = njoints * nfeats
        else:
            self.input_feats = input_feats
        
        # Initialize parameters
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.use_condition = use_condition
        self.num_class = num_class
        self.use_final_proj = use_final_proj
        self.output_var = output_var

        # Linear embedding layer for skeleton input features
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        # If using conditional inputs, set up layers for conditional processing
        if self.use_condition:
            if num_class is None:
                self.mu_layer = build_MLP(self.condition_dim, self.latent_dim)
                if self.output_var:
                    self.sigma_layer = build_MLP(self.condition_dim, self.latent_dim)
            else:
                self.mu_layer = nn.Parameter(torch.randn(num_class, self.latent_dim))
                if self.output_var:
                    self.sigma_layer = nn.Parameter(torch.randn(num_class, self.latent_dim))
        else:
            if self.output_var:
                self.query = nn.Parameter(torch.randn(2, self.latent_dim))  # Query for mu and sigma
            else:
                self.query = nn.Parameter(torch.randn(1, self.latent_dim))  # Query for mu only

        # Positional encoding setup
        if pos_embedding == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(latent_dim, dropout)
        else:
            self.pos_encoder = LearnedPositionalEncoding(latent_dim, dropout, max_len=max_seq_len + 2)

        # Transformer encoder layers
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)

    def forward(self, motion: torch.Tensor, motion_mask: Optional[torch.Tensor] = None, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for ACTOR Encoder.

        Args:
            motion (torch.Tensor): Input motion data of shape (B, T, njoints, nfeats).
            motion_mask (Optional[torch.Tensor]): Mask for valid motion data. Defaults to None.
            condition (Optional[torch.Tensor]): Conditional input. Defaults to None.

        Returns:
            torch.Tensor: Encoded latent representation.
        """
        # Get batch size (B) and sequence length (T)
        B, T = motion.shape[:2]
        
        # Flatten motion input into (B, T, input_feats)
        motion = motion.view(B, T, -1)
        
        # Embed the motion input features into latent space
        feature = self.skelEmbedding(motion)
        
        # Handle conditional inputs, concatenating condition queries
        if self.use_condition:
            if self.output_var:
                if self.num_class is None:
                    sigma_query = self.sigma_layer(condition)
                else:
                    sigma_query = self.sigma_layer[condition.long()]
                sigma_query = sigma_query.view(B, 1, -1)
                feature = torch.cat((sigma_query, feature), dim=1)

            if self.num_class is None:
                mu_query = self.mu_layer(condition).view(B, 1, -1)
            else:
                mu_query = self.mu_layer[condition.long()].view(B, 1, -1)
            feature = torch.cat((mu_query, feature), dim=1)
        else:
            query = self.query.view(1, -1, self.latent_dim).repeat(B, 1, 1)
            feature = torch.cat((query, feature), dim=1)

        # If outputting variance, adjust the mask accordingly
        if self.output_var:
            motion_mask = torch.cat((torch.zeros(B, 2).to(motion.device), 1 - motion_mask), dim=1).bool()
        else:
            motion_mask = torch.cat((torch.zeros(B, 1).to(motion.device), 1 - motion_mask), dim=1).bool()

        # Positional encoding and transformer encoder processing
        feature = feature.permute(1, 0, 2).contiguous()  # Permute for transformer
        feature = self.pos_encoder(feature)
        feature = self.seqTransEncoder(feature, src_key_padding_mask=motion_mask)

        # Apply final projection if required
        if self.use_final_proj:
            mu = self.final_mu(feature[0])
            if self.output_var:
                sigma = self.final_sigma(feature[1])
                return mu, sigma
            return mu
        else:
            if self.output_var:
                return feature[0], feature[1]
            else:
                return feature[0]


@SUBMODULES.register_module()
class ACTORDecoder(BaseModule):
    """ACTOR Decoder module for motion generation.

    Args:
        max_seq_len (Optional[int]): Maximum sequence length.
        njoints (Optional[int]): Number of joints for motion input. Defaults to None.
        nfeats (Optional[int]): Number of features for each joint. Defaults to None.
        input_feats (Optional[int]): Total input feature dimension. Defaults to None.
        input_dim (Optional[int]): Input feature dimension.
        latent_dim (Optional[int]): Latent feature dimension.
        condition_dim (Optional[int]): Dimension of condition features. Defaults to None.
        num_heads (Optional[int]): Number of heads in the Transformer decoder.
        ff_size (Optional[int]): Feedforward network size in the Transformer.
        num_layers (Optional[int]): Number of layers in the Transformer decoder.
        activation (Optional[str]): Activation function for the Transformer.
        dropout (Optional[float]): Dropout probability.
        use_condition (Optional[bool]): Whether to use conditioning inputs.
        num_class (Optional[int]): Number of classes for conditional encoding. Defaults to None.
        pos_embedding (Optional[str]): Type of positional encoding ('sinusoidal' or 'learned').
        init_cfg (Optional[dict]): Initialization configuration.
    """

    def __init__(self,
                 max_seq_len: Optional[int] = 16,
                 njoints: Optional[int] = None,
                 nfeats: Optional[int] = None,
                 input_feats: Optional[int] = None,
                 input_dim: Optional[int] = 256,
                 latent_dim: Optional[int] = 256,
                 condition_dim: Optional[int] = None,
                 num_heads: Optional[int] = 4,
                 ff_size: Optional[int] = 1024,
                 num_layers: Optional[int] = 8,
                 activation: Optional[str] = 'gelu',
                 dropout: Optional[float] = 0.1,
                 use_condition: Optional[bool] = False,
                 num_class: Optional[int] = None,
                 pos_embedding: Optional[str] = 'sinusoidal',
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        
        # If input_dim is different from latent_dim, we need a linear transformation
        if input_dim != latent_dim:
            self.linear = nn.Linear(input_dim, latent_dim)
        else:
            self.linear = nn.Identity()

        # Setting parameters for the number of joints, features, and sequence length
        self.njoints = njoints
        self.nfeats = nfeats
        if input_feats is None:
            assert self.njoints is not None and self.nfeats is not None
            self.input_feats = njoints * nfeats
        else:
            self.input_feats = input_feats

        # Model configuration parameters
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.use_condition = use_condition
        self.num_class = num_class

        # If using condition input, initialize condition bias
        if self.use_condition:
            if num_class is None:
                self.condition_bias = build_MLP(condition_dim, latent_dim)
            else:
                self.condition_bias = nn.Parameter(torch.randn(num_class, latent_dim))

        # Initialize positional encoding method
        if pos_embedding == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(latent_dim, dropout)
        else:
            self.pos_encoder = LearnedPositionalEncoding(latent_dim, dropout, max_len=max_seq_len)

        # Transformer Decoder layer definition
        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)

        # Define the transformer decoder with multiple layers
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=num_layers)

        # Final output layer to produce the pose from latent features
        self.final = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, input: torch.Tensor, motion_mask: Optional[torch.Tensor] = None, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for ACTOR Decoder.

        Args:
            input (torch.Tensor): Input tensor from the encoder, shape (B, latent_dim).
            motion_mask (Optional[torch.Tensor]): Mask for motion data, shape (B, T). Defaults to None.
            condition (Optional[torch.Tensor]): Conditional input, shape (B, condition_dim). Defaults to None.

        Returns:
            torch.Tensor: Output pose predictions of shape (B, T, njoints * nfeats).
        """
        B = input.shape[0]  # Get batch size
        T = self.max_seq_len  # Max sequence length for decoding

        # Transform input to latent space if needed
        input = self.linear(input)

        # Add condition bias to input if using conditional inputs
        if self.use_condition:
            if self.num_class is None:
                condition = self.condition_bias(condition)
            else:
                condition = self.condition_bias[condition.long()].squeeze(1)
            input = input + condition

        # Positional encoding for query
        query = self.pos_encoder.pe[:T, :].view(T, 1, -1).repeat(1, B, 1)

        # Prepare input and pass through Transformer Decoder
        input = input.view(1, B, -1)  # Prepare input shape for decoder
        feature = self.seqTransDecoder(
            tgt=query, memory=input, tgt_key_padding_mask=(1 - motion_mask).bool())

        # Final layer to produce pose from latent features
        pose = self.final(feature).permute(1, 0, 2).contiguous()

        return pose

