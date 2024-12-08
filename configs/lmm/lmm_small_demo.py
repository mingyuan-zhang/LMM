_base_ = ['lmm.py']

model = dict(
    model=dict(
        latent_part_dim=64,
        num_layers=8,
        num_cond_layers=2,
        dropout=0.1,
        ca_block_cfg=dict(
            num_experts=16,
            topk=4
        )
    )
)