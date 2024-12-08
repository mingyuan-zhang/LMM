_base_ = ['lmm.py']

model = dict(
    model=dict(
        latent_part_dim=128,
        num_layers=20,
        num_cond_layers=4,
        dropout=0.1,
        ca_block_cfg=dict(
            num_experts=32,
            topk=8
        )
    )
)
