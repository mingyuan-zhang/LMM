_base_ = ['lmm.py', 'motionverse.py']

model = dict(
    model=dict(
        latent_part_dim=64,
        num_layers=4,
        num_cond_layers=2,
        dropout=0.1,
        ca_block_cfg=dict(
            num_experts=16,
            topk=4
        ),
        guidance_cfg=dict(
            humanml3d_t2m=dict(type='linear', scale=10.5),
        ),
    ),
    diffusion_test_dict=dict(
        humanml3d_t2m='15,15,8,6,6',
    ),
)