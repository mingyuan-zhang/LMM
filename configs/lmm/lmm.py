dataset_names = [
    'all', 
    'amass_mocap', 'motionx_mocap', 'humanact12_mocap', 'uestc_mocap', 'ntu_mocap', 'aist_mocap',
    'beat_mocap', 'tedg_mocap', 'tedex_mocap', 's2g3d_mocap', 'h36m_mocap', 'mpi_mocap',
    
    'humanml3d_t2m', 'kitml_t2m', 'babel_t2m', 'motionx_t2m',
    'humanact12_t2m', 'uestc_t2m', 'ntu_t2m',   
    
    'aist_m2d',
    'beat_s2g', 'tedg_s2g', 'tedex_s2g', 's2g3d_s2g',
    
    'h36m_v2m', 'mpi_v2m'
]
num_datasets = len(dataset_names)
# model settings
model = dict(
    type='UnifiedMotionDiffusion',
    model=dict(
        type='LargeMotionModel',
        input_feats=669,
        max_seq_len=200,
        num_parts=10,
        latent_part_dim=64,
        time_embed_dim=2048,
        dataset_names=dataset_names,
        num_layers=4,
        num_cond_layers=2,
        num_datasets=num_datasets,
        dropout=0,
        ca_block_cfg=dict(
            type='ArtAttention',
            num_experts=16,
            topk=4,
            gate_type='cosine_top',
            gate_noise=1.0,
            num_datasets=num_datasets,
            has_text=True,
            has_music=True,
            has_speech=True,
            has_video=True
        ),
        text_input_dim=1024,
        music_input_dim=768,
        speech_input_dim=768,
        video_input_dim=1024,
        guidance_cfg=dict(
            all=dict(type='linear', scale=5.5),
        ),
        moe_route_loss_weight=10.0,
        template_kl_loss_weight=0.0001,
        use_pos_embedding=False,
        cond_drop_rate=0.1
    ),
    loss_recon=dict(
        type='KinematicLoss', loss_type='mse', loss_weight=[20], reduction='none'),
    train_repeat=1,
    diffusion_train=dict(
        beta_scheduler='linear',
        diffusion_steps=1000,
        model_mean_type='start_x',
        model_var_type='fixed_large',
    ),
    diffusion_test_dict=dict(
        base=dict(
            beta_scheduler='linear',
            diffusion_steps=1000,
            model_mean_type='start_x',
            model_var_type='fixed_large',
        ),
        all='15,15,8,6,6'
    ),
    inference_type='ddim',
    loss_reduction='batch',
    loss_weight='data/motionverse/statistics/loss_weight.npy'
)
