python3 ../sp_carl/main.py \
--num_worker=16 \
--batch_size=64 \
--update_targnets_every=50 \
--learning_steps=500 \
--smoothing_coefficient=1 \
--reward_scale=30 \
--actor_lr=3e-5 \
--critic_lr=2e-4 \
--init_std=0.2 \
--global_gradient_norm=0.5 \
--entropy_reg=0 \
--replay_buffer_size=50000 \
--num_trajectories=50  \