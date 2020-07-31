python3 -O ../sp_carl/main.py \
--num_worker=6 \
--batch_size=32 \
--update_targnets_every=30 \
--learning_steps=180 \
--smoothing_coefficient=1 \
--reward_scale=1 \
--actor_lr=2e-4 \
--critic_lr=2e-4 \
--init_std=0.5 \
--global_gradient_norm=0.5 \
--entropy_reg=0 \
--replay_buffer_size=1000 \
--num_trajectories=20  \
--num_evals=1
