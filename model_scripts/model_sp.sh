python3 ../sp_carl/main.py \
--num_worker=16 \
--batch_size=64 \
--update_targnets_every=1 \
--smoothing_coefficient=0.005 \
--learning_steps=500 \
--actor_lr=2e-5 \
--critic_lr=2e-4 \
--init_std=1e-3 \
--global_gradient_norm=-1 \
--entropy_reg=1e-2 \
--replay_buffer_size=20000 \
--num_trajectories=50  \
