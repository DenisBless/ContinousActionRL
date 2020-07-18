python3 ../mp_carl/main.py \
--num_worker=2 \
--num_grads=4 \
--update_targnets_every=1000 \
--learning_steps=3000 \
--actor_lr=1e-4 \
--critic_lr=1e-3 \
--init_std=-0.2 \
--global_gradient_norm=0.5 \
--entropy_reg=0 \
--replay_buffer_size=10000 \
--num_trajectories=10  \
