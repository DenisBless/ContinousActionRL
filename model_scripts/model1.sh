python3 ../mp_carl/main.py \
--num_worker=6 \
--num_grads=1 \
--update_targnets_every=200     \
--learning_steps=2000 \
--actor_lr=2e-4 \
--critic_lr=2e-4 \
--global_gradient_norm=1 \
--entropy_reg=0 \
--replay_buffer_size=300 \
