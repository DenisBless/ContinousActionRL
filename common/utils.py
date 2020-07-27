def get_hparam_dict(argp):
    return {'update_targets': argp.update_targnets_every,
            'learning_steps': argp.learning_steps,
            'actor_lr': argp.actor_lr,
            'critic_lr': argp.critic_lr,
            'entropy_reg': argp.entropy_reg,
            'init_std': argp.init_std,
            'global_gradient_norm': argp.global_gradient_norm,
            'replay_buffer_size': argp.replay_buffer_size,
            'num_trajectories': argp.num_trajectories
            }

def get_metric_dict(argp):
    return {'mean reward': ...,
            }

t2n = lambda t: t.detach().numpy()