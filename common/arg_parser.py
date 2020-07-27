from argparse import ArgumentParser


class ArgParser(ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(description='arg_parser')

        # Algorithm parameter
        # parser.add_argument('--num_worker', type=int, default=os.cpu_count(),
        self.add_argument('--num_worker', type=int, default=2,
                          help='Number of workers training the agent in parallel.')
        self.add_argument('--num_trajectories', type=int, default=10,
                          help='Number of trajectories sampled before entering the learning phase.')
        self.add_argument('--learning_steps', type=int, default=10,
                          help='Total number of learning timesteps before sampling trajectories.')
        self.add_argument('--batch_size', type=int, default=4,
                          help='Number of trajectories in a batch.')
        self.add_argument('--num_grads', type=int, default=1,
                          help='Number of gradients collected before updating the networks.')

        self.add_argument('--update_targnets_every', type=int, default=1,
                          help='Number of learning steps before the target networks are updated.')

        self.add_argument('--num_runs', type=int, default=5000,
                          help='Number of learning iterations.')
        self.add_argument('--actor_lr', type=float, default=1e-4,
                          help='Learning rate for the actor network.')
        self.add_argument('--critic_lr', type=float, default=1e-4,
                          help='Learning rate for the critic network.')
        self.add_argument('--init_std', type=float, default=1,
                          help='Initial standard deviation of the actor.')
        self.add_argument('--smoothing_coefficient', type=float, default=0.001,
                          help='Decides how the target networks are updated. One corresponds to a hard updates, whereas'
                               ' values between zero and one result in exponential moving average updates.')
        self.add_argument('--reward_scale', type=int, default=1,
                          help='Scales the reward.')

        self.add_argument('--global_gradient_norm', type=float, default=0.5,
                          help='Enables gradient clipping with a specified global parameter L2 norm')
        self.add_argument('--num_expectation_samples', type=int, default=1,
                          help='Number of action samples for approximating the value function from the Q function.')
        self.add_argument('--entropy_reg', type=float, default=0.01,
                          help='Scaling of entropy term in the actor loss function')
        self.add_argument('--trust_region_coeff', type=float, default=0,
                          help='Scaling of the KL-div. between the old action distribution and the current in actor loss.')
        self.add_argument('--action_mean_scale', type=float, default=2,
                          help='Scales the output of the actor net to [-action_mean_scale, action_mean_scale].')
        self.add_argument('--action_bound', type=float, default=2.,
                          help='Clips the action in the range [-action_bound, action_bound].')
        self.add_argument('--replay_buffer_size', type=int, default=1000,
                          help='Size of the replay buffer.')
        self.add_argument('--logging', type=bool, default=True,
                          help='Whether to log data or not.')

        # Environment parameter
        # parser.add_argument('--episode_length', type=int, default=200,
        #                     help='Length of a episode.')
        self.add_argument('--num_evals', type=int, default=1,
                          help='Number of trajectories used for evaluating the policy.')

        self.add_argument('--dt', type=float, default=1e-3,
                          help='Time between steps in the mujoco pyhsics simulation (in seconds).')
        self.add_argument('--percentage', type=float, default=2e-2,
                          help='Percent scaling of the action which is in [-1, 1] (in meters)')
        self.add_argument('--control_timesteps', type=int, default=100,
                          help='Number of steps in the simulation between before the next action is executed.')
        self.add_argument('--action_smoothing', type=bool, default=True,
                          help='Uses number <control_timsteps> to smooth between two action from the agent.')

        self.add_argument('--render', type=bool, default=False,
                          help='If true, the environment is rendered.')

        self.add_argument('--log_interval', type=int, default=1,
                          help='Interval of the logger writing data to the tensorboard.')

    def hparam_dict(self):
        return {'update_targets': ...,
                'learning_steps': ...,
                'actor_lr': ...,
                'critic_lr': ...,
                'entropy_reg': ...,
                'init_std': ...,
                'global_gradient_norm': ...,
                'replay_buffer_size': ...,
                'num_trajectories': ...
                }

