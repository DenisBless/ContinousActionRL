import argparse


parser = argparse.ArgumentParser(description='algorithm arguments')

# Algorithm parameter
parser.add_argument('--batch_size', type=int, default=32,
                    help='Size of the batches used for training the actor and the critic.')
parser.add_argument('--update_targnets_every', type=int, default=50,
                    help='Number of learning steps before the target networks are updated.')
parser.add_argument('--learning_steps', type=int, default=1000,
                    help='Total number of learning timesteps before sampling trajectories.')
parser.add_argument('--num_runs', type=int, default=5000,
                    help='Number of learning iterations.')
parser.add_argument('--out_layer', type=str, default='linear',
                    help='Output layer of the actor network. Choose between <linear>, <tanh>.')
parser.add_argument('--actor_lr', type=float, default=2e-4,
                    help='Learning rate for the actor network.')
parser.add_argument('--critic_lr', type=float, default=2e-4,
                    help='Learning rate for the critic network.')
parser.add_argument('--layer_norm', type=bool, default=False,
                    help='Includes a layer norm between FC layer in the actor and critic network.')
parser.add_argument('--global_gradient_norm', type=float, default=0.5,
                    help='Enables gradient clipping with a specified global parameter L2 norm')
parser.add_argument('--num_expectation_samples', type=int, default=1,
                    help='Number of action samples for approximating the value function from the Q function.')
parser.add_argument('--entropy_reg', type=float, default=0,
                    help='Scaling of entropy term in the actor loss function')
parser.add_argument('--trust_region_coeff', type=float, default=0,
                    help='Scaling of the KL-div. between the old action distribution and the current in actor loss.')
parser.add_argument('--action_mean_scale', type=float, default=1,
                    help='Scales the output of the actor net to [-action_mean_scale, action_mean_scale].')
parser.add_argument('--action_std_low', type=float, default=5e-1,
                    help='Lower bound on the standard deviation of the actions.')
parser.add_argument('--action_std_high', type=float, default=1.,
                    help='Upper bound on the standard deviation of the actions.')
parser.add_argument('--action_bound', type=float, default=1.,
                    help='Clips the action in the range [-action_bound, action_bound].')
parser.add_argument('--replay_buffer_size', type=int, default=5000,
                    help='Size of the replay buffer.')
parser.add_argument('--log_interval', type=int, default=10,
                    help='Interval of the logger to collect and print data to tensorboard')
parser.add_argument('--save_model_every', type=int, default=10,
                    help='Interval of learning iterations after which the model state dict is saved.')
parser.add_argument('--model_save_path', type=str,
                    default=str(pathlib.Path(__file__).resolve().parents[1]) + "/models/",
                    help='Directory to the saved models.')

# Environment parameter
parser.add_argument('--episode_length', type=int, default=100,
                    help='Length of a episode.')
parser.add_argument('--num_eval_trajectories', type=int, default=1,
                    help='Number of trajectories used for evaluating the policy.')
parser.add_argument('--num_trajectories', type=int, default=50,
                    help='Number of trajectories sampled before entering the learning phase.')
parser.add_argument('--dt', type=float, default=1e-3,
                    help='Time between steps in the mujoco pyhsics simulation (in seconds).')
parser.add_argument('--percentage', type=float, default=2e-2,
                    help='Percent scaling of the action which is in [-1, 1] (in meters)')
parser.add_argument('--control_timesteps', type=int, default=100,
                    help='Number of steps in the simulation between before the next action is executed.')
parser.add_argument('--action_smoothing', type=bool, default=True,
                    help='Uses number <control_timsteps> to smooth between two action from the agent.')
