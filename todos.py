"""To-Do's"""
# • Todo: Try gradient clipping (norm?, value?) -> DONE
# • Todo: Implement retrace without division -> DONE
# • Todo: target_Q_next[:, -1] has always the same value. Why?
# • Todo: Try hardtanh instead of tanh
# • Todo: Use mocap pos as endeffector pos in order to calculate the reward instead of using the endeffector
#  position of the robot
# • Todo: Try using a "burn in" phase where we train the critic a bit on policy to prevent the retrace weights from
#  going to zero
# • Todo: Implement torch.rsample. Enables reparameterization trick
# • Todo: Try out prioritized experience replay
# • Todo: Try out trust region updates
# • Todo: Try out learning rate decay
# • Todo: Try out orthogonal weight init and bias zero
# • Todo: Try to out entropy regularization
# • Todo: Try out increasing dt such that sampling an episode requires less time
# • Todo: Try out truncating the gaussian distribution directly at the output of the neural network
# • Todo: Make standard deviation state independent
# • Todo: Try out reward clippint (at least for pendulum)
# • Todo:



"""Notes and warnings"""
# • when using trjectories with different lengths be careful when reversing the trajectory for computing the
#   retrace loss because the trailing zeros should NOT be reversed!
# • target_Q_next[:, -1] has always the same value???
# •

"""
Working params:

    # PARAMETER
    NUM_OBSERVATIONS = env.observation_dim
    NUM_ACTIONS = env.agent.action_dimension
    NUM_EVAL_TRAJECTORIES = 1
    NUM_TRAJECTORIES = 150
    BATCH_SIZE = 32
    NUM_BUFFER_FRONTLOAD = 0
    # NUM_TRAJECTORIES = 16
    # BATCH_SIZE = 16
    # NUM_BUFFER_FRONTLOAD = 0
    UPDATE_TARGNETS_EVERY = 50
    NUM_TRAINING_ITERATIONS = 2000
    TOTAL_TIMESTEPS = 1000
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 1e-4
    GRADIENT_CLIPPING_VALUE = 0.5
    NUM_EXPECTATION_SAMPLES = 1
    ENTROPY_REGULARIZATION_ON = False
    ENTROPY_REGULARIZATION = 0
    TRUST_REGION_COEFF = 0
    ACTION_STD_LOW = 5e-1
    ACTION_STD_HIGH = 1
    ACTION_MEAN_SCALE = 1
    ACTION_BOUNDS = (-1, 1)
    REPLAY_BUFFER_SIZE = 5000
    LOG_EVERY = 10
    SAVE_MODEL_EVERY = 10
    MODEL_SAVE_PATH = str(pathlib.Path(__file__).resolve().parents[1]) + "/models/"
    
    
    Learning rates smaller than 5e-5 have shown to be too small.
"""