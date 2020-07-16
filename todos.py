"""To-Do's"""
# Algorithm based
# • Todo: Try gradient clipping (norm?, value?) -> DONE
# • Todo: Implement retrace without division -> DONE
# • Todo: target_Q_next[:, -1] has always the same value. Why? -> DONE
# • Todo: Try out orthogonal weight init and bias zero -> DONE
# • Todo: Implement torch.rsample. Enables reparameterization trick -> DONE
# • Todo: Make standard deviation state independent -> Done
# • Todo: Try to out entropy regularization
# • Todo: Try out reward clipping (at least for pendulum)
# • Todo: Implement action and observation normalization

# Environment based
# • Todo: Use mocap pos as endeffector pos in order to calculate the reward instead of using the endeffector
#  position of the robot
# • Todo: Try out increasing dt such that sampling an episode requires less time
# • Todo: Initialize each worker with a different seed
# • Todo: Why does 'render' not work with argparser
# • Todo: Include

"""Ideas"""
# • Todo: Try out prioritized experience replay
# • Todo: Try out trust region updates
# • Todo: Try out learning rate decay
# • Todo: Try using a "burn in" phase where we train the critic a bit on policy to prevent the retrace weights from
#  going to zero
# • Todo: Normalize max cumm reward between -1 and 1


"""Notes and warnings"""
# • when using trjectories with different lengths be careful when reversing the trajectory for computing the
#   retrace loss because the trailing zeros should NOT be reversed!

# • when updating the target networks in the mp setting the gradients will again be set to true
