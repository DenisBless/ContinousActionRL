"""To-Do's"""
# • Todo: Try gradient clipping (norm?, value?) -> DONE
# • Todo: Implement retrace without division -> DONE
# • Todo: Try different weight initialization
# • Todo: target_Q_next[:, -1] has always the same value. Why?
# • Todo: Try hardtanh instead of tanh
# • Todo: Do we sample trajectories with target actor or with normal actor (it matters because the it influences the
# retrace weights)
# • Todo: Use mocap pos as endeffector pos in order to calculate the reward instead of using the endeffector
#  position of the robot
# • Todo: Try using a "burn in" phase where we train the critic a bit on policy to prevent the retrace weights from
#  going to zero
# • Todo:
# • Todo:
# • Todo:
# • Todo:

"""Notes and warnings"""
# • when using trjectories with different lengths be careful when reversing the trajectory for computing the
#   retrace loss because the trailing zeros should NOT be reversed!
# • target_Q_next[:, -1] has always the same value???
# •