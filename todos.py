"""To-Do's"""
# • Todo: Try gradient clipping (norm?, value?) -> DONE
# • Todo: Implement retrace without division -> DONE
# • Todo: Try different weight initialization
# • Todo: target_Q_next[:, -1] has always the same value. Why?
# • Todo: Try hardtanh instead of tanh


"""Notes and warnings"""
# • when using trjectories with different lengths be careful when reversing the trajectory for computing the
#   retrace loss because the trailing zeros should NOT be reversed!
# • target_Q_next[:, -1] has always the same value???
# •