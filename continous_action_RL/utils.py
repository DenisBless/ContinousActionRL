import torch


class Utils:

    @staticmethod
    def create_batches(trajectories, trajectory_length, minibatch_size, num_obs, num_actions):
        state_batch = torch.zeros(size=[minibatch_size, trajectory_length, num_obs])
        action_batch = torch.zeros(size=[minibatch_size, trajectory_length, num_actions])
        reward_batch = torch.zeros(size=[minibatch_size, trajectory_length, 1])
        action_prob_batch = torch.zeros(size=[minibatch_size, trajectory_length, 1])
        for i in range(len(trajectories)):
            state_batch[i, :trajectories[i].state.shape[0]] = trajectories[i].state
            action_batch[i, :trajectories[i].action.shape[0]] = trajectories[i].action.unsqueeze(1)
            reward_batch[i, :trajectories[i].reward.shape[0]] = trajectories[i].reward.unsqueeze(1)
            action_prob_batch[i, :trajectories[i].action_prob.shape[0]] = trajectories[i].action_prob.unsqueeze(1)
        return state_batch, action_batch, reward_batch, action_prob_batch

    @staticmethod
    def freeze_net(net):
        for params in net.parameters():
            params.requires_grad = False
