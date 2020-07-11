import torch


class Utils:

    @staticmethod
    def create_batches(trajectories, trajectory_length, minibatch_size, num_obs, num_actions):

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        state_batch = torch.zeros(size=[minibatch_size, trajectory_length, num_obs], dtype=torch.float)
        action_batch = torch.zeros(size=[minibatch_size, trajectory_length, num_actions], dtype=torch.float)
        reward_batch = torch.zeros(size=[minibatch_size, trajectory_length, 1], dtype=torch.float)
        action_prob_batch = torch.zeros(size=[minibatch_size, trajectory_length, num_actions], dtype=torch.float)
        for i in range(len(trajectories)):
            state_batch[i, :] = trajectories[i].state
            action_batch[i, :] = trajectories[i].action
            reward_batch[i, :] = trajectories[i].reward#.unsqueeze(1)
            action_prob_batch[i, :] = trajectories[i].action_prob
        return state_batch.to(device), action_batch.to(device), reward_batch.to(device), action_prob_batch.to(device)

    @staticmethod
    def freeze_net(net):
        for params in net.parameters():
            params.requires_grad = False

    # @staticmethod
    # def create_batches_from_trajectories(trajectories, trajectory_length, minibatch_size, num_obs, num_actions):
    #     state_batch = torch.zeros(size=[minibatch_size, trajectory_length, num_obs])
    #     action_batch = torch.zeros(size=[minibatch_size, trajectory_length, num_actions])
    #     reward_batch = torch.zeros(size=[minibatch_size, trajectory_length, 1])
    #     action_prob_batch = torch.zeros(size=[minibatch_size, trajectory_length, 1])
    #     for i in range(len(trajectories)):
    #         state_batch[i, :trajectories[i].shape[0]] = trajectories[i][0]
    #         action_batch[i, :trajectories[i].shape[0]] = trajectories[i].unsqueeze(1)
    #         reward_batch[i, :trajectories[i].shape[0]] = trajectories[i].unsqueeze(1)
    #         action_prob_batch[i, :trajectories[i].shape[0]] = trajectories[i].unsqueeze(1)
    #     return state_batch, action_batch, reward_batch, action_prob_batch
