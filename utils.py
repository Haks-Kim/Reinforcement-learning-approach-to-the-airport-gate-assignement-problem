import torch


def save_snapshot(path, critic, target_critic, critic_optim):
    # print('adding checkpoints...')
    checkpoint_path = path + 'model.pth.tar'
    torch.save(
        {
         'critic': critic.state_dict(),
         'target_critic': target_critic.state_dict(),
         'critic_optimizer': critic_optim.state_dict()
         },
        checkpoint_path)
    return


def recover_snapshot(path, critic, target_critic, critic_optim, device):
    print('recovering snapshot...')
    checkpoint = torch.load(path, map_location=torch.device(device))

    critic.load_state_dict(checkpoint['critic'])
    target_critic.load_state_dict(checkpoint['target_critic'])
    critic_optim.load_state_dict(checkpoint['critic_optimizer'])
    return


def load_model(agent, path, device):
    print('loading pre-trained weight...')
    checkpoint = torch.load(path, map_location=torch.device(device))
    agent.critic.load_state_dict(checkpoint['critic'])
    return


