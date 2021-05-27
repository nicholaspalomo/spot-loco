import os
import argparse

from ruamel.yaml import YAML, dump, RoundTripDumper
from torch.nn.modules.rnn import RNN
from raisim_gym.env.RaisimGymVecEnv import RaisimGymVecEnv as Environment
from raisim_gym.env.env.Spot import __SPOT_RESOURCE_DIRECTORY__ as __RSCDIR__
from _raisim_gym import RaisimGymEnv
from raisim_gym.helper.raisim_gym_helper import ConfigurationSaver

import raisim_gym.algo.ppo_torch.module as ppo_module
import raisim_gym.algo.ppo_torch.ppo as PPO
import raisim_gym.algo.ppo_torch.rnnppo as RNNPPO

import math
import torch
import torch.nn as nn

def main(args): # cfg, env, saver

    # directories
    root = os.path.dirname(os.path.abspath(__file__)) + '/../'
    log_path = root + '/data/Spot'

    cfg = YAML().load(open(__RSCDIR__ + "/default_cfg.yaml", 'r'))

    if args.plot:
        cfg['environment']['num_threads'] = 1
        cfg['environment']['num_envs'] = 1

    # create environment from the configuration file
    env = Environment(RaisimGymEnv(__RSCDIR__, dump(
        cfg['environment'], Dumper=RoundTripDumper)),
        normalize_ob=cfg['environment']['normalize_obs'],
        normalize_rwd=cfg['environment']['normalize_rwd'])

    # save the configuration and other files
    saver = ConfigurationSaver(log_dir=log_path,
        save_items=[__RSCDIR__ + "/default_cfg.yaml",
        root + 'raisim_gym/env/env/Spot/Environment.hpp'])
    print('[spot.py] logging in: ', saver.data_dir)

    device = cfg['device']

    env_yaml_node = cfg['environment']
    algorithm_yaml_node = env_yaml_node['algorithm']

    # Training
    n_steps = math.floor(env_yaml_node['max_time'] / env_yaml_node['control_dt'])

    # ppo = RNNPPO.PPO(env.num_obs, env.num_acts, 64, os.getcwd() + '/trained_params/actorcritic', K_epochs=4, recurrent_actor=True, recurrent_critic=True)

    # ppo.learn(env, saver, n_steps)

    actor = ppo_module.Actor(ppo_module.MLP(env_yaml_node['architecture']['policy'],  # number of layers and neurons in each layer
                        getattr(nn, env_yaml_node['architecture']['activation']),  # activation function at each layer
                        env.num_obs,  # number of states (input dimension)
                        env.num_acts,  # number of actions (output)
                        env_yaml_node['architecture']['init_scale']),
                        ppo_module.MultivariateGaussianDiagonalCovariance(env.num_acts, 1.0),
                        device)

    critic = ppo_module.Critic(ppo_module.MLP(env_yaml_node['architecture']['value_net'],
                            getattr(nn, env_yaml_node['architecture']['activation']),
                            env.num_obs,
                            1,
                            env_yaml_node['architecture']['init_scale']), device)

    ppo = PPO.PPO(actor=actor,
                critic=critic,
                num_envs=env_yaml_node['num_envs'],
                num_transitions_per_env=n_steps,
                num_learning_epochs=algorithm_yaml_node['epoch'],
                clip_param=algorithm_yaml_node['clip_param'],
                gamma=algorithm_yaml_node['gamma'],
                lam=algorithm_yaml_node['lambda'],
                entropy_coef=algorithm_yaml_node['entropy_coeff'],
                learning_rate=algorithm_yaml_node['learning_rate'],
                num_mini_batches=algorithm_yaml_node['minibatch'],
                device=device,
                nets_dir=['actor_architecture.pth', 'actor_distribution.pth', 'critic.pth']
                )

    # serialize the controller and create torchscript file
    if args.torchscript:
        save_to_torchscript(actor, env.num_obs)
        return

    # create plots and perform a rollout of the policy
    if args.plot:
        plot(args.max_time, actor, env, cfg, log_path)
        return

    ppo.learn(env, saver, env_yaml_node)

def plot(max_time, controller, env, cfg, save_dir):

    # load the controller parameters
    controller.architecture.load_state_dict(torch.load(os.getcwd() + '/trained_params/actor_architecture.pth', map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

    # compute the number of timesteps for the rollout
    num_steps = max_time / cfg['environment']['control_dt']

    # specify the control callback
    ctrl_callback = controller.noiseless_action
    observe_callback = env.observe

    env.show_window()
    env.reset()
    env.start_recording_video(save_dir + "/spot.mp4")

    for i in range(int(num_steps)):

        state = observe_callback(False)

        action = ctrl_callback(torch.from_numpy(state).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))).detach().cpu().data.numpy()

        _, _ = env.step(action, visualize=True)

    env.hide_window()
    env.stop_recording_video()

def save_to_torchscript(actor, num_obs):

    dummy_input = torch.ones(num_obs)

    torchscript_module = torch.jit.trace(actor.architecture.architecture.eval().to('cpu'), dummy_input)

    print(torchscript_module.code)

    torch.jit.save(torchscript_module, "loco_controller.pt")

    print("[spot.py] Done! Locomotion network has been serialized for inferencing in C++.")

    print(actor.noiseless_action(dummy_input))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, help="create plots? true or false", default=False)
    parser.add_argument("--max_time", type=int, help="maximum time length of policy rollout", default=10)
    parser.add_argument("--torchscript", type=bool, help="serialize controller to torchscript file for inferencing in C++", default=False)
    
    args = parser.parse_args()

    main(args)