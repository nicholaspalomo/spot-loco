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

import numpy as np
from matplotlib import pyplot as plt
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

    # specify buffers for the plot values
    extras = np.zeros((1, env.num_extras, int(num_steps)))

    env.show_window()
    env.reset()
    env.start_recording_video(save_dir + "/spot.mp4")

    for i in range(int(num_steps)):
        extras[0, :, i] = env.get_extras()

        state = observe_callback(False)

        action = ctrl_callback(torch.from_numpy(state).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))).detach().cpu().data.numpy()

        _, _ = env.step(action, visualize=True)

    env.hide_window()
    env.stop_recording_video()

    # make the graphs...
    time = np.linspace(0., max_time, num=int(num_steps), endpoint=True)

    # velocity tracking
    start_idx = 0 # target velocity, 3: bodyLinVel_, 6: bodyAngularVel_

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(xlabel = 'Time [s]', ylabel='Body Velocity')
    ax.grid()

    ax.plot(time, extras[0, start_idx, :]) # x
    ax.plot(time, extras[0, start_idx+3, :])
    ax.plot(time, extras[0, start_idx+1, :]) # y
    ax.plot(time, extras[0, start_idx+1+3, :])
    ax.plot(time, extras[0, start_idx+2, :]) # r
    ax.plot(time, extras[0, start_idx+3+3+2, :])
    ax.legend(['$x_{target}$ [m/s]', 'x [m/s]', '$y_{target}$ [m/s]', 'y [m/s]', '$r_{target}$ [rad/s]', 'r [rad/s]'], ncol=2, bbox_to_anchor=(0.75,-0.25))
    fig.tight_layout()
    fig.savefig(save_dir + '/body_velocity.png', bbox_inches='tight')
    plt.close(fig)

    # gait following
    start_idx = 3 + 3 + 3

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].set(xlabel = '', ylabel='Contact state')
    ax[1].set(xlabel = 'Time [s]', ylabel='Gait error')
    ax[1].grid()

    color = ['red', 'green', 'blue', 'orange', 'black']
    for i in range(4):
        for t in range(time.shape[0]):
            if extras[0, start_idx+4+i, t] > 0.: # contact state
                ax[0].vlines(t * cfg["environment"]["control_dt"], i, i+1, colors=color[i], alpha=0.5)
            if extras[0, start_idx+8+i, t] > 0.: # target contact state
                ax[0].vlines(t * cfg["environment"]["control_dt"], i, i+1, colors=color[-1], alpha=0.5)
        ax[1].plot(time, extras[0,start_idx+i,:], color=color[i])
        
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])

    ax[1].legend(["LF", "RF", "LH", "RH"], ncol=4, bbox_to_anchor=(0.75,-0.25))
    leg = ax[1].get_legend()
    for i in range(4):
        leg.legendHandles[i].set_color(color[i])

    fig.tight_layout()
    fig.savefig(save_dir + '/gait_following.png', bbox_inches='tight')
    plt.close(fig)

    # target following BEV

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel = 'Position x [m]', ylabel='Position y [m]')
    ax.grid()

    start_idx = 3*3 + 4*3
    for t in range(time.shape[0]):
        if t % 20 == 0:
            # target velocity, x-y
            target_vel = extras[0, start_idx + 8 + 2:start_idx + 8 + 2 + 2, t]
            x = extras[0, start_idx + 8, t]
            y = extras[0, start_idx + 8 + 1, t]
            origin = np.array([[x], [y]])
            ax.quiver(*origin, target_vel[0], target_vel[1], color='c', scale=5, alpha=0.3, edgecolors='r', headwidth=2, headlength=4, width=0.004)

            # target velocity, yaw
            target_yaw = extras[0, start_idx + 8 + 2 + 2, t]
            if target_yaw >= 0:
                ax.scatter(x, y, color='m', marker='o', s=250 * target_yaw, alpha=0.3)
            else:
                ax.scatter(x, y, color='m', marker='x', s=250 * -target_yaw, alpha=0.3)

            # current velocity, x-y
            curr_vel = extras[0, start_idx + 8 + 2 + 3:start_idx + 8 + 2 + 3 + 2, t]
            ax.quiver(*origin, curr_vel[0], curr_vel[1], color='c', scale=5, headwidth=2, headlength=4, width=0.004)

            # current velocity, yaw
            curr_yaw = extras[0, start_idx + 8 + 2 + 3 + 2, t]
            if curr_yaw >= 0:
                ax.scatter(x, y, color='m', marker='o', s=250 * curr_yaw)
            else:
                ax.scatter(x, y, color='m', marker='x', s=250 * -curr_yaw)

        for i in range(4):
            if extras[0, 3*3 + 4 + i, t] > 0:
                ax.scatter(extras[0, 3*3 + 4*3 + i, t], extras[0, 3*3 + 4*3 + 1 + i, t], color=color[i], marker='o', s=20) # current contact state
            if extras[0, 3*3 + i, t] > 0:
                ax.scatter(extras[0, 3*3 + 4*3 + i, t], extras[0, 3*3 + 4*3 + i, t], color=color[i], alpha=0.3, marker='o', s=20) # desired contact state

    fig.savefig(save_dir + '/foot_position_xy.png', bbox_inches='tight')
    plt.close(fig)

    # average velocity error
    print("average x-direction velocity error: {}".format(np.mean(extras[0, 0, :] - extras[0, 3, :])))
    print("average y-direction velocity error: {}".format(np.mean(extras[0, 1, :] - extras[0, 4, :])))
    print("average y-direction velocity error: {}".format(np.mean(extras[0, 2, :] - extras[0, 8, :])))

    # average % of time gait in phase
    print("average percent time gait in phase: {}".format(np.count_nonzero(np.where(np.sum(abs(extras[0, 9:13, :]), axis=0) == 0)[0]) / time.shape[0] * 100.))

def save_to_torchscript(actor, num_obs):

    dummy_input = torch.ones(num_obs)

    torchscript_module = torch.jit.trace(actor.architecture.architecture.eval().to('cpu'), dummy_input)

    print(torchscript_module.code)

    torch.jit.save(torchscript_module, "loco_controller.pt")

    print("[spot.py] Done! Locomotion network has been serialized for inferencing in C++.")

    print(actor.noiseless_action(dummy_input))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, help="create plots? true or false", default=True)
    parser.add_argument("--max_time", type=int, help="maximum time length of policy rollout", default=10)
    parser.add_argument("--torchscript", type=bool, help="serialize controller to torchscript file for inferencing in C++", default=False)
    
    args = parser.parse_args()

    main(args)