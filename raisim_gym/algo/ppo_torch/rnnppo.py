import copy
from raisim_gym.algo.ppo_torch.memory import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np

from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Reference implementations:
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py
# https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py

class TensorboardLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def launchTensorboard(self):
        from tensorboard import program
        import webbrowser
        # learning visualizer
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir, '--host', 'localhost'])
        url = tb.launch()
        print("[RAISIM_GYM] Tensorboard session created: " + url)
        webbrowser.open_new(url)

    def __call__(self, scope, data, idx):
        for key in data:
            self.writer.add_scalar(scope + "/" + key, data[key], idx)

    def add_scalar(self, scope, key, data, idx):
        self.writer.add_scalar(scope + "/" + key, data, idx)

class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.log_std = nn.Parameter(np.log(init_std) * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        self.distribution = MultivariateNormal(logits, covariance)

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(logits, covariance)

        actions_log_prob = distribution.log_prob(outputs)
        entropy = distribution.entropy()

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

class ActorCritic(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim,
        policy_noise, init_std=1.0, is_recurrent=True
    ):
        super(ActorCritic, self).__init__()
        self.recurrent = is_recurrent
        self.action_dim = action_dim

        if self.recurrent:
            self.l1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)

        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.distribution = MultivariateGaussianDiagonalCovariance(action_dim, init_std)
        self.critic = nn.Linear(hidden_dim, 1)

        self.policy_noise = policy_noise

    def forward(self, state, hidden):
        if self.recurrent:
            self.l1.flatten_parameters()
            p, h = self.l1(state.unsqueeze(dim=0), hidden)
        else:
            p, h = torch.tanh(self.l1(state)), None

        p = self.l2(p.data)
        return p, h

    def act(self, state, hidden):
        p, h = self.forward(state, hidden)
        action = self.actor(p)

        return action, h

    def sample(self, state, hidden):
        logits, h = self.act(state, hidden)
        action, log_prob = self.distribution.sample(logits.squeeze(dim=0))

        return action, h, log_prob

    def evaluate(self, state, action, hidden):
        p, h = self.forward(state, hidden)
        action_mean, _ = self.act(state, hidden)

        action_logprob, entropy = self.distribution.evaluate(action_mean, action)

        values = self.critic(p)

        if self.recurrent:
            values = values[..., 0]
        else:
            action_logprob = action_logprob[..., None]

        return values, action_logprob, entropy


class PPO(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        fname="",
        load_params=True,
        memory_size = 1e6,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        eps_clip=.2,
        lmbda=0.95,
        lr=3e-4,
        K_epochs=80,
        recurrent_actor=False,
        recurrent_critic=False,
    ):
        self.on_policy = True
        self.recurrent = recurrent_actor
        self.actorcritic = ActorCritic(
            state_dim, action_dim, hidden_dim, policy_noise,
            is_recurrent=recurrent_actor
        ).to(device)
        self.target = copy.deepcopy(self.actorcritic)
        self.optimizer = torch.optim.Adam(self.target.parameters())

        # log data - for Tensorboard plots
        self.log_data = dict()

        self.discount = discount
        self.lmbda = lmbda
        self.tau = tau
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.actor_loss_coeff = 1.
        self.critic_loss_coeff = 0.74
        self.entropy_loss_coeff = 0.01

        self.replay_buffer = ReplayBuffer(
            state_dim, action_dim, hidden_dim, memory_size, recurrent_actor
        )

        self.fname = fname
        if load_params:
            self.load(f"{fname}")

    def get_initial_states(self):
        h_0, c_0 = None, None
        if self.actorcritic.recurrent:
            h_0 = torch.zeros((
                self.actorcritic.l1.num_layers,
                1,
                self.actorcritic.l1.hidden_size),
                dtype=torch.float)
            h_0 = h_0.to(device=device)

            c_0 = torch.zeros((
                self.actorcritic.l1.num_layers,
                1,
                self.actorcritic.l1.hidden_size),
                dtype=torch.float)
            c_0 = c_0.to(device=device)
        return (h_0, c_0)

    def reshape_input(self, state):
        if self.recurrent:
            state = torch.FloatTensor(
                state.reshape(1, -1)).to(device)[:, None, :]
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        return state

    def select_action(self, state, hidden):
        # state = self.reshape_input(state)

        action, hidden = self.actorcritic.act(state, hidden)
        return action.cpu().data.numpy().flatten(), hidden

    def sample_action(self, state, hidden):
        # state = self.reshape_input(state)

        action, hidden, _ = self.actorcritic.sample(state, hidden)
        return action.cpu().data.numpy(), hidden

    def train(self):

        # Sample replay buffer
        state, action, next_state, reward, not_done, hidden, next_hidden = \
            self.replay_buffer.on_policy_sample()

        running_actor_loss = 0
        running_critic_loss = 0

        discounted_reward = 0
        rewards = []

        for r, is_terminal in zip(reversed(reward), reversed(1 - not_done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = r + (self.discount * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards[..., None]

        # log_prob of pi(a|s)
        _, prob_a, _ = self.actorcritic.evaluate(
            state,
            action,
            hidden)

        # TODO: PPO Update
        # PPO allows for multiple gradient steps on the same data
        for _ in range(self.K_epochs):

            # V_pi'(s) and pi'(a|s)
            v_s, logprob, dist_entropy = self.target.evaluate(
                state,
                action,
                hidden)

            assert rewards.size() == v_s.size(), \
                '{}, {}'.format(rewards.size(), v_s.size())
            # Finding Surrogate Loss:
            advantages = rewards - v_s

            # Ratio between probabilities of action according to policy and
            # target policies

            assert logprob.size() == prob_a.size(), \
                '{}, {}'.format(logprob.size(), prob_a.size())
            ratio = torch.exp(logprob - prob_a)

            # Surrogate policy loss
            assert ratio.size() == advantages.size(), \
                '{}, {}'.format(ratio.size(), advantages.size())

            surrogate_policy_loss_1 = ratio * advantages
            surrogate_policy_loss_2 = torch.clamp(
                ratio,
                1-self.eps_clip,
                1+self.eps_clip) * advantages
            # PPO "pessimistic" policy loss
            actor_loss = -torch.min(
                surrogate_policy_loss_1,
                surrogate_policy_loss_2)

            # Surrogate critic loss: MSE between "true" rewards and prediction
            # TODO: Investigate size mismatch
            assert(v_s.size() == rewards.size())

            surrogate_critic_loss_1 = F.mse_loss(
                v_s,
                rewards)
            surrogate_critic_loss_2 = torch.clamp(
                surrogate_critic_loss_1,
                -self.eps_clip,
                self.eps_clip
            )
            # PPO "pessimistic" critic loss
            critic_loss = torch.max(
                surrogate_critic_loss_1,
                surrogate_critic_loss_2)

            # Entropy "loss" to promote entropy in the policy
            entropy_loss = dist_entropy[..., None].mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss = ((critic_loss * self.critic_loss_coeff) +
                    (self.actor_loss_coeff * actor_loss) -
                    (entropy_loss * self.entropy_loss_coeff))
            # print(loss.size(), loss)
            loss.mean().backward(retain_graph=True)
            # print([p.grad for p in self.target.parameters()])
            nn.utils.clip_grad_norm_(self.target.parameters(),
                                     0.5)
            self.optimizer.step()

            # Keep track of losses
            running_actor_loss += actor_loss.mean().cpu().detach().numpy()
            running_critic_loss += critic_loss.mean().cpu().detach().numpy()

        # update the log
        self.log_data['mean_std'] = self.actorcritic.distribution.log_std.exp().mean().item()
        self.log_data['value_function_loss'] = critic_loss.mean().item()
        self.log_data['surrogate_loss'] = loss.mean().item()
        
        self.save(self.fname)

        self.actorcritic.load_state_dict(self.target.state_dict())
        torch.cuda.empty_cache()

    def save(self, filename):
        torch.save(self.actorcritic.state_dict(), filename)
        torch.save(self.optimizer.state_dict(),
                   filename + "_optimizer")

    def load(self, filename):
        if(os.path.isfile(filename)):
            self.actorcritic.load_state_dict(torch.load(filename))
        if(os.path.isfile(filename + "_optimizer")):
            self.optimizer.load_state_dict(
                torch.load(filename + "_optimizer"))

    def eval_mode(self):
        self.actorcritic.eval()

    def train_mode(self):
        self.actorcritic.train()

    def learn(self, env, saver, num_transitions_per_env):
        tensorboard_logger = TensorboardLogger(log_dir=saver.data_dir)
        tensorboard_logger.launchTensorboard()

        n_steps = num_transitions_per_env
        total_steps = n_steps * env.num_envs
        avg_rewards = []

        for update in range(1000000):
            start = time.time()
            env.reset()
            reward_sum = 0
            done_sum = 0

            if update % 50 is 0:
                env.show_window()
                env.start_recording_video(saver.data_dir + "/" + str(update) + ".mp4")
                hidden = self.get_initial_states()
                env.reset()
                for _ in range(n_steps):
                    obs = env.observe()
                    action, next_hidden = self.sample_action(torch.from_numpy(obs).to(device), hidden)
                    env.step(action, True)
                    hidden = next_hidden
                env.stop_recording_video()
                env.hide_window()
                del hidden

            # actual training
            hidden = self.get_initial_states()
            env.reset()
            obs = env.observe()
            for step in range(n_steps):
                action, next_hidden = self.sample_action(torch.from_numpy(obs).to(device), hidden)
                reward, dones = env.step(action, False)
                next_obs = env.observe()

                self.replay_buffer.add(obs, action, next_obs, reward, dones, hidden, next_hidden)

                hidden = next_hidden
                obs = next_obs

                done_sum = done_sum + sum(dones)
                reward_sum = reward_sum + sum(reward) * np.sqrt(env.rwd_rms.var)

            # update the policy parameters by backpropagation
            self.train()
            self.replay_buffer.clear_memory()

            end = time.time()

            average_ll_performance = reward_sum / total_steps
            average_dones = done_sum / total_steps
            avg_rewards.append(average_ll_performance)

            tensorboard_logger("ppo", self.log_data, update)
            tensorboard_logger.add_scalar("ppo", "dones", average_dones, update)
            tensorboard_logger.add_scalar("ppo", "mean_reward", average_ll_performance, update)

            print('----------------------------------------------------')
            print('{:>6}th iteration'.format(update))
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
            print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
            print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
            print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
            print('----------------------------------------------------\n')