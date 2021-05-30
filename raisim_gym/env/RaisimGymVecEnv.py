import numpy as np
import os
from gym import spaces
from stable_baselines.common.vec_env import VecEnv


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RaisimGymVecEnv(VecEnv):

    def __init__(self, impl, normalize_ob=False, normalize_rwd = True, clip_obs=10.0):
        self.wrapper = impl
        self.wrapper.init()
        self.num_obs = self.wrapper.getObDim()
        # self.num_student_obs = self.wrapper.getStudentObDim()
        # self.time_series_len = self.wrapper.getTimeSeriesLen()
        # self.partial_obs_dim = self.wrapper.getPartialObsDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf,
                                             dtype=np.float32)
        self._action_space = spaces.Box(np.ones(self.num_acts) * -1., np.ones(self.num_acts) * 1., dtype=np.float32)

        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        # self._student_observation = np.zeros([self.num_envs, self.num_student_obs], dtype=np.float32)

        self._gait_strings = np.array([''] * self.num_envs, dtype='<U16').reshape((self.num_envs,1))

        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)

        # self._extraInfoNames = self.wrapper.getExtraInfoNames()
        # self._extraInfo = np.zeros([self.num_envs, len(self._extraInfoNames)], dtype=np.float32)
        # self.num_extras = self.wrapper.getExtraInfoDim()
        # self._extraInfo = np.zeros([self.num_envs, self.num_extras], dtype=np.float32)
        self.rewards = [[] for _ in range(self.num_envs)]

        self.normalize_ob = normalize_ob
        self.normalize_rwd = normalize_rwd
        self.obs_rms = RunningMeanStd(shape=[self.num_obs])
        self.rwd_rms = RunningMeanStd(shape=[])
        self.clip_obs = clip_obs

        self.num_extras = self.wrapper.getExtrasDim()
        self._extra_plots = np.zeros([self.num_envs, self.num_extras], dtype=np.float32)

        self._target_velocity = np.zeros([self.num_envs, 3], dtype=np.float32) # target velocity as [x, y, yaw]

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def step(self, action, visualize=False):
        if not visualize:
            self.wrapper.step(action, self._observation, self._reward, self._done)
        else:
            self.wrapper.testStep(action, self._observation, self._reward, self._done)

        # if len(self._extraInfoNames) is not 0:
        #     info = [{'extra_info': {self._extraInfoNames[j]: self._extraInfo[i, j]
        #                             for j in range(0, len(self._extraInfoNames))}} for i in range(self.num_envs)]
        # else:
        #     info = [{} for i in range(self.num_envs)]
        info = [{} for i in range(self.num_envs)]


        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])

            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        if self.normalize_rwd:
            self.rwd_rms.update(self._reward)
            self._reward = np.clip(self._reward / np.sqrt(self.rwd_rms.var + 1e-8), -self.clip_obs,
                          self.clip_obs)

        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name):
        mean_file_name = dir_name + "/mean.csv"
        var_file_name = dir_name + "/var.csv"
        self.obs_rms.mean = np.loadtxt(mean_file_name)
        self.obs_rms.var = np.loadtxt(var_file_name)

    def save_scaling(self, dir_name):
        mean_file_name = dir_name + "/mean.csv"
        var_file_name = dir_name + "/var.csv"
        for path in [mean_file_name, var_file_name]:
            if not os.path.exists(path):
                file_handle = open(path, 'w+')
                file_handle.close()

        np.savetxt(mean_file_name, self.obs_rms.mean)
        np.savetxt(var_file_name, self.obs_rms.var)

    def observe(self, update_mean=True):
        self.wrapper.observe(self._observation)
        # return self._observation.copy()
        if self.normalize_ob:
            return self._normalize_observation(self._observation, update_mean)
        else:
            return self._observation.copy()

    def observe_student(self, update_mean=True):
        self.wrapper.observeStudent(self._student_observation)
        return self._student_observation.copy()

    def set_velocity_target(self, x, y, yaw):
        self._target_velocity[:,0] = x
        self._target_velocity[:,1] = y
        self._target_velocity[:,2] = yaw

        self.wrapper.setTargetVelocity(self._target_velocity)
        return

    def get_extras(self):
        self.wrapper.getExtraInfo(self._extra_plots)
        return self._extra_plots.copy()

    def set_gait_string(self, gait):
        gait_strings = [gait] * self.num_envs
        for i in range(self.num_envs):
            self._gait_strings[i] = gait

        self.wrapper.setGaitString(gait_strings)
        return

    def get_gait_string(self):
        self._gait_strings = self.wrapper.getGaitString()
        return self._gait_strings.copy()

    def _normalize_observation(self, obs, training=True):
        if self.normalize_ob:
            if training:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs,
                          self.clip_obs)
            return obs
        else:
            return obs

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        # self.wrapper.reset(self._observation)
        # return self._observation.copy()

        self.wrapper.reset(self._observation)

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()

        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def start_recording_video(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_recording_video(self):
        self.wrapper.stopRecordingVideo()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')

    def show_window(self):
        self.wrapper.showWindow()

    def hide_window(self):
        self.wrapper.hideWindow()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames
