"""
    Same environment in on-policy varibad, with code refactor
    https://github.com/lmzintgraf/varibad/blob/master/environments/mujoco/humanoid_dir.py
"""
import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from gym import spaces

import random


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return np.sum(mass * xpos, 0) / np.sum(mass)


class HumanoidDirEnv(HumanoidEnv):
    def __init__(self, n_tasks=None, max_episode_steps=200):
        self.n_tasks = n_tasks
        assert n_tasks == None
        self._goal = self._sample_raw_task()["goal"]
        self._max_episode_steps = max_episode_steps
        super(HumanoidDirEnv, self).__init__()

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])

        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        goal_direction = (np.cos(self._goal), np.sin(self._goal))

        lin_vel_cost = (
            0.25
            * np.sum(goal_direction * (pos_after - pos_before))
            / self.model.opt.timestep
        )
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos

        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        # done = False

        return (
            self._get_obs(),
            reward,
            done,
            dict(
                reward_linvel=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
            ),
        )

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def get_current_task(self):
        # for multi-task MDP
        return np.array([np.cos(self._goal), np.sin(self._goal)])

    def _sample_raw_task(self):
        # uniform on unit circle
        direction = np.random.uniform(0.0, 2.0 * np.pi)  # 180 degree
        task = {"goal": direction}
        return task

    def reset_task(self, task_info):
        assert task_info is None
        self._goal = self._sample_raw_task()[
            "goal"
        ]  # assume parameterization of task by single vector
        self.reset()
