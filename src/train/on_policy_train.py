import numpy as np
import gymnasium as gym
from src.config.configure import RunConfig
from src.utils.mpi_tools import num_procs
from src.common.base_agent import OnPolicyAgent
from src.common.buffer import OnPolicyBuffer
from src.train.policy_train import BaseTrainer


class OnPolicyTrain(BaseTrainer):
    def __init__(self, configure: RunConfig = None, env: gym.Env = None) -> None:
        super().__init__(configure, env)
        self._ep_len = 0
        self._ep_ret = 0

    def _init_env(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._ep_ret, self._ep_len = 0, 0

        return obs

    def train(self, agent: OnPolicyAgent = None) -> None:
        self._log_agent_param(agent)

        # Setup buffer
        # act_dim = self._get_act_dim()
        buffer_size = int(
            self.configure.train_config.total_steps
            / self.configure.train_config.epochs
            / num_procs()
        )
        buffer = OnPolicyBuffer(
            buffer_size=buffer_size,
            batch_size=buffer_size,
            obs_shape=self.env.observation_space.shape,
            act_shape=self.env.action_space.shape,
            device=agent.device,
        )

        # Random Seed
        self._set_random_seed()

        # Main loop: collect rollout episode in env and update agent with on-policy
        self.logger.log("Start to train the model!\n", color="green")
        for epoch in range(self.configure.train_config.epochs):
            self._agent_explore_learn(buffer=buffer, agent=agent)

            # store the nessarry information
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("EpRet", with_min_and_max=True)
            self.logger.log_tabular("EpLen", average_only=True)
            self.logger.log_tabular("LossPi", average_only=True)
            self.logger.log_tabular("LossV", average_only=True)
            self.logger.dump_tabular()
            if self.configure.train_config.render:
                self._render_env(agent)

    def _agent_explore_learn(
        self, buffer: OnPolicyBuffer = None, agent: OnPolicyAgent = None
    ) -> None:
        max_steps = buffer.buffer_size
        max_ep_steps = self.configure.train_config.on_policy_train_config.max_ep_len

        obs = self._init_env()
        for steps in range(max_steps):
            act, log_pi = agent.evaluate(obs)
            state_v = agent.calc_state_value(obs)
            next_obs, rew, done, _, info = self.env.step(act)
            buffer.store(obs, act, next_obs, rew, done, log_pi, state_v)
            obs = next_obs
            self._ep_ret += rew
            self._ep_len += 1

            if done or (steps + 1) == max_ep_steps or (steps + 1) == max_steps:
                if done:
                    last_state_v = 0
                else:
                    last_state_v = agent.calc_state_value(obs)
                buffer.finish_path(
                    last_state_v=last_state_v,
                    gamma=self.configure.agent_config.gamma,
                    lam=self.configure.agent_config.lam,
                )
                self.logger.store(EpRet=self._ep_ret, EpLen=self._ep_len)
                obs = self._init_env()

        agent.learn(buffer.get())
