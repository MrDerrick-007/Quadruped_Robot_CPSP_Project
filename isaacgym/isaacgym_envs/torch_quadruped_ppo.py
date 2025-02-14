import isaacgym

import torch
import torch.nn as nn
import wandb
import yaml


# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from tasks import Quadruped
from tasks.base import VecTask, Env
# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)  #1 è il value

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}


# load and wrap the Isaac Gym environment
    #     """Load an Isaac Gym environment (preview 3)

    # Isaac Gym benchmark environments: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

    # :param task_name: The name of the task (default: ``""``).
    #                   If not specified, the task name is taken from the command line argument (``task=TASK_NAME``).
    #                   Command line argument has priority over function parameter if both are specified
    # :type task_name: str, optional
    # :param num_envs: Number of parallel environments to create (default: ``None``).
    #                  If not specified, the default number of environments defined in the task configuration is used.
    #                  Command line argument has priority over function parameter if both are specified
    # :type num_envs: int, optional
    # :param headless: Whether to use headless mode (no rendering) (default: ``None``).
    #                  If not specified, the default task configuration is used.
    #                  Command line argument has priority over function parameter if both are specified
    # :type headless: bool, optional
    # :param cli_args: IsaacGymEnvs configuration and command line arguments (default: ``[]``)
    # :type cli_args: list of str, optional
    # :param isaacgymenvs_path: The path to the ``isaacgymenvs`` directory (default: ``""``).
    #                           If empty, the path will obtained from isaacgymenvs package metadata
    # :type isaacgymenvs_path: str, optional
    # :param show_cfg: Whether to print the configuration (default: ``True``)
    # :type show_cfg: bool, optional

    # :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments
    # :raises RuntimeError: The isaacgymenvs package is not installed or the path is wrong

    # :return: Isaac Gym environment (preview 3)
    # :rtype: isaacgymenvs.tasks.base.vec_task.VecTask
    # """
#env = load_isaacgym_env_preview4(task_name="Quadruped")
#env = Quadruped
with open("./cfg/task/Quadruped.yaml", "r") as file:
    env_cfg = yaml.safe_load(file) 

headless_flag = False
env = Quadruped( cfg=env_cfg, 
                rl_device='cuda:0', 
                sim_device='cuda:0', 
                graphics_device_id=2, 
                headless=headless_flag, 
                virtual_screen_capture= False, 
                force_render= not headless_flag)


env = wrap_env(env)
device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 3  # 24 * 128 / 32768
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 120
cfg["experiment"]["checkpoint_interval"] = 1200
cfg["experiment"]["directory"] = "runs/torch/Quadruped"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

##############################################################################
#TO COMMENT WHEN YOU DO NOT WANT TO LOAD CHECKPOINTS
# Load the checkpoint
agent.load("./runs/torch/Quadruped/optimal/checkpoints/agent_82800.pt")
##############################################################################

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 24000000, "headless": True} 

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

#------------------------------------------------
# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",
# )

# # simulate training
# epochs = 10
# for epoch in range(2, epochs):
#     acc = 1 - 2 ** -epoch 
#     loss = 2 ** -epoch

#     # log metrics to wandb
#     wandb.log({"acc": acc, "loss": loss})
#-------------------------------------------
# start training
trainer.train()


# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# # download the trained agent's checkpoint from Hugging Face Hub and load it
# path = download_model_from_huggingface("skrl/IsaacGymEnvs-Anymal-PPO", filename="agent.pt")
# agent.load(path)

# # start evaluation
# trainer.eval()
