import os
import random
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from env.utils import FlattenMultiDiscreteActions
from env.cantilever_env import CantileverEnv_v0_1
from env.pyansys_sim import Cantilever
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.policy.modelfree.ppo import PPOTrainingStats
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic
config_kwargs = {
"task":"CantileverEnv_v0_1-Wrapped",
"seed":0,
"buffer_size":1_000_000,
"lr":3e-4,
"gamma":0.9,
"hidden_sizes":[128, 128],
"episode_per_test":5,
"num_test_env": 5,
"logdir":"log",
"device" : "cuda" if torch.cuda.is_available() else "cpu",
"vf_coef":0.5,
"ent_coef":0.0,
"eps_clip":0.2,
"max_grad_norm":0.5,
"gae_lambda":0.95,
"rew_norm":0,
"norm_adv":0,
"recompute_adv":0,
"dual_clip":None,
"value_clip":0,
#Ansys environment setup parameters 
"core_path": None, #Specify ansys-mechanical-core directory 
"sim_modes": [0,1,2], #mode index
"num_sensors": 4, #NUmber of Sensors
"render" : True,
"norm" : True,  #Mode Shape Normalisation
"eps_length" : 200
    }

def set_random_seeds(seed: int, using_cuda: bool = False) -> None:
  """
  Seed the different random generators.
  """
  # Set seed for Python random, NumPy, and Torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Set deterministic operations for CUDA
  if using_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#Environment setup
pyansys_env = Cantilever(config_kwargs)
config_kwargs['pyansys_env'] = pyansys_env

envs = FlattenMultiDiscreteActions(CantileverEnv_v0_1(config_kwargs))
state_shape = envs.observation_space.shape
action_shape = envs.action_space.n

# Set random seed
set_random_seeds(config_kwargs["seed"], using_cuda=torch.cuda.is_available())

# model
net = Net(state_shape=state_shape, hidden_sizes= config_kwargs.get("hidden_sizes"), device=config_kwargs.get("device"))
actor: nn.Module
critic: nn.Module
if torch.cuda.is_available():
    actor = DataParallelNet(Actor(net, action_shape, device=config_kwargs.get("device")).to(config_kwargs.get("device")))
    critic = DataParallelNet(Critic(net, device=config_kwargs.get("device")).to(config_kwargs.get("device")))
else:
    actor = Actor(net, action_shape, device=config_kwargs.get("device")).to(config_kwargs.get("device"))
    critic = Critic(net, device=config_kwargs.get("device")).to(config_kwargs.get("device"))
actor_critic = ActorCritic(actor, critic)
# orthogonal initialization
for m in actor_critic.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)
optim = torch.optim.Adam(actor_critic.parameters(), lr=config_kwargs.get("lr"))
dist = torch.distributions.Categorical
policy: PPOPolicy[PPOTrainingStats] = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=dist,
    action_scaling=isinstance(envs.action_space, Box),
    discount_factor=config_kwargs.get("gamma"),
    max_grad_norm=config_kwargs.get("max_grad_norm"),
    eps_clip=config_kwargs.get("eps_clip"),
    vf_coef=config_kwargs.get("vf_coef"),
    ent_coef=config_kwargs.get("ent_coef"),
    gae_lambda=config_kwargs.get("gae_lambda"),
    reward_normalization=config_kwargs.get("rew_norm"),
    dual_clip=config_kwargs.get("dual_clip"),
    value_clip=config_kwargs.get("value_clip"),
    action_space=envs.action_space,
    deterministic_eval=True,
    advantage_normalization=config_kwargs.get("norm_adv"),
    recompute_advantage=config_kwargs.get("recompute_adv"),
)
    
log_path = os.path.join( config_kwargs.get("logdir"),  config_kwargs.get("task"), "ppo")
policy.load_state_dict(torch.load(os.path.join(log_path,"final_policy.pth")))
#policy.load_state_dict(torch.load(os.path.join(log_path, "best_policy.pth")))
policy.eval()
buf_test = VectorReplayBuffer(config_kwargs.get("num_test_env") *config_kwargs.get('eps_length') \
                              *config_kwargs.get("episode_per_test"), buffer_num= config_kwargs.get("num_test_env")) 
buf_test.reset()
eval_collector = Collector(policy, envs, buf_test, exploration_noise=False)
eval_collector.collect(n_episode = 1, reset_before_collect=True)
# Save evaluation results
ep_rew = np.sum(buf_test.get(np.arange(config_kwargs.get('eps_length')),"rew" ))
rew_metric = buf_test.get(np.arange(config_kwargs.get('eps_length')),"info" )['reward_metric']
ep_rew_metric_sum  = np.sum(rew_metric)
ep_rew_metric_final = rew_metric[-1]
final_node_id = buf_test.get(np.arange(config_kwargs.get('eps_length')),"info" )['node_Id'][-1]

print('----------------------------------------------------------------')
print(f"Evaluation Final node_id: {final_node_id}")
print(f"Evaluation episode reward: {ep_rew}")
print(f"Evaluation episode reward metric sum: {ep_rew_metric_sum}")
print(f"Evaluation episode reward metric final: {ep_rew_metric_final}")
