from gymnasium.envs.registration import register

import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

register(
    id="GridWorld-v1",
    entry_point="envs.gridworld_env_v1:GridWorldEnv",
)

register(
    id="GridWorld-v2",
    entry_point="envs.gridworld_env_v2:GridWorldEnvV2",
)

register(
    id="GridWorld-v3",
    entry_point="envs.gridworld_env_v3_multi:GridWorldMultiAgentEnv",
)
