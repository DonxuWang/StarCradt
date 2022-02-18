from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from smac.env.multiagentenv import MultiAgentEnv
from gym.spaces import Tuple

import ray
from ray.tune import run_experiments, register_env

from smac.examples.rllib.env import RLlibStarCraft2Env
from smac.examples.covo.env import MyStarCraft2Env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--map-name", type=str, default="8m")
    args = parser.parse_args()

    def env_creator(smac_args):
        env = MyStarCraft2Env(**smac_args)
        agent_list = list(range(env.old_env._env.n_agents))
        grouping = {
            "group_1": agent_list,
        }
        obs_space = Tuple([env.observation_space for i in agent_list])
        act_space = Tuple([env.action_space for i in agent_list])
        return env.with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space
        )

    test = 1
    if test == 1:
        # env = RLlibStarCraft2Env(map_name=args.map_name,)
        # env = MyStarCraft2Env(map_name=args.map_name, communication_config={"communication":False,"message_allies":True,"message_enemies":True,"message_simple":True})
        env = MyStarCraft2Env(
            map_name=args.map_name,
            voting_config={
                "voting": False,
            },
            masked_actions = True,
            communication_config={
                "communication": True,
                "message_simple": True,
                "message_allies": False,
                "message_enemies": False,
                "message_range": 2,
            },
            replay_config={
                "save_replay": True,
                "replay_steps": 80,

            },
        )
        for i in range(10):
            env.reset()
            while 1:
                actions = {}
                for id in range(len(env._ready_agents)):
                    actions[id] = env.action_space.sample()

                ob, reward, done, info = env.step(actions)
                print(reward)
                print(done)

    else:

        ray.init()
        register_env("sc2_grouped", env_creator)

        run_experiments(
            {
                "qmix_sc2": {
                    "run": "QMIX",
                    "env": "sc2_grouped",
                    "stop": {
                        "training_iteration": args.num_iters,
                    },
                    "config": {
                        "num_workers": args.num_workers,
                        "env_config": {
                            "map_name": args.map_name,
                            "communication_config": {
                                "communication": True,
                                "message_allies": True,
                                "message_enemies": True,
                            },
                        },
                    },
                },
            }
        )
