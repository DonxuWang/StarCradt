from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from smac_communication.env import MyStarCraft2Env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--map-name", type=str, default="8m")
    args = parser.parse_args()

    env = MyStarCraft2Env(
        map_name=args.map_name,
        voting_config={
            "voting": True,
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
