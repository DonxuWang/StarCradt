from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Running StarCraft2 with RLlib PPO.

In this setup, each agent will be controlled by an independent PPO policy.
However the policies share weights.

Increase the level of parallelism by changing --num-workers.

"""

import argparse

import os
import ray
from ray.tune import run_experiments, register_env
from ray.rllib.models import ModelCatalog

from smac.examples.covo.env import MyStarCraft2Env
from smac.examples.covo.model import MaskedActionsModel

"""
Args:
    mode-name (int):
        0: general
        1: communication
        2: voting 
"""
if __name__ == "__main__":

    communication = voting = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-timesteps", type=int, default=1000000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--map-name", type=str, default="8m")
    parser.add_argument(
        "--local-dir",
        type=str,
        default=r'C:\Users\Bea\Desktop\Project\multi-agenten-reinforcement-learning-mit-consensus-algorithmen\results_new\2s3z',
    )
    parser.add_argument("--mode-name", type=int, default=1)
    parser.add_argument("--range", type=int, default=2)
    parser.add_argument("--experiment-name", type=str, default="PPO_SC2")
    parser.add_argument("--masked-actions", type=bool, default=True)
    parser.add_argument("--save-replay", type=bool, default=False)
    parser.add_argument("--replay-steps", type=int, default=1000)
    parser.add_argument("--replay-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--mini-batch", type=int, default=128)
    parser.add_argument("--num-epoch", type=int, default=30)

    args = parser.parse_args()

    replay_prefix = ""
    # dir name: local dir / mode name / map name / name
    if args.mode_name == 0:
        replay_prefix += "General"
        args.local_dir = os.path.join(args.local_dir, "General")
    if args.mode_name == 1:
        communication = True
        replay_prefix += "Communication"
        args.local_dir = os.path.join(args.local_dir, "Communication")
    if args.mode_name == 2:
        voting = True
        replay_prefix += "Voting"
        args.local_dir = os.path.join(args.local_dir, "Voting")
    if not os.path.exists(args.local_dir):
        os.mkdir(args.local_dir)

    if args.save_replay:
        args.local_dir = os.path.join(args.local_dir, args.map_name)
        if not os.path.exists(args.local_dir):
            os.mkdir(args.local_dir)

        replay_dir = '_'.join([replay_prefix, args.map_name, args.experiment_name])
        args.replay_dir = os.path.join(args.replay_dir, replay_dir)
        if not os.path.exists(args.replay_dir):
            os.mkdir(args.replay_dir)

    ray.init()

    register_env("smac", lambda smac_args: MyStarCraft2Env(**smac_args))
    ModelCatalog.register_custom_model("mask_model", MaskedActionsModel)

    if args.masked_actions:
        custom_model = {"custom_model": "mask_model",}
    else:
        custom_model = {"fcnet_hiddens": [256, 256],}

    print("######## Start #########")
    run_experiments(
        {
            args.experiment_name: {
                "run": "PPO",
                "env": "smac",
                "stop": {
                    #"training_iteration": args.num_iters,
                    "timesteps_total": args.num_timesteps,
                },
                "config": {
                    "num_workers": args.num_workers,
                    "observation_filter": "NoFilter",  # breaks the action mask
                    "vf_share_layers": True,  # no separate value model
                    "env_config": {
                        "map_name": args.map_name,
                        "communication_config": {
                            "communication": communication,
                            "message_simple": True,
                            "message_allies": False,
                            "message_enemies": False,
                            "message_range": args.range,
                        },
                        "voting_config": {
                            "voting": voting,
                            "voting_range": args.range,
                        },
                        "replay_config": {
                            "save_replay": args.save_replay,
                            "replay_steps": args.replay_steps,
                            "replay_dir": args.replay_dir,
                            "replay_prefix": None,
                        },
                        "masked_actions": args.masked_actions,
                    },
                    "model": {
                        "custom_model": "mask_model",
                    },
                    "train_batch_size": args.batch_size,
                    "sgd_minibatch_size": args.mini_batch,
                    #"rollout_fragment_length": 100,
                    "num_sgd_iter": args.num_epoch,
                    "lambda": 0.95,
                },
                "local_dir": args.local_dir,
            },
        }
    )
