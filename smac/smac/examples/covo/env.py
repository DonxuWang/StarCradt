from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from hashlib import new

import random
import numpy as np

from gym.spaces import Discrete, Box, MultiDiscrete, Tuple, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from smac.examples.rllib.env import RLlibStarCraft2Env


class MyStarCraft2Env(MultiAgentEnv):
    def __init__(self, **smac_args):

        self.onehot = True
        self.repeat_num = 1
        # Config for Communication
        if "communication_config" not in smac_args:
            self.communication_config = {}
        else:
            self.communication_config = smac_args.pop("communication_config")
        self.communication = self.communication_config.get("communication", True)
        self.all_share_message = self.communication_config.get("all_share_message", True)
        # communicate with simple message
        self.message_simple = self.communication_config.get("simple_communicate", True)
        self.message_range = self.communication_config.get("message_range", 2)
        # communicate with state(information)
        self.message_random = self.communication_config.get("message_random", False)
        self.message_allies = self.communication_config.get("message_allies", False)
        self.message_enemies = self.communication_config.get("message_enemies", False)
        self.message_last_actions = self.communication_config.get("message_last_actions", False)
        self.death_detection = True

        # Config for Voting
        if "voting_config" not in smac_args:
            self.voting_config = {}
        else:
            self.voting_config = smac_args.pop("voting_config")
        self.voting = self.voting_config.get("voting", False)
        self.voting_model = self.voting_config.get("voting_model", "Majorty")
        self.voting_type = self.voting_config.get("voting_type", "Discrete")
        self.voting_range = self.voting_config.get("voting_range", 2)

        self.comm_voting_range = 0

        # replay
        if "replay_config" not in smac_args:
            self.replay_config = {}
        else:
            self.replay_config = smac_args.pop("replay_config")
        self.save_replay = self.replay_config.get("save_replay", False)
        self.replay_steps = self.replay_config.get("replay_steps", 100)
        self.replay_dir = self.replay_config.get("replay_dir", None)
        self.replay_prefix = self.replay_config.get("replay_prefix", None)

        self.count = 0
        
        # if "masked_actions" not in smac_args:
        #     self.masked_actions = False
        # else:
        #     self.masked_actions = smac_args.pop("masked_actions", False)

        self.old_env = RLlibStarCraft2Env(**smac_args)
        # self.old_env.masked_actions = self.masked_actions
        self.old_env.masked_actions = True
        self.old_env._env.replay_dir = self.replay_dir
        self.old_env._env.replay_prefix = self.replay_prefix

        # set reward
        # self.old_env._env.reward_sparse = True
        # self.old_env._env.reward_death_value = 20
        # self.old_env._env.reward_win = 550
        # self.old_env._env.reward_scale = False

        self.observation_space = dict(
            {
                "obs": Box(-1, 1, shape=(self.old_env._env.get_obs_size(),)),
                # "action_mask": Box(
                #     0, 1, shape=(self.old_env._env.get_total_actions(),)
                # ),
            }
        )

        # voting
        if self.voting:
            # self.observation_space["action_mask"] = Box(
            #     0,
            #     1,
            #     shape=(
            #         self.old_env._env.get_total_actions() + self.voting_range,
            #     ),
            # )

            self.voting_space = {}
            if self.death_detection:
                self.comm_voting_range = self.voting_range + 1
            else:
                self.comm_voting_range = self.voting_range
            if self.voting_type == "Discrete":
                self.voting_space = self.observation_space[
                    "voting"
                ] = Discrete(self.comm_voting_range)

            self.observation_space = Dict(self.observation_space)
            self.action_space = Tuple(
                [
                    self.voting_space,
                    self.old_env.action_space,
                ]
            )

        # communication
        elif self.communication:
            # self.observation_space["action_mask"] = Box(
            #     0,
            #     1,
            #     shape=(
            #         self.old_env._env.get_total_actions() + self.message_range,
            #     ),
            # )

            if self.message_simple:
                self.message_enemies = (
                    self.message_allies
                ) = self.message_random = self.message_last_actions = False

            self.communication_space = {}

            n_allies, _ = self.old_env._env.get_obs_ally_feats_size()
            if self.death_detection:
                self.comm_voting_range = self.message_range + 1
            else:
                self.comm_voting_range = self.message_range

            if self.message_enemies:
                (
                    n_enemies,
                    n_enemy_feats,
                ) = self.old_env._env.get_obs_enemy_feats_size()
                if self.all_share_message:
                    self.observation_space["message_enemies"] = Box(
                        -1, 1, shape=(n_allies, n_enemies * n_enemy_feats)
                    )
                    self.communication_space["message_enemies"] = Box(
                        -1, 1, shape=(n_enemies * n_enemy_feats,)
                    )
                else:
                    # TODO: Some agents communicate, not all
                    self.observation_space["message_enemies"] = Discrete(2)

            if self.message_allies:
                self.window_size = self.old_env._env.window_size
                # TODO: not scale--The value range of box needs to be adjusted
                if self.all_share_message:
                    # rel_x, rel_y, rel_health
                    self.observation_space["message_allies"] = Box(
                        0, 1, shape=(n_allies, 3)
                    )
                    self.communication_space["message_allies"] = Box(
                        0, 1, shape=(3,)
                    )
                else:
                    # TODO: Some agents communicate, not all
                    self.observation_space["message_allies"] = Discrete(2)

            if self.message_last_actions:
                if self.all_share_message:
                    self.observation_space[
                        "message_last_actions"
                    ] = MultiDiscrete(
                        [self._env.get_total_actions()] * n_allies
                    )
                    self.observation_space[
                        "message_last_actions"
                    ] = self.old_env.action_space
                else:
                    # TODO: Some agents communicate, not all
                    self.observation_space["message_last_actions"] = Discrete(2)

            if self.message_simple:
                if self.all_share_message:
                    self.observation_space["message"] = MultiDiscrete(
                        [self.comm_voting_range] * n_allies
                    )
                else:
                    self.observation_space["message"] = Discrete(self.comm_voting_range)

            self.observation_space = Dict(self.observation_space)
            if self.message_simple:
                self.communication_space = Discrete(self.comm_voting_range)
            else:
                self.communication_space = Dict(self.communication_space)

            self.action_space = Tuple(
                [
                    self.communication_space,
                    self.old_env.action_space,
                ]
            )

        # without communication
        else:
            self.observation_space = Dict(self.observation_space)
            self.action_space = self.old_env.action_space


    def reset(self):
        obs_list, state_list = self.old_env._env.reset()

        return_obs = {}
        self.last_message = []
        for i, obs in enumerate(obs_list):
            # TODO: What should be the initial value of the message
            return_obs[i] = {
                # "action_mask": np.array(
                #     self.old_env._env.get_avail_agent_actions(i)
                # ),
                "obs": obs,
            }
            if self.voting:
                # return_obs[i]["action_mask"] = np.concatenate(
                #     (np.ones(self.voting_range), return_obs[i]["action_mask"])
                # )
                if self.voting_type == "Discrete":
                    return_obs[i]["voting"] = (
                        self.observation_space["voting"].sample() * 0
                    )
                    self.last_message.append([return_obs[i]["voting"]])
            elif self.communication:
                # return_obs[i]["action_mask"] = np.concatenate(
                #     (np.ones(self.message_range), return_obs[i]["action_mask"])
                # )
                message = []
                if self.message_enemies:
                    return_obs[i]["message_enemies"] = np.array(
                        [
                            ally[4:44]
                            for id, ally in enumerate(obs_list)
                            if id != i
                        ]
                    )
                    message.append(return_obs[i]["message_enemies"])
                if self.message_allies:
                    return_obs[i]["message_allies"] = self.observation_space["message_allies"].sample() * 0
                    message.append(return_obs[i]["message_allies"])
                if self.message_simple:
                    return_obs[i]["message"] = self.observation_space["message"].sample() * 0
                    message.append(return_obs[i]["message"])
                self.last_message.append(message)
                
        self._ready_agents = list(range(len(obs_list)))
        self.old_env._ready_agents = self._ready_agents

        return return_obs

    def step(self, action_dict):
        broad_rate = 500
        if self.voting:
            broad_VotingActions = False

            votings = []
            actions = []
            for i in self._ready_agents:
                if i not in action_dict:
                    raise ValueError(
                        "You must supply an action for agent: {}".format(i)
                    )
                if self.death_detection:
                    if self.old_env._env.get_unit_by_id(i).health > 0:
                        votings.append(action_dict[i][0])
                else:
                    votings.append(action_dict[i][0])
                actions.append(action_dict[i][1])

            if len(actions) != len(self._ready_agents):
                raise ValueError(
                    "Unexpected number of actions: {}".format(
                        action_dict,
                    )
                )

            new_action_dict = {}
            for key, value in action_dict.items():
                new_action_dict[key] = value[1]

            '''
            if broad_VotingActions:
                self.old_env._env._controller.chat("Voting Actions: ")
                broad_data = ""
                for k,v in enumerate(votings):
                    broad_data += f"Agent{k+1} vote：{v} ; "
                    if k%3 == 2:
                        self.old_env._env._controller.chat(broad_data)
                        broad_data = ""
                if broad_data != "": self.old_env._env._controller.chat(broad_data)
            '''

            return_obs, rews, dones, infos = self.old_env.step(new_action_dict)

            # voting_algorithm
            if self.voting_model == "Majorty":
                major = np.argmax(np.bincount(votings))

            if self.count%broad_rate == 0:
                if broad_VotingActions:
                    self.old_env._env._controller.chat("Voting Actions: " + " ".join([str(x) for x in votings]))
                #self.old_env._env._controller.chat("Output: %s" % major)

            for obs in return_obs.values():
                obs["voting"] = major
                obs["action_mask"] = np.concatenate(
                    (np.ones(self.voting_range), obs["action_mask"])
                )
            

        # with communication
        elif self.communication:
            broad_communications = False
            communications = []
            actions = []

            for i in self._ready_agents:
                if i not in action_dict:
                    raise ValueError(
                        "You must supply an action for agent: {}".format(i)
                    )
                if self.death_detection and self.old_env._env.get_unit_by_id(i).health < 0:
                    message = self.message_range
                else:
                    message = action_dict[i][0]
                communications.append(message)
                actions.append(action_dict[i][1])

            if len(actions) != len(self._ready_agents):
                raise ValueError(
                    "Unexpected number of actions: {}".format(
                        action_dict,
                    )
                )

            """
            if broad_communications:
                self.old_env._env._controller.chat("Communications:")
                broad_data = ""
                for k,v in enumerate(communications):
                    broad_data += f"Agent{k+1} chat：{v} ; "
                    if k%3 == 2:
                        self.old_env._env._controller.chat(broad_data)
                        broad_data = ""
                if broad_data != "": self.old_env._env._controller.chat(broad_data)

            """
            
            if self.count%broad_rate == 0 and broad_communications:
                self.old_env._env._controller.chat("Communications: " + " ".join([str(x) for x in communications]))

            new_action_dict = {}
            for key, value in action_dict.items():
                new_action_dict[key] = value[1]

            return_obs, rews, dones, infos = self.old_env.step(new_action_dict)

            for i, obs in enumerate(return_obs):
                return_obs[i]["action_mask"] = np.concatenate(
                    (np.ones(self.message_range), return_obs[i]["action_mask"])
                )

                if self.message_enemies:
                    return_obs[i]["message_enemies"] = np.array(
                        [
                            message["message_enemies"]
                            for id, message in enumerate(communications)
                            if id != i
                        ]
                    )
                if self.message_allies:
                    return_obs[i]["message_allies"] = np.array(
                        [
                            message["message_allies"]
                            for id, message in enumerate(communications)
                            if id != i
                        ]
                    )
                # TODO: Last Actions

                if self.message_simple:
                    if self.all_share_message:
                        return_obs[i]["message"] = np.array(
                            [
                                message
                                for id, message in enumerate(communications)
                                if id != i
                            ]
                        )
                    else:
                        return_obs[i]["message"] = np.array(communications[(i+1)%5])
                    self.last_message[i] = return_obs[i]["message"]
        else:
            return_obs, rews, dones, infos = self.old_env.step(action_dict)

        if self.save_replay:
            if self.count == self.replay_steps:
                self.count = 0
                self.old_env._env.save_replay()
            else:
                self.count += 1

        return return_obs, rews, dones, infos

    def get_obs(self):
        obs = []
        for i in range(self.old_env._env.n_agents):
            ob = self.old_env._env.get_obs_agent(i)
            if self.communication or self.voting:
                if self.onehot:
                    # messages_onehot = np.zeros((self.last_message.size, self.message_range+1))
                    # messages_onehot[np.arange(self.last_message.size), self.last_message] = 1
                    messages_onehot = np.eye(self.comm_voting_range)[self.last_message[i]]
                    messages_onehot = np.tile(messages_onehot, self.repeat_num)
                    ob_step = np.concatenate((ob, messages_onehot.flatten()))
                else:  
                    ob_step = np.concatenate((ob, np.tile(np.array(self.last_message[i]).flatten(), self.repeat_num)))
            else:
                ob_step = ob
            obs.append(ob_step)
        return obs

    def get_env_info(self):
        info = self.old_env._env.get_env_info()
        if self.communication or self.voting:
            info['n_comm_actions'] = self.comm_voting_range
            if self.onehot:
                info['message_range'] = self.comm_voting_range * self.repeat_num
            else:
                info['message_range'] = self.repeat_num
            if self.all_share_message:
                info['message_range'] *= (self.old_env._env.n_agents - 1)
        else:
            info['n_comm_actions'] = 0
            info['message_range'] = 0
       
        info["obs_shape"] += info['message_range']
        info["n_actions"] += info['n_comm_actions']

        return info

    def get_state(self):
        return self.old_env._env.get_state()

    def get_avail_agent_actions(self, agent_id):
        avail_action = []
        if self.communication or self.voting:
            avail_action = [0] * self.comm_voting_range
            avail_action.extend(self.old_env._env.get_avail_agent_actions(agent_id))
        else:
            avail_action = self.old_env._env.get_avail_agent_actions(agent_id)
        return avail_action

    def get_avail_agent_comm_actions(self, agent_id):
        avail_comm_voting = [1] * self.comm_voting_range
        if self.death_detection:
            if self.old_env._env.get_unit_by_id(agent_id).health > 0:
                avail_comm_voting[-1] = 0
            else:
                avail_comm_voting[:-1] = [0] * (self.comm_voting_range - 1)
        avail_comm_voting.extend([0] * self.old_env._env.n_actions)
        return avail_comm_voting

    def save_replay(self):
        return self.old_env._env.save_replay()

    def close(self):
        return self.old_env._env.close()
