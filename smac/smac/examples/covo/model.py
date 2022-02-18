from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" Custom RLlib model

For version rllab > 1.8.0
This is used to handle the variable-length StarCraft action space.
"""

import tensorflow as tf

from gym.spaces import Box
from ray.rllib.models.tf import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class MaskedActionsModel(TFModelV2):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name
    ):
        super(MaskedActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self._registered = False
        

    def forward(self, input_dict, state, seq_lens):
        # Extract the action mask tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        last_layer = input_dict["obs"]["obs"]
        hiddens = [256, 256]

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            for i, size in enumerate(hiddens):
                last_layer = tf.layers.dense(
                    last_layer,
                    size,
                    kernel_initializer=normc_initializer(1.0),
                    activation=tf.nn.tanh,
                    name="fc{}".format(i),
                )
            action_logits = tf.layers.dense(
                last_layer,
                self.num_outputs,
                kernel_initializer=normc_initializer(0.01),
                activation=None,
                name="fc_out",
            )
            self._value_out = tf.layers.dense(
                last_layer,
                1,
                kernel_initializer=normc_initializer(1.0),
                activation=None,
                name="vf",
            )

        if not self._registered:
            # Register already auto-detected variables (from the wrapping
            # Model, e.g. DQNTFModel).
            self.register_variables(self.variables())
            # Then register everything we added to the graph in this `forward`
            # call.
            self.register_variables(
                tf1.get_collection(
                    tf1.GraphKeys.TRAINABLE_VARIABLES, scope=".+/model/.+"
                )
            )
            self._registered = True
        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, []

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
