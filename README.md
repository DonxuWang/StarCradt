# StarCraft

Pytorch implementations of the multi-agent reinforcement learning algorithms, including 
[IQL](https://arxiv.org/abs/1511.08779),
[QMIX](https://arxiv.org/abs/1803.11485), [VDN](https://arxiv.org/abs/1706.05296), 
[COMA](https://arxiv.org/abs/1705.08926), [QTRAN](https://arxiv.org/abs/1905.05408)(both **QTRAN-base** and **QTRAN-alt**),
[MAVEN](https://arxiv.org/abs/1910.07483), [CommNet](https://arxiv.org/abs/1605.07736), 
[DyMA-CL](https://arxiv.org/abs/1909.02790?context=cs.MA), and [G2ANet](https://arxiv.org/abs/1911.10715), 

## Installing StarCraft II

SMAC is based on the full game of StarCraft II (versions >= 3.16.1). To install the game, follow the commands bellow.

### Linux

Please use the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads) to download the Linux version of StarCraft II. By default, the game is expected to be in `~/StarCraftII/` directory. This can be changed by setting the environment variable `SC2PATH`.

### MacOS/Windows

Please install StarCraft II from [Battle.net](https://battle.net). The free [Starter Edition](http://battle.net/sc2/en/legacy-of-the-void/) also works. PySC2 will find the latest binary should you use the default install location. Otherwise, similar to the Linux version, you would need to set the `SC2PATH` environment variable with the correct location of the game.

## SMAC maps

SMAC is composed of many combat scenarios with pre-configured maps. Before SMAC can be used, these maps need to be downloaded into the `Maps` directory of StarCraft II.

Download the [SMAC Maps](https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip) and extract them to your `$SC2PATH/Maps` directory. If you installed SMAC via git, simply copy the `SMAC_Maps` directory from `smac/env/starcraft2/maps/` into `$SC2PATH/Maps` directory.

### List the maps

To see the list of SMAC maps, together with the number of ally and enemy units and episode limit, run:

```shell
python -m smac.bin.map_list 
```

### Creating new maps

Users can extend SMAC by adding new maps/scenarios. To this end, one needs to:

- Design a new map/scenario using StarCraft II Editor:
  - Please take a close look at the existing maps to understand the basics that we use (e.g. Triggers, Units, etc),
  - We make use of special RL units which never automatically start attacking the enemy. [Here](https://docs.google.com/document/d/1BfAM_AtZWBRhUiOBcMkb_uK4DAZW3CpvO79-vnEOKxA/edit?usp=sharing) is the step-by-step guide on how to create new RL units based on existing SC2 units,
- Add the map information in [smac_maps.py](https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/maps/smac_maps.py),
- The newly designed RL units have new ids which need to be handled in [starcraft2.py](https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/starcraft2.py). Specifically, for heterogenious maps containing more than one unit types, one needs to manually set the unit ids in the `_init_ally_unit_types()` function.

## Corresponding Papers
- [IQL: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736)
- [From Few to More: Large-scale Dynamic Multiagent Curriculum Learning](https://arxiv.org/abs/1909.02790?context=cs.MA)
- [Multi-Agent Game Abstraction via Graph Attention Neural Network](https://arxiv.org/abs/1911.10715)
- [MAVEN: Multi-Agent Variational Exploration](https://arxiv.org/abs/1910.07483)

## Acknowledgement

+ [SMAC](https://github.com/oxwhirl/smac)
+ [pymarl](https://github.com/oxwhirl/pymarl)

## Quick Start

```shell
$ python main.py --map=3m --alg=qmix
```

