import numpy as np
import pandas as pd

csv_data = pd.read_csv(
    "/home/dongxu/Desktop/multi-agenten-reinforcement-learning-mit-consensus-algorithmen/results/StarCraft2_PPO/Voting/8m/PPO_200Batch_200iters_Discrete2/PPO_smac_b734a_00000_0_2021-09-24_00-29-29/progress.csv"
)

reward_mean = csv_data["episode_reward_mean"]
top5 = reward_mean.nlargest(5).mean()
top10 = reward_mean.nlargest(10).mean()

print(reward_mean.nlargest(1).mean())
print(top5)
print(top10)
print(csv_data["episode_len_mean"][199])
