import sys
sys.path.append("D:\\yves\\source\\repos\\RL\\AntMasters")
from stable_baselines3 import PPO
from Environment.AntMasterEnv_v4 import AntMasterEnv_v4
from Model.constants import Constants
import datetime

models_dir = Constants.path_model
log_dir = Constants.path_log
TIMESTEPS = 10000
EPISODES = 200


env =  AntMasterEnv_v4()

model = PPO("CnnPolicy", env, verbose=2, tensorboard_log=log_dir)

for ep in range(1,EPISODES + 1):
    print(str(ep))
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model_name = datetime.datetime.now().strftime("%Y-%m-%d--%H_%M_%S%z")
    if (ep % 10 == 0):
        model.save(f"{models_dir}/{model_name}")

