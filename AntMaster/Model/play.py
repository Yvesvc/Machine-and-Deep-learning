import sys
sys.path.append("D:\\yves\\source\\repos\\RL\\AntMasters")
from os import listdir
from Environment.AntMasterEnv_v4 import AntMasterEnv_v4
from stable_baselines3 import PPO
from Model.constants import Constants

def play_human(env, full_visibility = True):
    env.reset()
    env.render(full_visibility)
    # 0: #right
    # 1: #down
    # 2: #left
    # 3: #up
    done = False
    while not done:
        action = input("actie")
        obs, reward, done, truncation, info = env.step(action)
        env.render(full_visibility)

def play_AI(env, model_name = None, full_visibility = True):

    model_path = Constants.path_model

    if model_name == None:
        file_list = listdir(model_path)
        files_cnt = len(file_list)
        model_name = file_list[files_cnt - 1] 

    file_path = model_path + "\\" + model_name

    model = PPO.load(file_path, env=env)
    for ep in range(10):
        env.reset()
        done= False
        while(not done):
            obs = env.get_observation()
            chosen = model.predict(obs)[0]
            obs, reward, done, truncation, info = env.step(chosen) 
            env.render(full_visibility)



env = AntMasterEnv_v4(render_mode="human")

play_human(env, full_visibility=False)
play_AI(env, full_visibility=True)


    


