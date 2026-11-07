import gymnasium as gym
import stable_baselines3
import torch

print("gymnasium", gym.__version__)
print("stable-baselines3", stable_baselines3.__version__)
print("torch", torch.__version__)

env = gym.make("LunarLander-v3")
obs, info = env.reset(seed=0)
print("osservazione iniziale:", obs)
print("spazio osservazioni:", env.observation_space)
print("spazio azioni:", env.action_space)

totale = 0.0
for passo in range(1000):
    azione = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(azione)
    totale += reward
    if terminated or truncated:
        print(f"episodio finito al passo {passo + 1}, "
              f"reward totale {totale:.1f}, "
              f"terminated={terminated}, truncated={truncated}")
        break
env.close()