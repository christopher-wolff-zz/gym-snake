import time

import gym
import gym_snake

env = gym.make('Snake-v0')

for i in range(100):
    env.reset()
    for t in range(1000):
        env.render()
        observation, reward, done, _ = env.step(env.action_space.sample())
        if reward == 50:
            env.render()
            time.sleep(5)
        if done:
            print('episode {} finished after {} timesteps'.format(i, t))
            break
