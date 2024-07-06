import matplotlib.pyplot as plt
from DDPG import DDPGagent
import torch
from utils import *

env = NormalizedEnv(gym.make("Pendulum-v0"))

agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []
max_episode = 500
max_step = 5000


for episode in range(max_episode):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(max_step):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            print(f"Episode: {episode},"
                  f" Reward: {np.round(episode_reward, 2)},"
                  f" Average Reward: {np.mean(rewards[-10:])}")
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

    # Save models every 10 episodes
    if episode % 10 == 0:
        save_models_for_training(agent, f"experiments/ddpg_agent_{episode}.pth")

plt.plot(rewards)
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# Save final models for testing
save_models_for_testing(agent, "experiments/ddpg_agent_final.pth")
