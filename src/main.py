import gym
import agent, transition, memory, function_approx
env = gym.make('LunarLander-v2')

agent = agent.Agent()

# TODO: ↓ maak hier een functie van

for i_episode in range(20):
    current_state = env.reset()
    for step in range(1000):
        env.render()


        print(current_state)
        # TODO:  select_action
        action = agent.choose_action(current_state, 0.1) #env.action_space.sample()
        next_state, reward, done, info = env.step(action) # next_state is observation

        # Save transition in memory of agent
        t = transition.Transition(current_state, action, reward, next_state, done)
        agent.memory.record(t)

        current_state = next_state

        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break
env.close()