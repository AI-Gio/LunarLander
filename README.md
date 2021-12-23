# LunarLander

**The lunar lander problem using Deep Q-Learning under OpenAI Gym’s LunarLander-v2 Environment.** 

Lunar Lander problem is the task to control the fire orientation engine to help the lander land in the landing pad. LunarLander-v2 is a simplied version of the problem under OpenAI Gym environment[1], which requires the agent to move in 8-dimensional state space,with six continuous state variables and two discrete ones, using 4 actions to land on pad: do nothing, fire the left orientation engine, fire the main engine, fire the right orientation engine. The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame. 

Used source: https://medium.datadriveninvestor.com/training-the-lunar-lander-agent-with-deep-q-learning-and-its-variants-2f7ba63e822c

Files:
````
- src/
  - main.py - The main file to be run
  - agent.py - The logic centre
  - memory.py - The memory class
  - EpsilonPolicy.py - The policy of the agent
  - function_approx.py - the neural network
  - transition.py - The dataclass

- doc/
  - Plot of rewards image
  - Neural Network architecture pdf
````
