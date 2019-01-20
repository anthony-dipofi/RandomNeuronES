# RandomNeuronES
Evolutionary strategies for neural networks with random weights

Creates a neural network with random weights, and evolves only the final layer to to reduce training time.
The evolutionary stratgey used is fairly 'naive'. After each generation the top performing agent is retained and mutated with random noise of a fixed variance. It is likely that a more sophisticated stratagy would be more effective as well.

Training is typically quite quick due to the relatively small number of parameters being adapted. Using cuda is considerably faster but also uses considerably more memory.

Because the network is intialized with random wieghts, it is possible that the features the produce could be insufficient for solving the environment, however, in practice it seems for fairly simple environments this happens pretty infrequently. However the behaviours it learns are often quite strange, and basically do not generalize to other environments at all outside of the environment the agent was trained on.

Video showing a trained agent here: https://youtu.be/LB6OMTc2dFM

Uses:
Numpy
Scipy
openAI retro
openAI baselines
sonic_util.py (https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py)
pytorch
