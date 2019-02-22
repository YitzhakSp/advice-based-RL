# Bootstrap DQN


This repo contains our implementation of a Bootstrapped DQN with options to add a Randomized Prior, 
Dueling, and DoubleDQN in ALE games. 

[Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)

[Randomized Prior Functions for Deep Reinforcement Learning](https://arxiv.org/abs/1806.03335)

# Some results from Freeway

![alt text](figs/small_ATARI_step0001508988_r0024_testcolor.gif?raw=true "Freeway Agent - Bootstrap with Prior")

![alt text](figs/freeway_9heads_prior_episode_reward.png?raw=true "Freeway Agent - Bootstrap with Prior")

# Dependencies

atari-py installed from https://github.com/kastnerkyle/atari-py  
torch='1.0.1.post2'  
cv2='4.0.0'  


# References

We referenced several execellent examples/blogposts to build this codebase: 

[Discussion and debugging w/ Kyle Kaster](https://gist.github.com/kastnerkyle/a4498fdf431a3a6d551bcc30cd9a35a0)

[Fabio M. Graetz's DQN](https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb)

[hengyuan-hu's Rainbow](https://github.com/hengyuan-hu/rainbow)

[Dopamine's baseline](https://github.com/google/dopamine)
