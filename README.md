This repository contains my best rule-based agent, 
which achieved 6th place in the 
[Kore 2022](https://www.kaggle.com/competitions/kore-2022) 
Kaggle Competition. The code is built on top of my
[kore-beta-bot](https://github.com/w9PcJLyb/kore-beta-bot), 
which I open-sourced at the beginning of the competition.

Usage:

```python
from src.main import agent
from kaggle_environments import make

env = make("kore_fleets")
env.run([agent, agent])
env.render(mode="ipython", width=1000, height=800)
```
