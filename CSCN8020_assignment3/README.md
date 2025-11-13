#  CSCN8020 – Assignment 3  
## Deep Q-Learning (DQN) on Atari Pong

The second component of this repository implements a **Deep Q-Network (DQN)** trained to play **PongDeterministic-v4** using raw pixel input.

###  Features Implemented
- Raw frame preprocessing (crop → grayscale → resize)
- Frame stacking (4 frames)
- Convolutional neural network (CNN)
- Replay buffer
- Target network
- ε-decay exploration
- Adam optimizer + gradient clipping
- Three controlled hyperparameter experiments

###  Experiments (60 Episodes Each)
1. **Baseline**  
   - Batch size = 8  
   - Target update = 10 episodes  

2. **Batch Size Comparison**  
   - Batch size 8 vs 16  
   - Target update = 10  

3. **Target Network Frequency**  
   - Batch size = 8  
   - Target update = 3 vs 10  

###  Outputs
- Episode score plots  
- Moving-average reward plots  
- Comparative plots  
- CSV logs in `/results`  
- Full PDF report with analysis  

###  Folder Contents
- `dqn_pong.py` — Training script  
- `assignment3_utils.py` — Preprocessing utilities  
- `plots.ipynb` — Visualization and comparison notebook  
- `results/` — CSV logs for each experiment  
- `requirements.txt` — Dependency list  
- `README.md` — Assignment-level documentation  

---

 ### Relationship Between Workshop & Assignment

| MAB Concept | DQN Application |
|-------------|-----------------|
| ε-greedy exploration | Used for Pong action selection |
| Decaying ε | Same mechanism but in pixel-based RL |
| Reward drift handling | Similar to moving-target problem in deep RL |
| Constant α step-size | Similar effect as mini-batch size |
| Exploration–exploitation dilemma | Core motivation for deep RL algorithms |

The workshop provides intuition about exploration that directly supports the DQN assignment.


---

