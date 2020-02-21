# Search and Destroy

Problem: We have a landscape represented by a map of cells. The cells can be of four terrain types - 'flat', 'hilly', 'forested', ' complex maze of caves and tunnels'. There is a target hidden in one of the cells. Initially, the target is equally likely to be anywhere in the landscape. There is a chance that even if a target is there, you may not find it, depending on the terrain type:
1. P(Target not found in cell|Target in cell) = 0.1; if cell is flat
2. P(Target not found in cell|Target in cell) = 0.3; if cell is hilly
3. P(Target not found in cell|Target in cell) = 0.7; if cell is forested
4. P(Target not found in cell|Target in cell) = 0.9; if cell is a maze of cave

I modeled this information using Bayesian Networks to update knowledge/belief about a system probabilistically, and use this belief
state to efficiently direct future action. According to the current state of the belief of the agent, the next cell to search is chosen according to two rules:  
Rule 1:  At any time, search the cell with the highest probability of containing the target.   
Rule 2:  At any time, search the cell with the highest probability of finding the target.   
According to these rules, the belief of the agent is updated each time. 
