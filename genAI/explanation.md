## Explanation of the files in this folder

This folder contains content that was generated in a conversation between Emma and ChatGPT on 11-06-2026 about the model possibilities. I provided ChatGPT with the two main ideas we had for our project: either a discrete lattice with utility driven motion of agents between lattice sites, or a continuous space where agent strategy comes about as a probability of cooperation, and utility is only awarded after a full race.

Much of the feedback ChatGPT gave were issues and ideas we had already discussed within the group (velocity restrictions in a discrete lattice, possibility of partial cooperation rather than C/D, concerns about jamming in the discrete lattice). Some additional useful suggestions were:

1. strong recommendation to use a continuous space, as this more naturally relates to realistic racing dynamics;
2. in this continuous space, to use **networkx** for detecting which cyclists belong to one group;
3. treat the peloton as a *public goods game* (https://en.wikipedia.org/wiki/Public_goods_game), which is a game where all contestants privately decide how much of their tokens they put into a common pot (i.e., how much of their internal energy a rider is prepared to spend in the peloton, and how much they would want to depend on drafting). Then the total velocity of the peloton is set by the sum of contributions of all riders. This can be easily implemented as a binary choice (to pull or not to pull, ...), or perhaps in a more advanced version as a fraction.

Point 3 is not implemented yet but we could use it to inspire further investigations. The main components of the model as of yet are stated in the following.

### Contained in the implementation
- Agents are entities in continuous space;
- Their predifined attributes are: ID, cooperation probability
- Their state variables that are updated throughout the race are: position, velocity, energy, group ID (which group they are a part of), and is_cooperating (True/False)
- At each step, the group ID per agent is established through networkx scanning of all riders and their distanced to ther riders.
- A time step per agent is then structured as follows:
  - the neighborhood (of 3m) is scanned for other agents to establish drafting contributions, the neighbors in front of the agent determine the drafting factor (percentage of cost an agent truly pays);
  - according to their prob. of cooperation, their state is updated to either cooperative or not;
  - based on their state, the drafting factor and their internal energy, their speed is updated through some magic mathematics -- **we will definitely need to revise/replace this part**;
  - the agent's position is updated according to this updated speed in the direction of the finish line, with some additional lateral noise. NB: both the speed and position are clipped to be within physical bounds and the bounds of the domain.


### Not contained in implementation
- a realistic drafting factor: the drafting effect is independent of positioning/angle of agents in front of the agent, even though agents that are almost next to you should have a much smaller effect;
- a drafting / energy / cooperation -to-acceleration update that makes sense: in this implementation, a cooperating rider is associated with a 100% effort, while a non-cooperating one is associated with a 30% effort, which seems like a number pulled out of a magic hat. The energy cost is then calculated as effort * base_cost * (1 - draft), which is kind of justifiable, but the following acceleration update of acc = (0.3 * effort - 0.01 * (100 - self.energy)) comes out of nowhere.
- the existence of teams and different strategies within the teams as opposed to with strangers;
- utility at the end of the race or learning across races.