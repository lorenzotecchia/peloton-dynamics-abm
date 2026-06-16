\section{Meeting Minutes and Specification}

\subsection{MVP preparation by the end of Wed lecture (discuss after the lecture)}
\begin{itemize}
    \item \zebracomment[Lorenzo]{[DONE] specifications (draft)}
    \item  \todo[Francesca]{speed}
    
    \item \todo[Emma, Tai]{game theory building / game theory reading}
    \item \todo[Francesca]{environments rule (how to implement and the presentation in game theory)}
    \item \zebracomment[Tai, Lorenzo]{[DONE] mesa in parallel: \update{ feasible. Native Mesa could be multiprocessing}}
    \item \zebracomment[tai]{[done] applying for computing resource: \update{: please try it}}
\end{itemize}

\subsection{Agent Specification}
This specification doesn't aim at modelling the same exact behaviour here written. This would impose the agent behaviour without any room for learning and emergence. Instead here I provide a full overview of peloton dynamics so that we can all have a full idea of what we would like to expect from simple, rule constraint agents (cyclists).

To win, cyclists have to finish before their competitors. In a usual race a breakaway (small group of cyclists) would form and take minutes and kilometers of advantage on the bigger group. Underdogs (from minor teams) usually try to anticipate and break away in these small groups to firstly increase media coverage of their team t-shirts (lol) and secondly to increase their chance of anticipating moves from the bigger group. Sometimes, in stage races, components of favourites teams would go in the breakaway to gain a satellite rider in order to for the favourite rider to have a lead-out for the finish. In general the team of the favourite rider would set the pace in the peloton such to control the time advantage of the breakaway. Each team has a designated captain that is usually the most capable rider to win the race. For its captain their team mates provide shelter from wind (drafting) and favourable positioning without any competition. To catch the breakaway one or multiple teams that contain the best contenders for the final win, will start increasing the pace in the peloton to catch the breakaway. Regardless of catching or not: the bigger group will fight for positioning with 2/3 kilometres to go. At this point of the race, speed in the bigger group can reach values of even 80km/h. For this reason positioning is crucial. Other factors come into play: like turns in proximity of the finish line; just because taking the curve in a favorable line will mean spending less energy to keep your position. 

This is how an ideal lead-out works. Teams would position themself such that the captain is the last in the line of drafting. The second-to-last rider would be the second strongest rider in the team, the one with comparable speed of the captain. Captains would want to stick together with their team mates for the final sprint usually because team mates know their captain's energy expenditure capabilities; having trained with similar situations during their training, less stress factors come into play. Energy expenditure here is also crucial along with positioning since, would be very costly and risky to jump on someone else wheel that is not one of your team mate'. Staying on someone else's wheel that is not your team mate could also mean spending more energy that you can physically sustain to also finish with a sprint yourself in the last 300 meters from the finish line. Sprinters need, for the final push to the line, to balance the need to anticipate moves from other riders and not spend too much energy (we can than see this behaviour happening at multiple stages of the race and from almost any rider inside the peloton). 

It's not important for the simulation, but it's important to note that teams have radio-communications between team mates and their reference car that knows time gaps, visual clues and riders state (like velocity, energy expenditure and so on).

It's also clear then that each rider has a clear view of their energy expenditure, velocity, position with respect to the bigger group and the their team. 

\zebracomment[Emma]{Thanks for the nice details Lorenzo :) I understand it much better now. As for the model, I think it would help if we had a clear overview of what components of this we attempt to simulate. As Francesca said, I think it's best if we keep the model as simple as possible, and see what kind of behaviour we can already achieve with this. In my opinion, dynamics that might be too complicated are: 
\begin{itemize}
    % \item having group dynamics depend on a favourite rider;
    \item having a particular order in riders depending on skill;
    \item having riders compete for positions in the peloton;
    \item having the peloton anticipate or respond to a breakaway;
    \item having the peloton anticipate or respond to defectors (?). 
\end{itemize}
What I think is doable: 
\begin{itemize}
    \item let single riders or single teams have a probability of breaking away;
    \item let riders within a team have a higher probability of cooperation;
    \item have a distribution of skill/endurance per team. 
\end{itemize}
Lmk if you guys have any other ideas, we could discuss on Wednesday.}

\zebracomment[Francesca]{Hello! If I understood correctly, we aim to find the probability of defections, breaking away from the peloton and cooperation that the population of cyclists (citizens of Cycle-City) learn through evolutionary game theory. Is that correct? If they form plausible configurations, it's a gift that we happily accept. \\ \\
So, I agree to have probability of cooperation, and if not cooperating they can either pass their turn or breakaway, with certain probabilities that should depend imo on fatigue/energy and distance from finish line/ distance from the first cyclists. Also, in \textit{Cooperation in bike racing — When to work together and when to go it alone}, the probability of following a breakaway depends on the team. \\
\\
so I would say:
\begin{itemize}
    \item probability of cooperation (will change as they learn)
    \item probability of breaking away (will change as they learn)
    \item probability of following a breakaway (might depend on the teams in the breakaway) (will change as they learn)
    \item power over weight during a certain time interval (I discussed this with Lorenzo, who can explain it way better than me)
\end{itemize}}

\subsection{can be done after MVP}
\begin{itemize}
    \item data collection and separation
    \item see if and how increasing difficulty of the race (increasing power needed to move, like in windy situations) changes the cooperation distribution
\end{itemize}

\subsection{Still open question and TBD}

\begin{itemize}
    \item lattice or spatial continuously
\end{itemize}


\subsection{Notes by Emma on game theory in competitive cycling}
\begin{itemize}
    \item In \emph{Cooperation in bike racing — When to work together and when to go it alone}, the authors do not define any utility matrix, but rather assume that a strategy consists of a probability of cooperation/defection. As a result, the behavior is purely random, but they evaluate strategy success by picking one player in a simulation and varying their probabilities to see which one would have lead to the best outcome. I do not know whether this classifies as bounded rationality, as there is essentially no rationality.
    \item In \emph{Strategic Behavior in Road Cycling Competitions (2022)} (\url{https://link.springer.com/chapter/10.1007/978-3-031-11258-4_10}), the authors discuss qualitatively what kind of strategy decisions are made in cycling competitions, and mention a few examples of game theory in there. There something similar to a ``game of matching pennies'' when riders need to time a breakaway; too early will leave them exhausted, too late will increase the probability that the peloton will prevent it. In between, their success depends on the probability that the peloton is off guard, and so the payoff is a combination of the timing of the opponent (peloton) and the rider. Furthermore, they compare the strategy between the riders in a breakaway (cooperate with each other or defect) as a ``centipede game'', which is an iterated game where, each time both players cooperate, the payoff for defecting the next round increases. There are more examples, but the paper is very qualitative, and the game theoretic example are in my opinion not applicable to ABM.
    \item From my scan of literature, I cannot find a developed model that is both agent-based, concerns bicycle racing and contains game theory in the form of payoff matrices. There are studies where agents evolve their strategy of cooperating/defecting, but in a stationary lattice (\url{https://www.nature.com/articles/359826a0.pdf?utm_source=wiley&getft_integrator=wiley}, \url{https://research.ebsco.com/c/bkudj5/viewer/pdf/jb5y4r6yav}); there are studies that mainly discuss qualitatively that there is game theory present in racing (\url{https://firstmonday.org/ojs/index.php/fm/article/view/727/636}, \url{https://link.springer.com/chapter/10.1007/978-3-031-11258-4_10}); there are studies that incorporate games as well as spatial motion, but not in the form of (bicycle) racing (\url{https://www.jasss.org/12/1/8.html}, \url{https://dl.acm.org/doi/pdf/10.1145/545056.545076}); and there is of course \emph{Cooperation in bike racing...} that does not use game theory in the classical sense (with explicit utility), but does consider strategies. What I haven't lookup up enough yet are game theory ABM models in other or more general racing phenomena. There are some papers about for example strategy optimization a race between autonomous entities (cars) (\url{https://ieeexplore.ieee.org/abstract/document/10598237}, \\\url{Game-Theoretic Planning for Self-Driving Cars in Multivehicle Competitive Scenarios}). \\ 
    \textbf{Concerns}: the aim of other-than-bicycle-races is more to find the optimal line of trajectory based on what all agents expect the others to do, giving a much larger focus on kinematics and spatial position (avoiding collisions), and there is little to no cooperation involved which is crucial in bicycle race dynamics. Paper with game theory in race between two cars with limited actions available to them and payoff matrices: \url{https://ieeexplore.ieee.org/abstract/document/8643396}.
\end{itemize}

\subsection{Notes by Francesca on rules/probabilities}
\begin{itemize}
    \item \textbf{a possible way to describe probabilities with parameters:} $$
    \sigma_i(\alpha_i + \beta_i d_{e}+\gamma_i E_{left})$$
    where $d_{e}$ is the distance from the finish line, $E_{left}$ is the "energy"/stamina left and $\alpha_i, \beta_i, \gamma_i$ are coefficients that player $i$ has to learn. \\ \textbf{Note regarding sensitivity analysis:} from what I understand now, sensitivity analysis regards the parameters that characterize the model, not the output. And since $\alpha, \beta$ and $\gamma$ would be our outputs, there is no need to run 512 simulation for each of them. Instead, the parameters on which we should run the simulations are for example the starting energy or max power and the coefficient that express how much power is lost depending on velocity (I think... tell me if you agree or if I am making mistakes!)
    \item \textbf{how old papers modelled energy consumption:} This is meant to summarize what has been done previously. Since in my opinion is not perfectly clear what they did, I am not sure this is exactly how they modelled it, but anyway.
    \\
    Cyclist are in the same group if they are less than $3m$ distant. If a rider is behind someone, drafting occurs and the air resistance reduces of a factor $$
    CF_{draft}=0.62-0.0104d_w+0.0452d_w^2,$$
    where $d_w$ is the wheel-to-wheel distance ($m$) between the rider and the one in front of him.
\end{itemize}

## Running the MVP simulation

```bash
uv sync
uv run solara run run_app.py   # opens the interactive peloton visualization
uv run pytest                  # run the test suite
```
