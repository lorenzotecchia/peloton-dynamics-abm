"Simulation script for the peloton dynamics agent-based model."

from model import CyclingRace
import matplotlib.pyplot as plt

model = CyclingRace(
    n_riders=50,
    finish_x=5000
)

positions = []

for step in range(500):

    model.step()
    x = [
        rider.pos[0]
        for rider in model.agents
    ]
    positions.append(x)

leader = max(
    rider.pos[0]
    for rider in model.agents
)

print("Leader distance:", leader)

data = model.datacollector.get_model_vars_dataframe()

# plt.figure(figsize=(8, 4))
# plt.plot(data["MeanEnergy"])
# plt.title("Mean Rider Energy")
# plt.xlabel("Time")
# plt.ylabel("Energy")
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.plot(data["PelotonSize"])
# plt.title("Largest Group Size")
# plt.xlabel("Time")
# plt.ylabel("Riders")
# plt.show()

plt.figure(figsize=(8, 4))
for i in range(10):
    plt.plot(
        [pos[i] for pos in positions],
        label=f"Rider {i}"
    )
plt.title("Rider Positions")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.show()