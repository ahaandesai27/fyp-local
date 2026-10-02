import random
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================

P_MAX = 4.0  # Maximum acceptable load (kW)

APPLIANCES = {
    "Refrigerator": {
        "power": 0.2,
        "cost": 100,
        "flexible": False,
    },
    "AC": {
        "power": 1.5,
        "cost": 5,
        "flexible": True,
    },
    "Water Heater": {
        "power": 2.0,
        "cost": 2,
        "flexible": True,
    },
    "TV": {
        "power": 0.15,
        "cost": 1,
        "flexible": False,
    },
    "Washing Machine": {
        "power": 0.8,
        "cost": 1,
        "flexible": True,
    },
}


# ============================================================
# GENETIC ALGORITHM
# ============================================================

def genetic_controller(
    predicted_load,
    appliances,
    max_load,
    population_size=30,
    generations=50,
    mutation_rate=0.1,
):
    """
    Genetic Algorithm controller.

    Each chromosome represents which controllable appliances
    should be turned off.

    1 = turn appliance OFF
    0 = leave appliance ON
    """

    controllable = [
        (name, data)
        for name, data in appliances.items()
        if data["flexible"] and data["on"]
    ]

    if not controllable:
        return []

    n = len(controllable)

    # --------------------------------------------------------
    # Fitness function
    # --------------------------------------------------------

    def fitness(chromosome):

        load_reduction = sum(
            data["power"]
            for gene, (_, data) in zip(chromosome, controllable)
            if gene == 1
        )

        remaining_load = max(
            0,
            predicted_load - load_reduction
        )

        # Amount by which we still exceed the limit
        excess = max(
            0,
            remaining_load - max_load
        )

        # User discomfort
        discomfort = sum(
            data["cost"]
            for gene, (_, data) in zip(chromosome, controllable)
            if gene == 1
        )

        # Number of appliances switched
        actions = sum(chromosome)

        # Large penalty for violating P_MAX
        penalty = excess * 1000

        return penalty + discomfort + 0.1 * actions

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = [
        [random.randint(0, 1) for _ in range(n)]
        for _ in range(population_size)
    ]

    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    for _ in range(generations):

        population.sort(key=fitness)

        # Keep best individuals
        elite_count = max(2, population_size // 5)
        new_population = population[:elite_count]

        while len(new_population) < population_size:

            # Tournament selection
            parent1 = min(
                random.sample(population, 3),
                key=fitness
            )

            parent2 = min(
                random.sample(population, 3),
                key=fitness
            )

            # Crossover
            if n > 1:
                point = random.randint(1, n - 1)

                child = (
                    parent1[:point]
                    + parent2[point:]
                )
            else:
                child = parent1.copy()

            # Mutation
            for i in range(n):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]

            new_population.append(child)

        population = new_population

    # Best solution
    best = min(population, key=fitness)

    return [
        name
        for gene, (name, _) in zip(best, controllable)
        if gene == 1
    ]


# ============================================================
# SIMULATION
# ============================================================

historical_loads = []
predicted_loads = []
actual_loads = []
controlled_loads = []
turn_off_history = []

# Generate household load
for t in range(51):
    current_load = random.uniform(1, 6)
    historical_loads.append(current_load)


for t in range(50):

    # Current actual load
    current_load = historical_loads[t]

    # Simple noisy forecast
    noise = random.gauss(
        0,
        0.1 * current_load
    )

    predicted_load = max(
        0,
        current_load + noise
    )

    # Reset appliance states
    appliances = {
        name: {
            **data,
            "on": random.random() > 0.3
        }
        for name, data in APPLIANCES.items()
    }

    # --------------------------------------------------------
    # GENETIC ALGORITHM CONTROLLER
    # --------------------------------------------------------

    turned_off = genetic_controller(
        predicted_load,
        appliances,
        P_MAX
    )

    # Calculate load reduction
    load_reduction = sum(
        appliances[name]["power"]
        for name in turned_off
    )

    controlled_load = max(
        0,
        current_load - load_reduction
    )

    # Store results
    actual_loads.append(current_load)
    predicted_loads.append(predicted_load)
    controlled_loads.append(controlled_load)
    turn_off_history.append(turned_off)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("GENETIC ALGORITHM CONTROL SUMMARY")
print("=" * 50)

total_actions = 0

for name in APPLIANCES:

    count = sum(
        name in actions
        for actions in turn_off_history
    )

    if count > 0:
        print(
            f"{name:20s}: turned off {count} times"
        )

        total_actions += count

print("-" * 50)

print(
    f"Total control actions:   {total_actions}"
)

print(
    f"Average actual load:     "
    f"{sum(actual_loads) / len(actual_loads):.2f} kW"
)

print(
    f"Average controlled load:  "
    f"{sum(controlled_loads) / len(controlled_loads):.2f} kW"
)

print(
    f"Peak actual load:         "
    f"{max(actual_loads):.2f} kW"
)

print(
    f"Peak controlled load:     "
    f"{max(controlled_loads):.2f} kW"
)


# ============================================================
# PLOT
# ============================================================

os.makedirs(
    "results/controller",
    exist_ok=True
)

plt.figure(figsize=(12, 6))


plt.plot(
    controlled_loads,
    label="GA Controlled Load"
)

plt.axhline(
    P_MAX,
    linestyle="--",
    label="Maximum Load"
)

plt.xlabel("Time step")
plt.ylabel("Power (kW)")
plt.title("Genetic Algorithm Energy Control")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/controller/ga_control_simulation.png",
    dpi=300
)

plt.close()

print("\nPlot saved to:")
print("results/controller/ga_control_simulation.png")