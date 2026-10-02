import random
import matplotlib.pyplot as plt

def noise(value, noise_level=0.1):
    """
    Add Gaussian noise to a value.

    noise_level = 0.1 means roughly ±10% noise.
    """
    noise = random.gauss(0, noise_level * value)
    return max(0, value + noise)

# ============================================================
# CONFIGURATION
# ============================================================

P_MAX = 4.0  # Maximum acceptable load (kW)


# update true and false based on if person is nearby 
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
        "flexible": True,
    },
    "Washing Machine": {
        "power": 0.8,
        "cost": 1,
        "flexible": True,
    },
}


# ============================================================
# CONTROLLER
# ============================================================

def controller(predicted_load, appliances, max_load):
    """
    Decide which appliances to turn off/defer.

    Input:
        predicted_load : predicted future household load
        appliances     : appliance information
        max_load       : maximum allowed load

    Output:
        list of appliances to turn off
    """

    # How much load do we need to remove?
    excess = max(0, predicted_load - max_load)

    if excess == 0:
        return []

    # Find appliances that:
    # 1. Are flexible
    # 2. Are currently ON
    controllable = [
        (name, data)
        for name, data in appliances.items()
        if data["flexible"] and data["on"]
    ]

    # Turn off the least costly appliances first
    controllable.sort(key=lambda x: x[1]["cost"])

    turned_off = []

    for name, data in controllable:

        if excess <= 0:
            break

        turned_off.append(name)
        excess -= data["power"]

    return turned_off


# ============================================================
# SIMULATION
# ============================================================

historical_loads = []

predicted_loads = []
actual_loads = []
controlled_loads = []

turn_off_history = []

for t in range(51):
    current_load = random.uniform(1,6)
    historical_loads.append(current_load)

for t in range(50):

    # Random current household load
    current_load = historical_loads[t]

    # Simple baseline forecast:
    # predict the mean of all loads seen so far
    predicted_load = noise(historical_loads[t])

    # Reset appliance states
    appliances = {
        name: {
            **data,
            "on": random.random() > 0.3
        }
        for name, data in APPLIANCES.items()
    }

    # -------------------------------
    # CONTROLLER
    # -------------------------------

    turned_off = controller(
        predicted_load,
        appliances,
        P_MAX
    )

    # Calculate how much load was removed
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
print("CONTROL SUMMARY")
print("=" * 50)

total_actions = 0

for name in APPLIANCES:

    count = sum(
        name in actions
        for actions in turn_off_history
    )

    if count > 0:
        print(f"{name:20s}: turned off {count} times")
        total_actions += count

print("-" * 50)
print(f"Total control actions: {total_actions}")

print(
    f"Average actual load:    "
    f"{sum(actual_loads) / len(actual_loads):.2f} kW"
)

print(
    f"Average controlled load: "
    f"{sum(controlled_loads) / len(controlled_loads):.2f} kW"
)

print(
    f"Peak actual load:        "
    f"{max(actual_loads):.2f} kW"
)

print(
    f"Peak controlled load:    "
    f"{max(controlled_loads):.2f} kW"
)


# ============================================================
# PLOT
# ============================================================
import os

os.makedirs("results/controller", exist_ok=True)

plt.figure(figsize=(12, 6))

plt.plot(controlled_loads, label="Controlled Load")

plt.axhline(
    P_MAX,
    linestyle="--",
    label="Maximum Load"
)

plt.xlabel("Time step")
plt.ylabel("Power (kW)")
plt.title("Forecast-Based Energy Control")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/controller_greedy.png",
    dpi=300
)

plt.close()

print("\nPlot saved to:")
print("results/controller/control_simulation.png")

for arr in [actual_loads, predicted_loads, controlled_loads]:
    print([round(x, 3) for x in arr])