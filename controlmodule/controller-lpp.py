import random
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


FEATURES = [
    "grid",
    "air1",
    "furnace1",
    "solar",
    "refrigerator1",
    "car1",
    "waterheater1",
    "drye1",
    "dishwasher1",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

P_MAX = 4.0
N_STEPS = 168

APPLIANCES = {
    "air1": {
        "power": 1.5,
        "cost": 5,
        "flexible": True,
        "base_rate": 0.08,
        "persistence": 0.88,
    },
    "furnace1": {
        "power": 2.0,
        "cost": 5,
        "flexible": True,
        "base_rate": 0.04,
        "persistence": 0.92,
    },
    "refrigerator1": {
        "power": 0.2,
        "cost": 100,
        "flexible": False,
        "base_rate": 0.7,
        "persistence": 0.97,
    },
    "car1": {
        "power": 3.0,
        "cost": 3,
        "flexible": True,
        "base_rate": 0.03,
        "persistence": 0.90,
    },
    "waterheater1": {
        "power": 2.0,
        "cost": 2,
        "flexible": True,
        "base_rate": 0.08,
        "persistence": 0.86,
    },
    "drye1": {
        "power": 2.5,
        "cost": 1,
        "flexible": True,
        "base_rate": 0.025,
        "persistence": 0.88,
    },
    "dishwasher1": {
        "power": 1.2,
        "cost": 1,
        "flexible": True,
        "base_rate": 0.035,
        "persistence": 0.90,
    },
}


ACTIVITY_PROFILES = {
    "air1": {
        0: 0.15,
        6: 0.25,
        9: 0.45,
        12: 0.75,
        15: 1.0,
        18: 1.1,
        21: 0.8,
    },
    "furnace1": {
        0: 1.2,
        6: 1.0,
        9: 0.45,
        12: 0.2,
        15: 0.25,
        18: 0.8,
        21: 1.1,
    },
    "refrigerator1": {
        0: 1.0,
        6: 1.0,
        12: 1.0,
        18: 1.0,
        23: 1.0,
    },
    "car1": {
        0: 0.05,
        6: 0.15,
        9: 0.45,
        12: 0.55,
        15: 0.7,
        18: 0.9,
        21: 0.6,
    },
    "waterheater1": {
        0: 0.25,
        6: 1.1,
        9: 0.45,
        12: 0.2,
        15: 0.3,
        18: 0.8,
        21: 1.0,
    },
    "drye1": {
        0: 0.05,
        6: 0.15,
        9: 0.35,
        12: 0.4,
        15: 0.5,
        18: 0.9,
        21: 0.7,
    },
    "dishwasher1": {
        0: 0.05,
        6: 0.1,
        9: 0.1,
        12: 0.15,
        15: 0.2,
        18: 0.8,
        21: 1.0,
    },
}


def interpolate_profile(profile, hour):
    hours = sorted(profile.keys())

    if hour <= hours[0]:
        return profile[hours[0]]

    if hour >= hours[-1]:
        return profile[hours[-1]]

    for i in range(len(hours) - 1):
        h1 = hours[i]
        h2 = hours[i + 1]

        if h1 <= hour <= h2:
            v1 = profile[h1]
            v2 = profile[h2]

            return v1 + (v2 - v1) * (hour - h1) / (h2 - h1)

    return 1.0


def solar_generation(hour):
    if hour < 6 or hour > 18:
        return 0.0

    angle = np.pi * (hour - 6) / 12

    return max(
        0.0,
        3.0 * np.sin(angle)
    )


def update_appliance_states(states, hour):
    new_states = {}

    for name, data in APPLIANCES.items():

        activity = interpolate_profile(
            ACTIVITY_PROFILES[name],
            hour
        )

        if states[name]:

            stay_on = (
                random.random()
                < data["persistence"]
            )

            if stay_on:
                new_states[name] = True
            else:
                new_states[name] = False

        else:

            lam = (
                data["base_rate"]
                * activity
            )

            new_states[name] = (
                np.random.poisson(lam) > 0
            )

    return new_states


def calculate_load(states, hour):

    appliance_load = sum(
        APPLIANCES[name]["power"]
        for name, on in states.items()
        if on
    )

    solar = solar_generation(hour)

    base_load = (
        0.8
        + 0.3 * np.sin(
            2 * np.pi * hour / 24
        )
    )

    noise = np.random.normal(
        0,
        0.08
    )

    grid = max(
        0,
        base_load
        + appliance_load
        - solar
        + noise
    )

    return grid, solar


def create_features(
    grid,
    states,
    solar,
    hour,
    dow
):

    return {
        "grid": grid,
        "air1": int(states["air1"]),
        "furnace1": int(states["furnace1"]),
        "solar": solar,
        "refrigerator1": int(states["refrigerator1"]),
        "car1": int(states["car1"]),
        "waterheater1": int(states["waterheater1"]),
        "drye1": int(states["drye1"]),
        "dishwasher1": int(states["dishwasher1"]),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    }


def controller(
    predicted_load,
    appliances,
    max_load
):

    excess = max(
        0,
        predicted_load - max_load
    )

    if excess <= 0:
        return []

    controllable = [
        (name, data)
        for name, data in appliances.items()
        if data["flexible"] and data["on"]
    ]

    if not controllable:
        return []

    names = [
        name
        for name, _ in controllable
    ]

    powers = np.array([
        data["power"]
        for _, data in controllable
    ])

    costs = np.array([
        data["cost"]
        for _, data in controllable
    ])

    if powers.sum() < excess:
        return []

    constraint = LinearConstraint(
        powers,
        excess,
        np.inf
    )

    result = milp(
        c=costs,
        integrality=np.ones(
            len(names)
        ),
        bounds=Bounds(
            0,
            1
        ),
        constraints=constraint
    )

    if not result.success:
        return []

    return [
        names[i]
        for i, x in enumerate(result.x)
        if x > 0.5
    ]


def calculate_controlled_load(
    states,
    turned_off,
    hour
):

    controlled_states = states.copy()

    for name in turned_off:
        controlled_states[name] = False

    controlled_load, solar = calculate_load(
        controlled_states,
        hour
    )

    return (
        controlled_load,
        controlled_states
    )


states = {
    name: False
    for name in APPLIANCES
}

actual_loads = []
predicted_loads = []
controlled_loads = []

feature_history = []
state_history = []
turn_off_history = []

control_failures = []
violations_before = []
violations_after = []


for t in range(N_STEPS):

    hour = t % 24
    dow = (t // 24) % 7

    states = update_appliance_states(
        states,
        hour
    )

    current_load, solar = calculate_load(
        states,
        hour
    )

    predicted_load = max(
        0,
        current_load
        + np.random.normal(
            0,
            0.12 * max(current_load, 1)
        )
    )

    appliances = {
        name: {
            **data,
            "on": states[name]
        }
        for name, data in APPLIANCES.items()
    }

    turned_off = controller(
        predicted_load,
        appliances,
        P_MAX
    )

    controlled_load, controlled_states = (
        calculate_controlled_load(
            states,
            turned_off,
            hour
        )
    )

    if predicted_load > P_MAX and not turned_off:
        control_failures.append(t)

    if current_load > P_MAX:
        violations_before.append(t)

    if controlled_load > P_MAX:
        violations_after.append(t)

    features = create_features(
        current_load,
        states,
        solar,
        hour,
        dow
    )

    actual_loads.append(current_load)
    predicted_loads.append(predicted_load)
    controlled_loads.append(controlled_load)

    feature_history.append(features)
    state_history.append(states.copy())
    turn_off_history.append(turned_off)


print()
print("=" * 60)
print("CONTROL SUMMARY")
print("=" * 60)

total_actions = 0

for name in APPLIANCES:

    count = sum(
        name in actions
        for actions in turn_off_history
    )

    if count > 0:

        print(
            f"{name:20s}: "
            f"turned off {count:3d} times"
        )

        total_actions += count

print("-" * 60)

print(
    f"Total control actions:     "
    f"{total_actions}"
)

print(
    f"Average actual load:       "
    f"{np.mean(actual_loads):.2f} kW"
)

print(
    f"Average controlled load:   "
    f"{np.mean(controlled_loads):.2f} kW"
)

print(
    f"Peak actual load:           "
    f"{max(actual_loads):.2f} kW"
)

print(
    f"Peak controlled load:       "
    f"{max(controlled_loads):.2f} kW"
)

print(
    f"Violations before control: "
    f"{len(violations_before)}"
)

print(
    f"Violations after control:  "
    f"{len(violations_after)}"
)

print(
    f"Control failures:          "
    f"{len(control_failures)}"
)

print(
    f"Violation reduction:       "
    f"{100 * (1 - len(violations_after) / max(len(violations_before), 1)):.1f}%"
)

print("=" * 60)


os.makedirs(
    "results",
    exist_ok=True
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    actual_loads,
    label="Actual Load",
    linewidth=1.8
)

plt.plot(
    predicted_loads,
    label="Predicted Load",
    linewidth=1.2,
    alpha=0.7
)

plt.plot(
    controlled_loads,
    label="Controlled Load",
    linewidth=1.8
)

plt.axhline(
    P_MAX,
    linestyle="--",
    label="Maximum Load"
)

plt.xlabel("Time step")
plt.ylabel("Power (kW)")
plt.title("Forecast-Based MILP Energy Control")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/controller_milp.png",
    dpi=300
)

plt.close()


print()
print(
    "Plot saved to:"
)

print(
    "results/controller_milp.png"
)


print()
print("Actual loads:")

print([
    round(x, 3)
    for x in actual_loads
])

print()
print("Predicted loads:")

print([
    round(x, 3)
    for x in predicted_loads
])

print()
print("Controlled loads:")

print([
    round(x, 3)
    for x in controlled_loads
])