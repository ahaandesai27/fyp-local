import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

np.random.seed(42)

N = 7 * 24 * 4          # 7 days, 15-minute intervals
dt = 15                 # minutes

time = pd.date_range(
    start="2026-01-01",
    periods=N,
    freq="15min"
)

# ============================================================
# Outdoor temperature
# ============================================================

t = np.arange(N)

# Daily temperature cycle
daily_cycle = 5 * np.sin(2 * np.pi * (t - 28) / 96)

# Slow variation over the week
weekly_cycle = 1.5 * np.sin(2 * np.pi * t / N)

# Noise
noise = np.random.normal(0, 0.5, N)

outdoor_temp = 15 + daily_cycle + weekly_cycle + noise

# ============================================================
# Humidity
# ============================================================

humidity = (
    60
    - 0.5 * (outdoor_temp - 15)
    + np.random.normal(0, 3, N)
)

humidity = np.clip(humidity, 30, 90)

# ============================================================
# Setpoint
# ============================================================

setpoint = np.full(N, 22.0)

# Slightly different nighttime setpoint
hours = time.hour + time.minute / 60

night = (hours < 7) | (hours >= 23)

setpoint[night] = 21.0

# ============================================================
# Indoor temperature simulation
# ============================================================

indoor_temp = np.zeros(N)

# Initial indoor temperature
indoor_temp[0] = 21.5

# HVAC control signals
heating = np.zeros(N)
cooling = np.zeros(N)
ventilation = np.zeros(N)

# ============================================================
# Simulate HVAC + building dynamics
# ============================================================

for i in range(N - 1):

    # Temperature error
    error = setpoint[i] - indoor_temp[i]

    # Simple thermostat behavior
    if error > 0.3:
        heating[i] = min(error * 2.0, 1.0)

    elif error < -0.3:
        cooling[i] = min(-error * 2.0, 1.0)

    else:
        heating[i] = 0
        cooling[i] = 0

    # Ventilation increases with humidity
    if humidity[i] > 65:
        ventilation[i] = min((humidity[i] - 65) / 20, 1.0)

    # Building naturally moves toward outdoor temperature
    heat_loss = 0.025 * (outdoor_temp[i] - indoor_temp[i])

    # Heating / cooling effect
    heating_effect = 0.12 * heating[i]
    cooling_effect = 0.12 * cooling[i]

    # Ventilation moves indoor temperature toward outdoor
    ventilation_effect = (
        0.02 * ventilation[i]
        * (outdoor_temp[i] - indoor_temp[i])
    )

    # Random disturbance
    disturbance = np.random.normal(0, 0.03)

    indoor_temp[i + 1] = (
        indoor_temp[i]
        + heat_loss
        + heating_effect
        - cooling_effect
        + ventilation_effect
        + disturbance
    )

# ============================================================
# HVAC energy consumption
# ============================================================

# Assume:
# Heater       = 2.5 kW maximum
# AC           = 3.0 kW maximum
# Ventilation  = 0.3 kW maximum

heating_power = 2.5 * heating
cooling_power = 3.0 * cooling
ventilation_power = 0.3 * ventilation

hvac_energy = (
    heating_power
    + cooling_power
    + ventilation_power
)

# ============================================================
# Create DataFrame
# ============================================================

df = pd.DataFrame({
    "Time": time,
    "OutdoorTemp": outdoor_temp,
    "IndoorTemp": indoor_temp,
    "Humidity": humidity,
    "Setpoint": setpoint,
    "Heating": heating,
    "Cooling": cooling,
    "Ventilation": ventilation,
    "HeatingPower": heating_power,
    "CoolingPower": cooling_power,
    "VentilationPower": ventilation_power,
    "HVAC_Energy": hvac_energy
})

# ============================================================
# Save
# ============================================================

df.to_csv("mock_hvac_data.csv", index=False)

print(df.head())
print("\nShape:", df.shape)
print("\nSaved as mock_hvac_data.csv")

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(df["Time"], df["IndoorTemp"], label="Indoor Temperature")
plt.plot(df["Time"], df["OutdoorTemp"], label="Outdoor Temperature")
plt.plot(df["Time"], df["Setpoint"], label="Setpoint")

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Mock HVAC Temperature Data")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()