# Variation 1:

### 1. Design SNN

SNN shall take in past power usage, along with current appliance factors.

Want:

* Heating
* Ventilation
* AC
* Fans
* Other things

**Dataset links:**

1. UK-REFIT Smart Meter Dataset
2. ECD-UY
3. Pecan Street

It will try to predict the energy consumption in the next 15 minutes.

Currently the SNN is able to predict rises and falls, but not the exact magnitudes of rises and falls.

The control module should be able to work with it.

### 2. Design Control Module

(Not the crux of the study, but will use the SNN results)

* Control Module can have something like **"Demand likely to be unusually high"**
* Can set a limit consumption for a particular time and try to optimize consumption below
* RL formulation can be:
  `R = -l1 max(0, P_total - P_limit)^2 - l2 E_cost - l3 C_discomfort - l4 N_switches`
* Forecasting errors can directly help influence the RL agent

# Variation 2:

SNN itself predicts the control factors (study later)

The input will be current HVAC + other appliance values + energy consumption.

SNN will directly output optimal usage parameters.

**Dataset links:**

1. Pecan Street
2. ECD-UY
3. ?
