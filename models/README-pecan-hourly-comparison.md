# Pecan Street hourly model comparison

Three independent runners train the SNN, LSTM, and GRU on the same Pecan Street Austin data pipeline:

- `pecan_hourly_snn.py`
- `pecan_hourly_lstm.py`
- `pecan_hourly_gru.py`

Each 15-minute row is an interval-average power reading. The pipeline averages the four quarter-hour readings into one hourly average, uses the prior 24 hours to predict the next hour, and predicts the change in `grid` before reconstructing the absolute forecast. The shared input features follow the existing Pecan SNN setup: `grid`, `air1`, `furnace1`, and cyclical hour/day-of-week features.

Homes are split chronologically 70%/15%/15%. Scalers use training rows only. Windows with missing hourly grid target/history are skipped. Short appliance gaps are interpolated, with remaining missing appliance readings filled as zero. Appliance outlier bounds are derived from training data. All three models use the same loss, optimizer family, early stopping rule, metrics, and persistence baseline.

Run each model from the project root in the project virtual environment:

```bash
python models/pecan_hourly_snn.py ~/Downloads/15minute_data_austin.csv --metadata ~/Downloads/metadata.csv
python models/pecan_hourly_lstm.py ~/Downloads/15minute_data_austin.csv --metadata ~/Downloads/metadata.csv
python models/pecan_hourly_gru.py ~/Downloads/15minute_data_austin.csv --metadata ~/Downloads/metadata.csv
```

The metadata argument is accepted for convenience but is not required by the current model pipeline. Results follow the project conventions and are saved separately under `results/pecan_hourly/{snn,lstm,gru}/`, each with `model.pt`, `metrics.json`, `training_history.csv`, `predictions.csv`, and plots.

Run all three with the same seed and hyperparameters when comparing them. Defaults use seed 42 and batch size 1024, with one fixed shuffle generator so all three see the same training-window order. Report metrics in the source grid units; the current Pecan CSV stores power values as kW-like interval averages, so hourly aggregation is an average power in the same units rather than kWh energy.
