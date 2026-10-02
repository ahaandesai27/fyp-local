Updated todo list

I’ll inspect the current experiment notes and saved metrics alongside the notebook, then I’ll separate the highest-impact modeling changes from lower-value tuning ideas. The main question is whether the next gain is likely to come from data/split design, target formulation, or the SNN architecture itself.

Read 

Read 

Read 

Read 

The local notes already point to the central limitation: the model gets direction right but underestimates rare large transitions, and your experiments show the best result came from recurrent LIF dynamics with a spike readout, while weighted MSE worsened overall error. I’m checking the available result artifacts and neighboring model variants now so the recommendations are tied to what you have actually measured.

Read 

Read 

Read 

Read 

Read 

Read 

One important implementation detail surfaced: the current notebook is already a delta-target model, so the next upgrade should focus on preserving amplitude and evaluating peaks explicitly rather than trying another generic loss first. I’m pulling the saved baseline metrics and prediction distributions to identify the most discriminating next experiment.

Read , lines 1 to 260

Read 

Read 

Read 

Read , lines 1 to 300

Your next gains are most likely to come from **peak handling and evaluation**, not simply making the SNN larger.

**Highest-priority upgrades**

1. **Add peak-specific metrics**
   Track peak MAE/RMSE, peak bias, actual vs predicted maximum, and amplitude ratio. Your normal RMSE can improve while peak forecasting gets worse. The experiment specs in `peak_aware_symmetric_loss.txt` already define a good evaluation protocol.

2. **Add a persistence/residual skip connection**
   You already predict:

   ```python
   predicted_grid = last_grid + predicted_delta
   ```

   Add a direct non-spiking path from recent deltas or the latest grid value into the final prediction. This lets the SNN model temporal structure while the skip path preserves sudden changes. Your proposed design in `rate_of_change_skip_connection.txt` is a strong next experiment.

3. **Use multi-scale recent-change features**
   Add train-only-scaled features such as:

   ```python
   grid_delta_1 = grid.diff(1)
   grid_delta_2 = grid.diff(2)
   grid_delta_4 = grid.diff(4)
   grid_delta_8 = grid.diff(8)
   rolling_mean_4 = grid.rolling(4).mean()
   rolling_max_4 = grid.rolling(4).max()
   rolling_std_8 = grid.rolling(8).std()
   ```

   The current model sees the raw history, but explicit changes and local volatility make appliance switching events easier to learn.

4. **Try a two-head objective**
   Predict both:

   - next-step delta
   - whether the next step is a large transition

   Then use:

   ```text
   total_loss = delta_loss + lambda * peak_classification_loss
   ```

   This separates “will a peak occur?” from “what exact magnitude will it have?”, which is often easier for an SNN.

5. **Try asymmetric or quantile regression**
   Your weighted Huber experiment appears to improve neither overall error nor peak behavior. Instead, try a quantile head, especially the 0.8 or 0.9 quantile, and compare peak recall and bias. This directly discourages systematic under-prediction.

6. **Use an ANN readout after the SNN encoder**
   Keep the SNN layers, but replace the single leaky membrane readout with a small non-spiking head using:

   - final membrane state
   - maximum membrane state
   - mean membrane state
   - spike count

   Concatenate those representations and predict the delta with a two-layer MLP. The current single readout can smooth rare high activations.

7. **Test graded spikes**
   Your notes correctly identify binary spike saturation as a possible bottleneck. Experiment with graded-spike or analog membrane features in the readout. This is especially relevant because the model must preserve magnitude, not just event timing.

8. **Compare against a non-SNN upper bound**
   Train a small GRU, TCN, or MLP on exactly the same windows and features. If the ANN substantially outperforms the SNN, the issue is likely spike representation or readout capacity rather than the data.

**Data and experiment improvements**

- Add outdoor temperature, humidity, solar/weather forecasts, and calendar/holiday indicators if available.
- Evaluate by home, season, hour, and peak magnitude rather than reporting one aggregate score.
- Run at least 3 to 5 seeds; one seed is not enough to establish an improvement.
- Test windows of 48, 96, and 192 steps. A 24-hour window may miss weekly behavior; a 48-hour window may also make the SNN harder to train.
- Consider a signed transform such as `asinh(delta / scale)` rather than `log1p`, since grid deltas can be negative.

One important limitation: the per-home embedding in `snn-v3.3.ipynb` helps when the same homes appear in train and test, but it cannot represent an unseen home. For a stronger generalization claim, run a second evaluation where entire homes are held out.

My recommended order is:

1. Add peak metrics and naive baselines.
2. Run the rate-of-change skip connection.
3. Add the ANN multi-statistic readout.
4. Try the two-head peak auxiliary loss.
5. Compare against a GRU or TCN baseline.

Also update the saved metadata: the notebook currently labels the model as `SNN_v2_delta` and the plot titles as `SNN v2`, although this is your v3.3 experiment.