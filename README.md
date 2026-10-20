# Power Ranger

Power Ranger is an engine-monitoring prototype developed collaboratively for the HackHive hackathon. This repository is my personal copy of a project I helped build with my teammates; the [original project repository](https://github.com/AtharvaMaik/HackHive) preserves our shared work. External datasets and reused acoustic-simulator code are credited below under their respective licenses.

## Project overview

Power Ranger explores engine fault detection through synchronized sensor data. It compares related readings to identify inconsistencies, presents the evidence in a dashboard, and includes separate tools for signal and engine-condition analysis. It is a prototype based on public test-bench recordings and simulation, not a validated vehicle safety system.

The main app replays a real engine test-bench recording with a known, deliberately injected sensor fault. It shows ten synchronized channels, compares each channel with what the other nine suggest it should read, and can stop the replay after repeated evidence of an inconsistency. You can choose a preset fault, add your own sensor bias, or turn off the guard to see a **simulated** failure sequence. The red failure sequence and microphone/vibration display are demonstrations; the source recording does not contain an engine breaking down.

A separate analysis workbench covers signal analysis for uploaded audio/CSV files, a sensor-correlation matrix, fault-probability experiments, an optional OBD-II capture path, a separate high-channel marine-engine dataset, remaining-useful-life experiments, and Bosch operating-map exploration. These are related experiments, **not one combined model trained on all datasets**. The Bosch “sweet spot” compares measured points under selected limits; it does not establish a real-world way to produce the same power using less fuel.

| Part | What it does | Where |
| --- | --- | --- |
| Live engine | Replays held-out LiU-ICE runs and demonstrates sensor-consistency shutdown | `app.py` |
| Sensor relationships | Shows a Spearman correlation matrix for selected Bosch trace channels in the Time series view | `legacy_app.py` |
| Signal lab | Inspects uploaded WAV/CSV signals and can train an experimental classifier from user-labeled recordings | `legacy_app.py` |
| Other research views | Explore Bosch maps, benchmark fault probabilities, marine data, OBD-II, and RUL | `legacy_app.py` |

## Run locally

Use Python 3.10 or newer. From the repository folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python prepare_liu_ice.py
streamlit run app.py
```

`prepare_liu_ice.py` downloads and prepares the public benchmark data if needed. The first run needs internet access and may take a while. To open the older workbench, run `streamlit run legacy_app.py`. Its marine and NASA views need `python prepare_marine.py` and `python prepare_cmapss.py`, respectively; the optional OBD-II view needs an adapter and `python -m pip install obd`.

## Results at a glance

On three held-out LiU-ICE replay presets, the live guard's window-level results were **100% precision, 73.4% recall, 84.6% F1, and 86.7% balanced accuracy**, with no pre-injection false alarms in those replays. This is a small, controlled benchmark, so those numbers are **not expected vehicle accuracy**. The other workbench models have separate evaluations and limitations described below; in particular, the marine detector's current threshold produces a 100% false-alarm rate and is not useful as a warning system.

## Live fault detection

One real-time simulation cockpit that replays predefined LiU-ICE WLTP engine runs and shows ten synchronized sensor streams arriving together.

The dashboard starts automatically. **New Run** randomly selects a different held-out preset, **Alert** jumps to ten seconds before the model's first confirmed inconsistency, and **Reset** returns to the beginning of the current run.

**EngineGuard** uses ten redundant-sensor consistency models, one for each stream. Each Extra Trees regressor is trained only on healthy windows and estimates one sensor from the other nine synchronized sensor means. The original three benchmark channels retain their bundled limits; the seven added channels calibrate limits on healthy trajectories held out from each regressor's fit. The live score is the observed-versus-estimated residual divided by its 98.5th-percentile healthy residual. Shutdown requires the same sensor to cross its learned limit in at least two of three consecutive five-second windows and to reach at least 1.25× in the current window.

The **manual sensor injection** panel has an independent relative-bias slider for each of the ten streams. It applies deterministic companion shifts to a few related sensors, previews the result against the healthy-only consistency models, and arms only when the same two-of-three confirmation rule can stop the replay. This demonstration does not change the measured LiU-ICE source data.

The three dashboard presets are separate complete trajectories excluded from model fitting: 5% intake-manifold-pressure bias, 15% intercooler-pressure bias, and 15% air-mass-flow bias. **New Run** always switches to a different recording. Across these held-out replays, the stricter confirmation rule has **100% precision, 73.4% recall, 84.6% F1, 86.7% balanced accuracy, and zero pre-injection false alarms** at the window level. First confirmation occurs 18.1, 6.1, and 9.1 seconds after injection, respectively. The model never reads the injection timestamp when deciding to stop.

In this benchmark, **fault injection** means deliberately activating a sensor bias at a published time; it does not mean fuel injection or a physical pressure pulse. EngineGuard waits for measured inconsistency after that injection exists. The old 30-second onset classifier remains in the legacy workbench for analysis, but it no longer controls this dashboard because its apparent early warnings were not supported by causal sensor evidence.

The source recordings do not contain terminal engine failures, so the unified dashboard adds a clearly labeled simulation layer after injection. If EngineGuard is disabled, the affected sensor drifts further, engine speed, airflow, manifold pressure, and fuel delivery deteriorate together, health reaches zero, and the replay ends at **SIMULATED FAILURE**. Airflow is shown in the top KPI row so this collapse is visible without scrolling. Injection and failure markers appear only after the replay reaches each timestamp. If EngineGuard is enabled, the evidence-based shutdown happens earlier and reports the remaining margin to the same failure point. The injected sensor data and model decision remain measured benchmark evidence; the subsequent damage progression is a deterministic demonstration rather than a claim about real component lifetime.

The live display includes:

- Ten source sensors with engineering-unit conversions and deviation from a matching healthy WLTP reference run.
- A scrolling normalized sensor envelope, data-bus packet animation, health index, and event feed.
- A synthetic microphone and accelerometer carrier driven by the replay's measured engine speed and throttle-derived load.

Start it with `streamlit run app.py`. The prior multi-page workbench is preserved in `legacy_app.py`.

## Additional tools

- **Measured evidence:** Bosch gasoline-engine performance maps and trace, LiU-ICE automotive injected-fault runs, and the separate 65-channel marine-diesel benchmark.
- **Signal tools:** uploaded WAV/CSV analysis and recording-held-out experiments, plus EngineTwin's simulated inline-four sound, vibration, OBD values, order spectrum, valve envelope, and cylinder-energy views under **Acoustic twin**.
- **Monitoring and prognosis:** calibrated LiU-ICE active-fault and 30-second onset models, live OBD-II capture, and user-supplied or NASA C-MAPSS RUL workflows.
- **Efficiency:** feasible measured Bosch operating points and exploratory control importance.

These workflows remain available in `legacy_app.py` for development and future reuse. Real recordings, injected benchmark faults, and simulated signals stay clearly labeled and are not combined into one training set. The integrated acoustic-simulator package lives in `enginetwin/` and retains its MIT license.

EngineTwin's CNN/autoencoder dashboard and its Ocslab/VED trip optimizer require its separately prepared source datasets and trained `cache/` artifacts. Those assets are not included in the clone, so the unified app uses its reusable simulator and signal-analysis modules while the measured Bosch, LiU-ICE, marine, OBD-II, and RUL workflows run from this workspace's available data.

`prepare_liu_ice.py` downloads the official public benchmark archive only if the raw CSVs are not present, validates archive paths before extraction, and makes a local compressed feature cache. The raw benchmark archive is not committed. Once the cache exists, model fitting runs locally and is cached for the Streamlit session.

## Datasets and provenance

### Bosch engine maps

The bundled files are from [Bosch-Engine-Datasets](https://github.com/boschresearch/Bosch-Engine-Datasets), licensed CC BY-SA 4.0; the license is copied under `data/bosch/LICENSE.txt`.

- `engine1_normalized.xlsx` and `engine2_normalized.xlsx`: 795 and 785 measured operating points after cleanup. Each has 11 columns: four controls (`engine_speed`, `engine_load`, `intake_valve_opening`, `air_fuel_ratio`) and seven responses. The source standardizes values to z-scores, so the app cannot claim SI units or absolute operating limits.
- `safety_constraint.xlsx`: suggested upper bounds for response values, also in standardized coordinates.
- `gengine1_trajectory_10.csv`: one 13-column time trace sampled at 10 Hz. The repository documents 33 trajectories in the full series; this workspace bundles one. The map, trace, and LiU data are separate experiments and are never fused into one model.

The Overview shows the loaded Bosch time series and the best observed fuel-use point that satisfies available safety caps. The **Sweet spot** page lets users change objective and guardrails. Neither page extrapolates an unconstrained response surface: “sweet spot” means the best feasible measured point for the selected objective and engine map, not a universal physical optimum. Model feature importance uses a held-out row split because the map workbooks do not provide run IDs; its estimates are exploratory.

### LiU-ICE fault diagnosis benchmark

The application uses the [official LiU-ICE benchmark and training archive](https://vehsys.gitlab-pages.liu.se/dx_benchmarks/liuice/liuice_index), which provides synchronized engine test-bench runs at 20 Hz, eight sensor channels, two actuator channels, two drive cycles, and known injected fault-onset times. The current downloadable archive used here contains **41 trajectories** (22 WLTP, 19 FTP75 city), although the benchmark page currently describes 26 training files; this archive/page count discrepancy is noted rather than hidden.

After dropping incomplete final windows, this copy produces **11,252 non-overlapping five-second windows** and **40 engineered features**. Four features are generated per each of the ten channels: mean, population standard deviation, range, and linear slope. This benchmark supplies ten synchronous streams; the separate marine dataset below supplies 65 usable channels. There is no fused 75-channel model.

Run labels in the archive: 12 intercooler pressure sensor fault runs, 12 air-mass-flow sensor fault runs, 12 intake-manifold pressure sensor fault runs, two intake-manifold leak runs, and three normal runs. The small normal/leak counts constrain calibration and fault isolation. The benchmark's four classes are controlled injected faults, not catastrophic failures or component remaining-life events.

### Marine engine dataset

To meet the 30+ synchronous-signal goal without fabricating channels, the **65-channel diesel** page separately uses the [Marine Engine Fault dataset on Zenodo](https://zenodo.org/records/19857425), acquired on a physically operated marine diesel test bench and released under CC BY 4.0. The archive has 15 scenario files and one baseline reference file, five physical fault modes, and 70 additional measured/derived channels per scenario file. **65 channels** pass the minimum coverage/variation filter over the 15 scenario files; this workspace parses 89,468 labeled samples. This source is marine diesel rather than automotive gasoline, so its data and models stay separate from the automotive prognosis pipeline.

Two injector-nozzle recordings contain only anomalous samples and no normal lead-in, so they remain in the source inventory but are excluded from the pre-fault anomaly model. The model uses the other 13 transition runs, keeps whole recordings together during an 8/5 train/test split, selects 24 channels by training-only random-forest importance, median-imputes from training runs, then calibrates grouped out-of-fold random-forest scores with logistic sigmoid calibration. At the F2-selected threshold the deterministic holdout predicts every sample as anomalous (100% recall and 100% false-alarm rate); ROC AUC is 0.751, PR AUC 0.869 against a 67.7% anomaly base rate, and Brier score 0.215. This is not an actionable detector. The page exposes the high-channel model's limitations and sensor ranking as an exploratory track, not a reliable fault warning.

The marine model's current 24 selected channels are: `Charge Air IC Cooling Water Temp. Out`, `Compressor Filter Loss`, `Sea Cooling Water Press.`, `No.3 Exh.Gas Temp.`, `LO Temp. TCH In`, `Charge Air Press.`, `Charge Air IC Air Temp. Out`, `Exh.Gas Temp. Turbine Out`, `LO Temp. TCH Out`, `Cooling Water Temp. Engine Out II`, `LO Temp. Engine Out`, `Turbine Back Pressure`, `Water Brake Weight`, `Engine Speed`, `Shaft Torque`, `Fuel transfer pump Press.`, `LO Cooling Water Temp. In`, `Fuel Temp.`, `Charge Air IC Cooling Water Temp. In`, `LO Circulating Pump Press.`, `Exh.Gas Vol. Flow (From Engine Speed)`, `No.2 Exh.Gas Temp.`, `Engine Cooling water flow`, `TCH LO Cooling water flow`.

## Methods

### Fault detection and onset estimates

The **Failure probability** page fits two binary models and one conditional fault-isolation model:

1. **Active-fault probability:** estimates whether one of the four known injected faults is active in a five-second window.
2. **30-second onset probability:** estimates whether an official fault injection will begin within the next 30 seconds. This is the main prevention-oriented forecast shown in the dashboard; it is not real-world probability of engine failure.
3. **Fault-mode isolation:** a random forest ranks the four known fault types conditional on an active fault. Mode scores are not calibrated failure probabilities.

Both probability models use random forests (280 trees, depth 14, minimum leaf size 4, class weighting) and a logistic sigmoid calibrator trained on grouped out-of-fold predictions. Complete trajectories—not windows—are assigned to train/test partitions. Feature selection is repeated inside each calibration fold to avoid selecting on its validation trajectories. The outer held-out split reserves about 25% of trajectories. Thresholds are selected from grouped training predictions: a 95th-percentile healthy threshold for active-fault detection and maximum F2 for the onset warning. F2 favors recall while still penalizing false alarms.

Each binary model starts with 40 features, then keeps the training-run random-forest importance ranking up to 95% cumulative importance, bounded to 8–24 features. On the current active-fault selection, the 24 retained features are:

`ambient_temperature__mean`, `ambient_pressure__mean`, `Intercooler_pressure__mean`, `ambient_temperature__slope`, `engine_speed__mean`, `wastegate_position__mean`, `intake_manifold_pressure__mean`, `intercooler_temperature__mean`, `wastegate_position__range`, `wastegate_position__std`, `ambient_pressure__std`, `ambient_temperature__range`, `air_mass_flow__mean`, `engine_speed__slope`, `injected_fuel_mass__mean`, `intake_manifold_pressure__range`, `engine_speed__range`, `throttle_position__std`, `ambient_temperature__std`, `Intercooler_pressure__range`, `throttle_position__mean`, `intake_manifold_pressure__std`, `air_mass_flow__std`, `engine_speed__std`.

The selection is recomputed from the training partition when the model is fit; the list above documents the deterministic seed used for the checked build. The page includes selected-feature rankings, held-out probabilities over a trajectory, the injected onset marker, and fault-mode scores.

The checked 30-second onset model selected:

`wastegate_position__mean`, `Intercooler_pressure__mean`, `ambient_temperature__slope`, `air_mass_flow__mean`, `injected_fuel_mass__mean`, `intake_manifold_pressure__mean`, `engine_speed__mean`, `ambient_temperature__mean`, `ambient_pressure__mean`, `ambient_temperature__std`, `ambient_temperature__range`, `throttle_position__mean`, `air_mass_flow__std`, `Intercooler_pressure__std`, `intercooler_temperature__mean`, `engine_speed__slope`, `Intercooler_pressure__slope`, `throttle_position__std`, `wastegate_position__std`, `injected_fuel_mass__slope`, `engine_speed__std`, `air_mass_flow__slope`, `Intercooler_pressure__range`, `intake_manifold_pressure__std`.

The **Simulation** page replays 36 five-second windows (180 simulated seconds) in 60 seconds at 3× playback. It builds a fault what-if by ramping in a severity-scaled, per-feature median shift (fault-active training rows minus normal training rows), matched by drive cycle when sample counts permit; the ramp spans the early-warning model's 30-second horizon before the configured fault onset. Both probability models score each visible window as the replay advances. An animated conceptual inline-four cutaway turns red at the user-configured scripted failure event; that event is explicitly separate from, and not caused by, model output. This is a transparent feature-space demonstration, not a first-principles engine/vehicle digital twin: the local cache has window features rather than the raw 20 Hz waveforms, and its scores remain probabilities for the benchmark's injected fault labels.

### Signal analysis and monitoring

- **Signatures:** Welch power spectral density, STFT, RMS, spectral centroid, excess kurtosis, skewness, and crest factor for uploaded CSV sensor traces or mono/multichannel WAV audio. The page can train an experimental RBF-SVM from user-labeled recordings, evaluate it with whole recordings held out, and score new recordings using the fitted model. The project still ships with no labeled automotive knock, misfire, or valve-wear recordings, so it has no pre-trained acoustic fault classifier.
- **Live OBD-II:** the optional Live OBD-II page captures supported standard sensor PIDs through a connected ELM327-compatible serial adapter using the optional `obd` package (`python -m pip install obd`). Captures are read-only and downloadable. PID availability varies by vehicle, standard OBD-II may expose fewer than 30 fields, and this has not been exercised against a physical adapter in this workspace. High-rate acoustic/vibration acquisition still requires separate sensors and synchronized acquisition hardware.
- **Time series:** Isolation Forest produces a relative out-of-envelope score on the Bosch trace. The score is not a failure probability.

### Remaining useful life

The **RUL prognosis** page accepts labeled long-format histories. It uses causal rolling means, standard deviations, and trailing slopes, selects features by permutation importance on inner held-out engines, and evaluates a random-forest regressor on separate engines. It reports MAE and R², and can estimate RUL for current histories. Automotive use still needs representative automotive run-to-failure records or supplied RUL labels.

An optional NASA C-MAPSS FD001 benchmark can be prepared with `python prepare_cmapss.py`. It provides simulated aircraft turbofan run-to-failure histories and labeled truncated test engines, allowing the RUL path to be evaluated without a user upload. This is a separate aerospace benchmark, not automotive evidence.

### Operating-point exploration

The operating-map pages inspect four controls against fuel use, exhaust/catalyst temperature, roughness, HC, and NOx. A random forest and held-out permutation importance identify influential controls. The candidate list is filtered against selected Bosch upper bounds and ranked among observed map points. All map coordinates and limits are standardized z-scores. Validate candidate settings on the target engine before treating any point as a calibration recommendation.

## Evaluation

The reproducible grouped holdout currently has 30 training and 11 held-out trajectories. On the onset target, the holdout ROC AUC is **0.968**, PR AUC **0.497** (base rate **1.86%**), Brier score **0.0132**, recall **83.3%**, precision **20.6%**, and false-alarm rate **6.1%** at the F2-selected threshold. The onset test set has a low positive base rate, so read PR AUC against that baseline and do not rely on ROC AUC alone. At the 5%-healthy-OOF threshold, the active-fault model achieves **0.5% recall** and **0% false-alarm rate** on this holdout, which is not a usable active-state detector. Fault isolation has **40.1% accuracy** / **0.307 macro F1** over ten held-out trajectories, constrained by just two leak trajectories. These are single-split benchmark scores, not production performance claims.

Do not deploy this as a production safety system. Validation still needs repeated group splits or a hidden test set, more independent normal and fault magnitudes, other engine/vehicle populations, and probability recalibration on representative field prevalence. The official benchmark itself warns its training fault magnitudes and operating conditions are not representative of all possible test faults.

## Limitations

- **Failure probability:** benchmark fault-onset probability is implemented and evaluated. A true probability of mechanical failure or time-to-failure still requires representative automotive run-to-failure labels.
- **30 data streams:** the separate marine diesel runs provide 65 usable synchronous channels. The automotive gasoline/vehicle fault benchmark has ten, and the Bosch maximum bundled trace has 13; data sources are kept separate because the engines and experiments differ.
- **Acoustic knock / misfire / valve wear:** no such labeled recordings are bundled. The Signatures page now trains an experimental SVM from user-labeled recordings with recording-level holdout; verified vehicle fault recordings are still needed for an out-of-box detector.
- **CAN/OBD telemetry:** a Live OBD-II page now captures supported standard PIDs through an ELM327-compatible serial adapter (`python -m pip install obd`). No adapter is connected here; vehicle PID support varies, and generic OBD-II does not provide 30+ guaranteed streams. Aligning this with high-rate sensor hardware remains an acquisition task.
- **Engine sweet spot:** best feasible measured points can be explored in Bosch z-score maps. Physical-unit calibration, target-engine validation, and approved safety limits are required for engineering recommendations.

The Signatures page now supports training and inference with a recording-held-out SVM when users provide labeled, sample-aligned CSV/WAV recordings (at least two recordings per class, with a shared sample rate). The Live OBD-II page provides a read-only adapter capture path for standard PIDs. NASA C-MAPSS FD001 provides a separate simulated run-to-failure example. These add integration points and benchmarks; they do not close the automotive source-data gaps by themselves. Publicly described fine-grained engine-fault audio/vibration datasets include 2,643 paired audio/vibration samples across 232 vehicle models, but the research publication does not provide a public download link. The public vehicle-sound corpus available from Zenodo covers engine sound and operating conditions rather than verified fault labels, so it cannot train the requested fault classes. Real labeled automotive data, adapter hardware, and synchronized high-rate acquisition remain required for an end-to-end automotive detector.

## References

- LiU-ICE official dataset, signal definitions, fault definitions, baseline model/code links, and benchmark scoring: [benchmark page](https://vehsys.gitlab-pages.liu.se/dx_benchmarks/liuice/liuice_index).
- Benchmark paper: Jung, Frisk, and Krysander, [The LiU-ICE Benchmark – An Industrial Fault Diagnosis Case Study](https://arxiv.org/abs/2408.13269).
- Marine Engine Fault dataset and 70-channel variable inventory: [Zenodo record](https://zenodo.org/records/19857425); [data descriptor preprint](https://arxiv.org/abs/2607.19444).
- NASA C-MAPSS run-to-failure data: [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
- Fine-grained automotive fault audio/vibration paper (dataset access is not linked in the publication): [Fedorishin et al.](https://arxiv.org/abs/2403.11037).
- Public engine sound corpus (operating-condition and engine-type labels, not verified fault labels): [Zenodo record](https://zenodo.org/records/18777405).
- Read-only OBD capture library: [python-OBD documentation](https://python-obd.readthedocs.io/en/latest/).
- Bosch source data and variable documentation: [Bosch-Engine-Datasets repository](https://github.com/boschresearch/Bosch-Engine-Datasets).
- Random-forest method: [Breiman (2001)](https://doi.org/10.1023/A:1010933404324); probability calibration: [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html); permutation importance: [scikit-learn documentation](https://scikit-learn.org/stable/modules/permutation_importance.html).
- Bosch safe active learning reference: [SALMOGP repository](https://github.com/boschresearch/SALMOGP) and [paper](https://arxiv.org/abs/2203.14849). It is cited as future work, not copied into this project.

### Interface references

- [Plotly time-series range slider examples](https://github.com/plotly/plotly.py/blob/main/doc/python/range-slider.md): the shared time navigator and quick-range controls on synchronized signal panels and held-out fault replays follow this native Plotly pattern.
- [Streamlit `st.plotly_chart` documentation](https://github.com/streamlit/docs/blob/main/content/develop/api-reference/charts/plotly_chart.md): charts use responsive full-width containers and consistent app styling.
- [plotly-resampler](https://github.com/predict-idlab/plotly-resampler) and its [MinMaxLTTB paper](https://arxiv.org/abs/2305.00332): reviewed for high-volume trace scaling. The current Bosch trace is only 13,364 points, so the app displays all source samples instead of adding a resampling dependency that could hide transient peaks.
