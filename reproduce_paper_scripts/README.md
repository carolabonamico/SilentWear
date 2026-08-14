# Running the experiments

Two entry points. Everything else in the repository builds on them.

| script | what it does |
|---|---|
| `20_make_windows_and_features.py` | raw `.bio` → prepared recordings → windowed dataset |
| `30_run_experiments.py` | windowed dataset → trained models, per-fold metrics and dumps |

Both are plain command-line programs: no launcher, no queue, no job array. That
is deliberate, because it is what lets you place each run on the GPU of your
choice and start them independently.

---

## 1. Choosing the GPU

The trainer calls `torch.device("cuda")` without an index
(`models/TorchTrainer.py`), so it always takes the **first visible** device.
Selection is therefore done with `CUDA_VISIBLE_DEVICES`, which is what makes
manual sharding across GPUs straightforward:

```bash
# run this experiment on GPU 2
CUDA_VISIBLE_DEVICES=2 python reproduce_paper_scripts/30_run_experiments.py ...
```

Check what you have and what is busy before you start:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv
```

One training run occupies one GPU. Two runs on the same GPU will both fit in
memory for these model sizes but will contend for the SMs and each will be
slower than if run in sequence, so one run per GPU is the right granularity.

---

## 2. Preparing the windows

Once per corpus and per window length. This step is CPU-bound and does not touch
the GPU, so run it once before dispatching anything.

```bash
python reproduce_paper_scripts/20_make_windows_and_features.py \
  --config   config_new/create_windows_sentences.yaml \
  --data_dir data_sentences/data_sentences_5_subjects_6_sessions \
  --subjects S01 S02 S03 S04 S05 S06 S07 \
  --conditions silent vocalized \
  --windows_s 2.0
```

| flag | meaning |
|---|---|
| `--config` | window/segmentation YAML from `config_new/` |
| `--data_dir` | corpus root; must already hold `raw_and_processed/` |
| `--windows_s` | one or more window lengths, each written to its own `WIN_<ms>/` |
| `--label_mode` | `sentence` or `word`, overrides the YAML |
| `--manual_features` | `true`/`false`, the handcrafted descriptors for the random-forest baseline |

Output lands under `<data_dir>/wins_and_features/<subject>/<condition>/WIN_<ms>/`.
Recordings already windowed are skipped, so the command is safe to re-run.

**Trigger-free windows** go through a wrapper instead, which runs the detector
first and writes to a separate root so that both datasets coexist:

```bash
./scripts/make_onset_windows.sh --report_only          # score the detector, extract nothing
./scripts/make_onset_windows.sh --windows_s 2.0
./scripts/make_onset_windows.sh --windows_s 2.4
```

---

## 3. Running one experiment

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce_paper_scripts/30_run_experiments.py \
  --base_config   config_new/thesis_base_sentences_w2000_norest.yaml \
  --model_config  config_new/ablation_configs/transformer_stft_ctc_classification_greedy_nearest.yaml \
  --data_dir      data_sentences/data_sentences_5_subjects_6_sessions \
  --artifacts_dir artifacts_thesis/my_run \
  --experiment    global \
  --subjects      S01 S02 S03 S04 S05 S06 S07 \
  --conditions    silent vocalized \
  --plot_loss
```

### The arguments that matter

| flag | meaning |
|---|---|
| `--base_config` | corpus, window length, label set, normalization, CV mode |
| `--model_config` | architecture, objective and decoder |
| `--artifacts_dir` | everything the run writes; **use a fresh directory per run** |
| `--experiment` | `global`, `inter_session`, `inter_session_ft`, `train_from_scratch`, `data_augmentation_ablation`, `session_count_ablation` |
| `--subjects`, `--conditions` | the loop the script runs; this is your unit of parallelism |
| `--pool_subjects` | train one `all_subjects` model instead of one per participant |
| `--inter_session_windows_s` | **required for `inter_session`**, see the trap below |
| `--plot_loss` | per-fold training and validation loss curves |

### One trap worth knowing

`global` takes the window length from `base_config["window"]["window_size_s"]`,
whereas `inter_session` takes it from `--inter_session_windows_s` and **ignores
the base config**. Passing a different value produces a `w2400ms` global run and
a `w2000ms` inter-session run under the same artefact root, with no error
raised, and the analysis step then silently finds only one of them.

Always pass the same value:

```bash
--experiment inter_session --inter_session_windows_s 2.0    # base config says 2.0
```

The `config_new/thesis_base_*.yaml` files exist precisely so that the base
config states the window unambiguously; check it before you launch.

---

## 4. Building the tables

Training writes `cv_summary.csv` per fold; the summary tables are a separate,
CPU-only step, so run it after the GPUs are free.

```bash
python utils/III_results_analysis/I_global_intersession_analysis.py \
  --artifacts_dir artifacts_thesis/my_run \
  --experiment    global \
  --model_name    speechnet_transformer \
  --model_name_id w2000ms \
  --model_run     model_1 \
  --subjects      S01 S02 S03 S04 S05 S06 S07 \
  --conditions    silent vocalized \
  --plot_confusion_matrix --plot_block_scatter
```

`--model_name` must match the `name` field of the model config
(`speechnet`, `speechnet_transformer`, `speechnet_cnn`, `transformer_only`,
`random_forest`) and `--model_name_id` the window (`w2000ms`), otherwise the
script reports `No runs found` and writes nothing.

---

## 5. Decoder sweeps

A sweep costs **one** training run. Train once with the log-probability dump
enabled (the `*_beam_sweep.yaml` model configs do this), then re-decode the
cached posteriors offline. The offline pass is CPU-parallel, so give it cores
rather than a GPU.

```bash
# 1. train, on a GPU
CUDA_VISIBLE_DEVICES=1 python reproduce_paper_scripts/30_run_experiments.py \
  --base_config  config_new/thesis_base_sentences_w2000_norest.yaml \
  --model_config config_new/ablation_configs/transformer_stft_ctc_recognition_beam_sweep.yaml \
  --data_dir     data_sentences/data_sentences_5_subjects_6_sessions \
  --artifacts_dir artifacts_thesis/sweep --experiment global \
  --subjects S01 S02 S03 S04 S05 S06 S07 --conditions silent vocalized

# 2. sweep the grid, on CPU. --dumps is scoped to models/<experiment> so that
#    global and inter_session caches are never pooled.
python offline_experiments/VII_beam_sweep.py \
  --dumps artifacts_thesis/sweep/models/global \
  --out   artifacts_thesis/sweep/beam_sweep_global.csv \
  --beam_widths 1 5 10 25 --temperatures 1.0 1.3 1.6 2.0 \
  --blank_penalties 0.0 1.0 2.0 4.0 --length_bonuses 0.0 0.5 1.0 2.0 --jobs 16

# 3. per-subject tables at the selected configuration
python utils/III_results_analysis/VII_beam_sweep_tables.py \
  --artifacts_dir artifacts_thesis/sweep --experiment global \
  --model_name speechnet_transformer --model_name_id w2000ms --model_run model_1 \
  --subjects S01 S02 S03 S04 S05 S06 S07 --conditions silent vocalized \
  --sweep_csv artifacts_thesis/sweep/beam_sweep_global.csv --jobs 16 \
  --dump_predictions
```

---

## 6. Spreading work over several GPUs

There is no orchestrator, and none is wanted: you decide what goes where. Three
ways to split, in increasing order of how much you have to keep track of.

### By experiment, one run per GPU

The simplest and the one to prefer. Each command is independent and writes to
its own artefact root.

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/reproduce_thesis_results/sentences/10_transformer_classification_greedy.sh > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/reproduce_thesis_results/sentences/14_transformer_recognition_greedy.sh  > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/reproduce_thesis_results/sentences/08_bilstm_classification_greedy.sh    > /dev/null 2>&1 &
```

Each script mirrors its output to `artifacts_thesis/logs/<name>_<timestamp>.log`;
follow one with `tail -f`.

### By protocol, two GPUs per experiment

`global` and `inter_session` are independent and write to different
subdirectories of the same root, so they may run concurrently:

```bash
R=artifacts_thesis/my_run
CUDA_VISIBLE_DEVICES=0 EXPERIMENTS=global        bash scripts/reproduce_thesis_results/sentences/10_transformer_classification_greedy.sh &
CUDA_VISIBLE_DEVICES=1 EXPERIMENTS=inter_session bash scripts/reproduce_thesis_results/sentences/10_transformer_classification_greedy.sh &
```

Build the tables once, after both have finished.

### By participant, up to seven GPUs per experiment

The finest split. Subject-specific runs share nothing, so one participant per
GPU is safe **provided every shard writes to the same artefact root**, which is
what lets the analysis step see them as one experiment:

```bash
R=artifacts_thesis/my_run
i=0
for s in S01 S02 S03 S04 S05 S06 S07; do
  CUDA_VISIBLE_DEVICES=$i python reproduce_paper_scripts/30_run_experiments.py \
    --base_config config_new/thesis_base_sentences_w2000_norest.yaml \
    --model_config config_new/ablation_configs/transformer_stft_ctc_classification_greedy_nearest.yaml \
    --data_dir data_sentences/data_sentences_5_subjects_6_sessions \
    --artifacts_dir "$R" --experiment global \
    --subjects "$s" --conditions silent vocalized \
    > "logs/${s}.log" 2>&1 &
  i=$(( (i + 1) % 4 ))          # number of GPUs you have
done
wait
```

Then run the analysis **once**, over all seven subjects. Do not shard the
analysis: it aggregates across participants and needs to see them all.

**Do not** shard by participant into different artefact roots. The tables would
then be built per participant and the `All` row, which is the figure the thesis
quotes, would be missing.

---

## 7. What a run writes

```
<artifacts_dir>/
├── models/<protocol>/<subject>/<condition>/<model>/<w####ms>/model_1/
│   ├── cv_summary.csv                 one row per fold, every metric
│   ├── run_cfg.json                   the exact configuration used
│   ├── <cv_mode>_fold_<k>.pt          checkpoint of the best epoch
│   └── <cv_mode>_fold_<k>_predictions.csv
├── tables/                            per-subject summaries plus an "All" row
├── tables_beam/                       the same at the sweep-selected beam config
├── beam_sweep_<protocol>.csv          the whole grid, pooled over conditions
└── figures/                           loss curves, confusion matrices
```

`run_cfg.json` is the record of what was actually run and is worth reading
before quoting any number: it holds the resolved window length, label set,
normalization and decoder for that specific fold.

---

## 8. Common failures

| symptom | cause |
|---|---|
| `No runs found for experiment=...` | `--model_name` or `--model_name_id` does not match what training wrote; read `run_cfg.json` |
| global and inter-session tables disagree on the window | `--inter_session_windows_s` differs from the base config, see §3 |
| `CUDA out of memory` with two runs on one GPU | expected; one run per GPU |
| a sweep finds no dumps | the model config was not one of the `*_beam_sweep.yaml` variants, so no log-probabilities were cached |
| inter-session skipped for a participant | fewer than two sessions found under `--data_dir` for that subject and condition |

---

## 9. Where the thesis experiments live

`scripts/reproduce_thesis_results/` wraps the two entry points above into one
script per table and figure of the thesis, with the configuration pinned. Its
`README.md` maps each script to what it produces. The scripts are independent
by design, so they are the natural unit to dispatch across GPUs as in §6.
