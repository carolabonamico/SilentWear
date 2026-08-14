# SilentWear: Silent Speech Decoding from Surface EMG

This repository is a fork of [pulp-bio/SilentWear](https://github.com/pulp-bio/SilentWear),
extended for the master's thesis *From Biosignals to Words: Exploiting Novel Deep
Learning Architectures for Speech Understanding* (ETH Zurich, 2026). The upstream
project decodes eight isolated commands from surface electromyography (sEMG). This
fork carries the task to connected speech, decoding twenty sentences either as
closed-set classes or as free character strings, and adds the acquisition of a
paired EEG modality.

Fork: <https://github.com/carolabonamico/SilentWear>

## Contributors

The SilentWear system was developed at ETH Zurich by the
[PULP-Bio](https://iis-projects.ee.ethz.ch/index.php?title=Biomedical_Circuits,_Systems,_and_Applications)
team. The upstream contributors are listed in the
[original repository](https://github.com/pulp-bio/SilentWear).

The work in this fork was carried out by **Carola Bonamico** as a contributor to
the project. It covers the sentence-level corpus acquisition with the paired EMG
and EEG setup, the CTC training and decoding path, the sequence stages compared in
the thesis, the evaluation protocols and metrics, and the ablation studies.
Supervision by Giusy Spacone (ETH Zurich), Prof. Alessio Burrello (Politecnico di
Torino) and Prof. Luca Benini (ETH Zurich).

## System Components

**BioGAP-Ultra**: ultra-low-power acquisition platform for biopotentials. Two
units are used simultaneously, one for the EMG neckband and one for the EEG
headband. Hardware and firmware: <https://github.com/pulp-bio/BioGAP>

**SilentWear neckband**: 14-channel differential dry-electrode EMG neckband.
System overview: <https://ieeexplore.ieee.org/abstract/document/11330464>
(arXiv: <https://arxiv.org/abs/2509.21964>)

**EEG headband**: dry-electrode headband with Datwyler SoftPulse electrodes,
Brush Medium on the measurement channels and Dome on bias and reference, driven by
a second BioGAP-Ultra unit. The EEG is recorded in parallel with every utterance
and is retained for future EMG and EEG fusion. The decoding pipeline of the thesis
reads the EMG alone.

**BioGUI**: Qt application for acquisition, stimulus presentation and labelling.
The version used in this work is the fork
<https://github.com/carolabonamico/biogui>

## What This Fork Adds

* A sentence corpus of twenty lexically overlapping commands plus a rest class,
  recorded from seven participants over six sessions in vocalized and silent
  conditions, alongside the fifteen-word subset.
* A CTC training and decoding path over character tokens, with greedy best-path
  and prefix beam search. The beam carries a temperature, a blank penalty and a
  length bonus, and an offline sweep re-decodes cached posteriors without
  retraining.
* Two sequence stages compared at equal parameter count, a two-layer BiLSTM and a
  pre-norm Transformer encoder, on top of the shared convolutional backbone, which
  may also be run on its own.
* Two readings of the same posteriors, closed-set classification against the
  lexicon and free-character continuous recognition scored by WER and CER,
  including a vocabulary-constrained variant.
* In-model front-ends selecting the input domain (time, STFT, MFCC) inside the
  network, so that one windowed dataset on disk serves all three.
* Ablations on the number of enrolment sessions and on the sliding-window
  augmentation, and pooled multi-subject models with amplitude normalization.

## Results at a Glance

Sentence task, seven participants, six sessions, 2.0 s cue-anchored windows, rest
modelled (21 classes), STFT front-end and CTC objective. Accuracies are balanced
accuracies and error rates are balanced vocabulary-constrained WER, averaged over
folds and then over participants. `G` is the global protocol and `I` the
inter-session one.

| task and model | G voc | G sil | I voc | I sil |
|---|---|---|---|---|
| classification, BiLSTM | 94.0 | 87.8 | 83.9 | 77.9 |
| classification, Transformer | 94.2 | 90.7 | 85.8 | 82.1 |
| recognition, BiLSTM (beam) | 6.6 | 11.5 | 15.9 | 21.5 |
| recognition, Transformer (beam) | 8.2 | 12.7 | 18.4 | 21.0 |

The two tasks select different sequence stages on the same posteriors. The
encoder is ahead on the closed set in all four settings, and the recurrent block
is ahead on free text in three of the four. The trigger-free detector, scored as a
binary classifier of speech against rest over the 84 sentence recordings, reaches
97.3 % recall and 93.6 % specificity at 0.70 spurious events per minute.

## Environment Setup

```bash
conda create -n silent_wear python=3.11.9
conda activate silent_wear
git clone https://github.com/carolabonamico/SilentWear.git
cd SilentWear
pip install -r requirements.txt
```

## Data

The word-level corpus published with the upstream paper is available at
<https://huggingface.co/datasets/PulpBio/SilentWear> and is used in the thesis to
reproduce the SpeechNet baseline. The sentence corpus recorded for the thesis is
not public.

Recordings are read as `.bio` files written by BioGUI and converted to windows by
the two entry points below. Data paths are set in the configuration files of
`config_thesis/`, which is the configuration folder of the thesis; the older
`config/` folder belongs to the upstream pipeline.

```bash
# .bio recordings -> filtered, labeled HDF5 tables
python utils/I_data_preparation/data_preparation.py --data_dir <RAW_DIR>

# HDF5 tables -> fixed-length windows (and, optionally, handcrafted features)
python reproduce_paper_scripts/20_make_windows_and_features.py \
  --config config_thesis/create_windows_sentences.yaml \
  --data_dir <RAW_DIR> --windows_s 2.0 --label_mode sentence
```

The window length, the label mode and the sliding-window augmentation are
declared in `config_thesis/create_windows*.yaml`. The flags `--windows_s` and
`--label_mode` override the file when a single dataset is being produced.

### Where the data has to live

The reproduction scripts of `scripts/reproduce_thesis/` read three corpus roots,
each with a default name and an environment variable that overrides it. The
defaults are declared in `scripts/reproduce_thesis/common.sh`, so a corpus kept
elsewhere needs no edit to any script.

| root | override | holds |
|---|---|---|
| `data_sentences/` | `DATA_SENTENCES` | the sentence corpus of the thesis: 7 participants, 6 sessions, 20 sentences |
| `data_words/` | `DATA_WORDS` | the word corpus: 15 command words plus rest |
| — | `DATA_WORDS_PUBLISHED` | the published SilentWear corpus, for the baseline reproduction of Section 4.2 |

The ablations of Section 4.8 have no root of their own. They read the word corpus
and restrict themselves through `--subjects`, which the scripts set to S01, S03
and S04.

Inside a root the layout is fixed, and the folder names are the ones the
configurations select through `paths.processed` and `paths.win_and_feats`:

```
<root>/
  raw/<subject>/<condition>/*.bio                     # optional, the unfiltered recordings
  raw_and_processed/<subject>/<condition>/*.h5        # filtered, labelled tables
  wins_and_features/<subject>/<condition>/WIN_<ms>/   # cue-aligned windows, one folder per window length
  wins_and_features_onset/<subject>/<condition>/WIN_<ms>/   # trigger-free windows, sentences only
```

Three rules are easy to get wrong:

- **The window length is a subfolder, not a root.** `WIN_2000` and `WIN_2400`
  live side by side under the same `wins_and_features*`, so a second extraction
  at another `--windows_s` is additive and touches nothing already there.
- **Trigger-free windows never share a root with cue-aligned ones.** The onset
  extraction writes to `wins_and_features_onset/`, selected by the
  `thesis_base_sentences_onset_*` configurations. Pointing it at
  `wins_and_features/` overwrites the cue-aligned corpus. Those windows carry no
  rest class, the extraction emitting none.
- **An ablation never writes into a corpus root.** The augmentation ablation
  re-windows the filtered recordings once per point of the sweep, in its own
  workspace under `<artifacts_dir>/.ablation_working_data/<run_label>/`, where
  only `raw_and_processed/` is a symlink back to the corpus. Every window it
  produces is written under that workspace, so the root it reads stays untouched.

The `data/` folder holds datasets from the earlier iteration of this work, some
of them pre-augmented one dataset per sweep point. No script under
`scripts/reproduce_thesis/` reads it.

## Aligning the EMG and EEG Recordings

The two BioGAP-Ultra units write independent `.bio` files with independent clocks,
so a session is a pair of recordings that has to be aligned before the two
modalities can be read on a common timeline. The scripts of
`utils/V_data_alignment/` align the pair and then measure the delay that is left.

`inter_file_alignment.py` first repairs each file on its own, trimming every
signal to the window covered by its hardware timestamps and filling the packets
lost by the hardware with NaNs. It then maps the second file onto the first
through the trigger sequence the two share. The clock offset is computed at every
matching rising edge and interpolated in between, so the drift is followed and not
only the constant offset, and the raw samples are never resampled.

```bash
python utils/V_data_alignment/inter_file_alignment.py \
  emg_mic_test_<N>_<TS>.bio eeg_mic_test_<N>_<TS>.bio <OUT_DIR> --debug
```

`compute_peak_delay.py` measures the residual delay on a periodic stimulus
recorded by both boards and by the microphone of each. It detects the onsets of
every available signal by high-pass filtering, rectification and thresholding, at
most one onset per period, clusters the onsets closer than half a period into a
single event, and writes the per-event delays with their mean and standard
deviation to a CSV, excluding the outliers above `OUTLIER_THRESHOLD_MS`.

```bash
python utils/V_data_alignment/compute_peak_delay.py \
  <OUT_DIR>/emg_mic_test_<N>_<TS>_inter_aligned.bio \
  <OUT_DIR>/eeg_mic_test_<N>_<TS>_inter_aligned.bio \
  --output-dir utils/V_data_alignment/results/b2
```

The same command runs on the raw files, and comparing the two CSVs is what shows
the effect of the alignment. `utils/V_data_alignment/results/b1` to `results/b4`
hold both for the four test blocks recorded on 2026-05-21. The acquisition
parameters, that is the channel and the threshold of each signal, the stimulus
period and the outlier threshold, are the constants at the top of
`compute_peak_delay.py` and have to match the protocol of the recording. The
intra-file repair and the packet-loss report also run on their own:

```bash
python utils/V_data_alignment/align_bio_signals.py <FILE>.bio <OUT_DIR> --debug
python utils/V_data_alignment/check_packet_loss.py <FILE>.bio
```

## Running the Experiments

Every run composes a **base** configuration, which pins the corpus, the window
length, the label set and the cross-validation scheme, with a **model**
configuration, which pins the architecture, the input domain and the objective.
The base configurations used in the thesis are the
`config_thesis/thesis_base_*.yaml` files, and the model configurations live in
`config_thesis/models_configs/`, named
`<corpus>_<domain>_<objective>_<sequence>[_<task>_<decoder>].yaml`. See
`config_thesis/README.md` for the full map from configuration to result.

The scripts under `scripts/reproduce_thesis/` wrap the commands below, one per
step of the chain, and `run_all.sh` executes them in order. The folder `single/`
holds one script per experiment unit, for running them separately. The commands
are given here directly so that a single experiment can be run without the
wrappers.

### Global and inter-session protocols

The `--experiment` flag selects the protocol. `global` is five-fold stratified
cross-validation over the whole corpus of a participant, `inter_session` is
leave-one-session-out over the recording sessions.

BiLSTM, STFT front-end, CTC, sentence classification:

```bash
python reproduce_paper_scripts/30_run_experiments.py \
  --base_config config_thesis/thesis_base_sentences_w2000_rest.yaml \
  --model_config config_thesis/models_configs/sentences_stft_ctc_bilstm_classification_greedy.yaml \
  --data_dir <DATA_DIR> --artifacts_dir artifacts_sentences \
  --experiment global --subjects S01 S02 S03 S04 S05 S06 S07 \
  --conditions silent vocalized
```

Transformer sequence stage, same protocol:

```bash
python reproduce_paper_scripts/30_run_experiments.py \
  --base_config config_thesis/thesis_base_sentences_w2000_rest.yaml \
  --model_config config_thesis/models_configs/sentences_stft_ctc_transformer_classification_greedy.yaml \
  --data_dir <DATA_DIR> --artifacts_dir artifacts_sentences \
  --experiment inter_session
```

The three input domains are selected by the model configuration alone. The matrix
`<corpus>_<domain>_<objective>_<sequence>.yaml` covers {words, sentences} × {time,
stft, mfcc} × {cross-entropy, CTC} × {none, BiLSTM}, so a domain comparison is
three runs differing in one field:

```bash
for DOMAIN in time stft mfcc_b64_q40; do
  python reproduce_paper_scripts/30_run_experiments.py \
    --base_config config_thesis/thesis_base_sentences_w2000_rest.yaml \
    --model_config config_thesis/models_configs/sentences_${DOMAIN}_ctc_bilstm_classification_greedy.yaml \
    --data_dir <DATA_DIR> --artifacts_dir artifacts_domains \
    --experiment global
done
```

Continuous recognition is the same run with the decoding key switched from
`lexicon` to `recognition` in the model configuration, which widens the head to
the fixed English alphabet and scores the run by WER and CER instead of accuracy.
The `*_recognition_*.yaml` files of `config_thesis/models_configs/` are those
variants.

### Prefix beam search sweep

The sweep re-decodes the per-frame log-probabilities cached beside every fold
checkpoint, so nothing is retrained and the acoustic model is held fixed. Train
once with the dump enabled, then sweep:

```bash
python offline_experiments/VII_beam_sweep.py \
  --dumps artifacts_beam_sweep/<RUN>/models/global \
  --lexicon lexicon/silentwear_lexicon_sentences.txt \
  --beam_widths 1 5 10 25 \
  --temperatures 1.0 1.3 1.6 2.0 \
  --blank_penalties 0.0 1.0 2.0 4.0 \
  --length_bonuses 0.0 0.5 1.0 2.0 \
  --out artifacts_beam_sweep/<RUN>/beam_sweep_global.csv
```

The grid is scoped to one protocol at a time so that the global and inter-session
caches are never pooled. The selected operating point is written per condition to
`tables_beam/`, and the greedy reference is recomputed from the same cache.

### Ablation on the number of enrolment sessions

Retrains on the first 1 to 6 sessions of each participant. The inter-session
protocol is defined from two sessions onwards.

```bash
python reproduce_paper_scripts/30_run_experiments.py \
  --base_config config_thesis/thesis_base_words_w1400_rest.yaml \
  --model_config config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
  --data_dir <DATA_WORDS> --artifacts_dir artifacts_ablation/session_count \
  --experiment session_count_ablation \
  --subjects S01 S03 S04 --conditions silent vocalized \
  --session_windows_s 1.4 --min_sessions 1
```

### Ablation on the sliding-window augmentation

Two sweeps vary one parameter at a time against an un-augmented baseline, the
stride and the number of shifts per side. Both are declared in the windowing
configuration, so each point of the sweep is one windowed dataset and one run:

```bash
# Set data_augmentation.stride_ms to 10, 20, 50 or 100 in the windowing
# configuration, at num_strides: 2, before each point of the sweep.
python reproduce_paper_scripts/20_make_windows_and_features.py \
  --config config_thesis/create_windows_words.yaml \
  --data_dir <DATA_WORDS> --windows_s 1.4 --label_mode word

python reproduce_paper_scripts/30_run_experiments.py \
  --base_config config_thesis/thesis_base_words_w1400_rest.yaml \
  --model_config config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
  --data_dir <DATA_WORDS> --artifacts_dir artifacts_ablation/stride50_n2 \
  --experiment session_count_ablation --subjects S01 S03 S04 \
  --session_windows_s 1.4 --min_sessions 1
```

The shift-count sweep is the same with `num_strides` in {2, 5, 10} at a 10 ms
stride. Setting the augmentation consumption mode to `original_size` in the
windowing configuration resamples the augmented pool back to the cardinality of
the base split, which separates the quantity of the added windows from their
distribution.

The shell wrappers for both ablations are
`scripts/reproduce_thesis/single/60_ablation_session_count.sh` and
`scripts/reproduce_thesis/single/60_ablation_augmentation.sh`. They are the only
experiments restricted to three participants; every step of the chain runs on all
seven.

### Pooled multi-subject models

One model is trained on all seven participants at once, with no amplitude
normalization, with a per-subject z-score or with a per-subject min-max scaling.
The three base configurations differ in that field alone, and the runs are
`60_pooled_none.sh`, `60_pooled_zscore.sh` and `60_pooled_minmax.sh` under
`scripts/reproduce_thesis/single/`.

## Where the Results Land

Per-fold metrics are written to a `cv_summary.csv` beside every set of
checkpoints, and every scalar of the metrics dictionary becomes a column. The
aggregate tables land under `tables/`, with the pooled figure in the `All` row.
The folders of `artifacts_thesis/` follow the scripts of
`scripts/reproduce_thesis/`, one per decision axis:

| folder | holds | thesis |
|---|---|---|
| `01_gate_baseline_published/` | SpeechNet reproduced on the published corpus, CE and CTC, plus the STFT and BiLSTM upgrade | Table 4.1, upper block |
| `02_gate_baseline_new_corpus/` | the same models on the word subset recorded here | Table 4.1, lower block |
| `10_axis1_input_domain/` | four input domains × two sequence stages, words and sentences, CE | Table 4.2 |
| `11_axis1_ctc_domains/` | the time and mel rows of the same table under CTC | Table 4.2, CTC rows |
| `20_axis2_objective/` | the STFT rows under CTC, words and sentences | Table 4.2, CTC rows |
| `30_axis3_sequence_stage/` | backbone alone, BiLSTM and Transformer, both tasks, plus the mel counterparts | Tables 4.4 and 6.1 |
| `40_axis4_decoder_sweep/` | the beam grid re-decoded offline, both tasks and both architectures | Tables 4.5 and 6.2 |
| `50_axis5_trigger_free/` | models retrained on onset-anchored windows, kept as an exploratory run and not reported in the thesis | — |
| `60_ablations/` | enrolment sessions, sliding-window augmentation, pooled models | Figures 4.4 and 4.5, Table 4.6 |
| `embeddings/` | the encoder activations projected for the appendix | Appendix H |

## Analysing the Results

The analysis scripts detect the run mode from the columns of `cv_summary.csv`,
`balanced_accuracy` for classification and `wer` for recognition, and skip the
metrics a run did not produce, so older artefacts stay readable.

```bash
# per-subject and pooled tables, plus confusion matrices
python utils/III_results_analysis/I_global_intersession_analysis.py \
  --artifacts_dir artifacts_sentences --experiment global \
  --model_name speechnet_transformer --model_name_id w2000ms \
  --plot_confusion_matrix

# the sweep-selected beam configuration, per condition and protocol
python utils/III_results_analysis/VII_beam_sweep_tables.py \
  --artifacts_dir artifacts_beam_sweep/<RUN> --experiment global \
  --model_name speechnet_transformer --model_name_id w2000ms

# ablation figures
python utils/IV_plots/plot_ablation_results.py \
  --artifacts_root artifacts_ablation --out_dir figures

# the 21-class sentence results, greedy against the selected beam
python utils/III_results_analysis/aggregate_rest_sentence_results.py

# the same table for the 20-class runs, or at another window
python utils/III_results_analysis/aggregate_rest_sentence_results.py \
  --root artifacts_beam_sweep_no_rest --window w2000ms
```

The trigger-free detector is scored on its own, against the trigger and never
against a model, by `utils/I_data_preparation/onset_detection_report.py`, which
writes one row per recording with the detection yield and the binary rates.

## Extending the Pipeline

To add a model, place its configuration under `config_thesis/models_configs/`,
implement it under `models/cnn_architectures/` and register it in
`models/models_factory.py`. Task behaviour, that is the loss and the decoding, is
owned by the strategies in `models/strategies.py`, so a new objective is a new
strategy rather than a change to the training loop.

## Citation

If you use this work, please cite the SilentWear system and the platform it runs
on:

```bibtex
@online{spacone_silentwear_26,
  author = {Spacone, Giusy and Frey, Sebastian and Pollo, Giovanni and Burrello, Alessio and Pagliari, J. Daniele and Kartsch, Victor and Cossettini, Andrea and Benini, Luca},
  title = {SilentWear: An Ultra-Low Power Wearable System for EMG-Based Silent Speech Recognition},
  year = {2026},
  url = {coming soon}
}
```

```bibtex
@inproceedings{meier_wearneck_26,
  author={Meier, Fiona and Spacone, Giusy and Frey, Sebastian and Benini, Luca and Cossettini, Andrea},
  booktitle={2025 IEEE SENSORS},
  title={A Parallel Ultra-Low Power Silent Speech Interface Based on a Wearable, Fully-Dry EMG Neckband},
  year={2025},
  pages={1-4},
  doi={10.1109/SENSORS59705.2025.11330464}}
```

```bibtex
@article{frey_biogapultra_26,
  author={Frey, Sebastian and Spacone, Giusy and Cossettini, Andrea and Guermandi, Marco and Schilk, Philipp and Benini, Luca and Kartsch, Victor},
  journal={IEEE Transactions on Biomedical Circuits and Systems},
  title={BioGAP-Ultra: A Modular Edge-AI Platform for Wearable Multimodal Biosignal Acquisition and Processing},
  year={2026},
  pages={1-17},
  doi={10.1109/TBCAS.2026.3652501}}
```

## License

* Apache License 2.0, see [LICENSE](LICENSE).
* Images under `extras/` are released under the Creative Commons Attribution 4.0
  International License, see [LICENSE_IMG](LICENSE.images).
