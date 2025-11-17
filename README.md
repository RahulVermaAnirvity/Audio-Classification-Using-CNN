# Audio-CNN-CLASSIFICATION

Audio-CNN is an end-to-end demo that trains a residual convolutional neural network on the ESC-50 environmental sound dataset, exposes the model behind a Modal-powered FastAPI endpoint, and visualizes intermediate feature maps, spectrograms, and predictions inside a modern Next.js UI.

The repository is split between the Python training/inference code in the root directory and a dedicated `audio-cnn/` frontend workspace.

## Features

- Residual CNN architecture with mixup, label smoothing, OneCycle LR schedule, and SpecAugment-style masking for robust ESC-50 training.
- Fully managed training/inference workflows packaged as Modal functions with automatic dataset download and volume persistence for checkpoints.
- FastAPI inference endpoint that returns top-3 classes, mel-spectrograms, feature maps from every major block, and a down-sampled waveform for visualization.
- Next.js 15 + Tailwind UI that lets you upload a WAV file, call the API, and inspect predictions, feature maps, spectrograms, and waveform details.
- Example WAV files (`1.wav`, `2.wav`) for quick smoke testing.

## Repository Layout

| Path | Description |
| --- | --- |
| `model.py` | Residual CNN definition and helper blocks. |
| `train.py` | Modal job to download ESC-50, train the model, and persist checkpoints/tensorboard logs. |
| `main.py` | Modal FastAPI endpoint for inference plus a local entrypoint that hits the endpoint with `2.wav`. |
| `requirements.txt` | Python dependencies for training/inference images. |
| `audio-cnn/` | Next.js frontend that calls the Modal endpoint and renders visualizations. |
| `audio-cnn/src/components` | Visualization widgets (`FeatureMap`, `Waveform`, color scales, UI primitives). |

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm 10+
- [Modal](https://modal.com) account and CLI (`pip install modal` + `modal setup`)
- GPU-backed Modal workspace (training uses `A10G`, inference also targets `A10G`)

## Python Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt modal
modal token new             # if you have not authenticated yet
```

## Training on ESC-50

`train.py` packages the whole pipeline inside a Modal function:

1. Downloads ESC-50 once into a shared Modal volume `esc50-data`.
2. Builds mel spectrograms with SpecAugment, applies mixup randomly, trains for 100 epochs.
3. Stores TensorBoard logs and the best checkpoint (`best_model.pth`) in the `esc-model` volume.

Run training remotely from your machine:

```bash
modal run train.py::train
```

Logs stream to your terminal; TensorBoard logs are stored in `/models/tensorboard_logs` inside the Modal volume.

## Inference Service

`main.py` defines a Modal class `AudioClassifier` that:

- Mounts the `esc-model` volume to load `best_model.pth`.
- Converts uploaded WAV bytes to a mel-spectrogram matching the training configuration.
- Returns predictions, feature maps, input spectrogram, and waveform metadata to mimic the frontend expectations.

Deploy the API:

```bash
modal deploy main.py
```

The deploy output shows a FastAPI URL. Use it to populate the frontend `NEXT_PUBLIC_API_URL`. To smoke test locally without the frontend:

```bash
modal run main.py::main   # encodes 2.wav, sends it to the deployed endpoint, prints predictions
```

## Frontend (Next.js UI)

The UI lives inside `audio-cnn/` and expects an HTTPS endpoint from the Modal deploy.

1. Install dependencies:

   ```bash
   cd audio-cnn
   npm install
   ```

2. Create `audio-cnn/.env.local`:

   ```env
   NEXT_PUBLIC_API_URL="https://<modal-endpoint-url>"
   ```

3. Start the dev server:

   ```bash
   npm run dev
   ```

4. Open `http://localhost:3000`, upload a `.wav` file (ESC-50 examples work great), and inspect the rendered feature maps, spectrograms, and waveform.

## Common Workflows

- **Update the frontend only**: work inside `audio-cnn/`, then run `npm run lint`, `npm run typecheck`, and `npm run build` before pushing.
- **Retrain the model**: tweak hyperparameters in `train.py`, re-run `modal run train.py::train`, then redeploy inference so the new checkpoint is mounted.
- **Troubleshoot inference**: run `modal shell main.py::AudioClassifier` and execute ad-hoc code inside the container, or add logging inside `inference()`.

## Troubleshooting

- *Modal volume not found*: ensure the names `esc50-data` and `esc-model` exist (`modal volume list`). Create them manually via `modal volume create ...` if needed.
- *Frontend 500 errors*: confirm `NEXT_PUBLIC_API_URL` points to a live endpoint and that CORS is allowed (Modal FastAPI endpoints default to permissive CORS).
- *Libsndfile errors*: the Modal images install `libsndfile1`, but on local development you might need `brew install libsndfile` (macOS) or `apt install libsndfile1` (WSL/Linux).

## License

This project packages the ESC-50 dataset (MIT License). Review the dataset terms before redistributing models trained on it.

