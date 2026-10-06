# 🎙️ VoiceBot — Voice-Enabled Chatbot (Speech Recognition + Deep Learning)

Speak a question → **Whisper** (speech recognition, runs in the browser) converts it to text →
a **BiLSTM + attention** deep-learning model (trained on CLINC150, 150 intents + out-of-scope)
classifies the intent on a Python API → the bot shows the recognized text, the intent, the
confidence and a reply, and speaks the reply aloud.

```
Browser                                              Vercel serverless (Python)
┌───────────────────────────────────────┐   text    ┌──────────────────────────────────┐
│ mic → MediaRecorder → 16 kHz audio    │ ────────▶ │ /api/chat                        │
│ → Whisper (transformers.js, ONNX)     │           │  tokenize → BiLSTM-attention     │
│ → recognized text                     │ ◀──────── │  (NumPy) → intent + confidence   │
│ show text, intent, reply; speak reply │   JSON    │  → response module → reply       │
└───────────────────────────────────────┘           └──────────────────────────────────┘
```

## Project structure
```
index.html            web page (UI, recording, Web Speech fallback, text-to-speech)
whisper-worker.js     Web Worker running Whisper in the browser
api/chat.py           Vercel function: POST {"text": ...} → intent + reply
api/_lib/intent_np.py BiLSTM-attention forward pass in pure NumPy
api/_lib/responses.py intent → response logic
api/_model/           exported weights (weights.npz), vocabulary + labels (meta.json), metrics
dev_server.py         run everything locally
requirements.txt      runtime dependency (numpy only)
vercel.json           Vercel function config
training/             PyTorch training / evaluation code, dataset, trained model, plots
```

## Run it locally in VS Code
1. Unzip the project and open the `voicebot` folder in VS Code (**File → Open Folder**).
2. Open a terminal (**Terminal → New Terminal**) and run:
   ```bash
   python -m venv .venv
   # Windows:        .venv\Scripts\activate
   # macOS / Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   python dev_server.py
   ```
3. Open **http://localhost:8000** in Chrome or Edge, allow the microphone, click the mic and speak.
   The first time, the Whisper model (~40 MB) downloads into the browser; after that it's cached.

## Deploy on Vercel
**Option A — GitHub (recommended)**
1. Push this folder to a new GitHub repository.
2. On vercel.com → **Add New… → Project** → import the repo.
3. Framework preset: **Other**. Leave build/output settings empty → **Deploy**.

**Option B — Vercel CLI**
```bash
npm i -g vercel
vercel          # first time: log in and accept the defaults
vercel --prod   # prints the public https://….vercel.app link
```
The mic only works over HTTPS, which Vercel provides automatically.

## Retrain the model (optional)
```bash
cd training
pip install -r requirements.txt
python -m spacy download en_core_web_md
python train.py          # ~20 min on CPU → artifacts/
python evaluate.py
python tune_oos.py       # tune the out-of-scope rule on the validation split
python export_numpy.py   # export to ../api/_model and evaluate the deployed path
```

## Results (CLINC150 test set, 4,500 in-scope + 1,000 out-of-scope utterances)
| Model | In-scope accuracy | OOS recall | Macro-F1 |
|---|---|---|---|
| TF-IDF + logistic regression (baseline) | 91.0% | 15.2% | 83.8% |
| BiLSTM-attention (argmax) | 91.2% | 21.2% | 84.7% |
| **BiLSTM-attention + OOS rule (deployed)** | **89.4%** | **63.0%** | **88.3%** |

Dataset: Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP 2019.
