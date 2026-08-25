# Context-Aware Sign-to-Speech Communication System

A webcam-based system that recognizes sign-language gestures, understands
their meaning, and speaks natural, context-aware English sentences —
demonstrated through a job-interview scenario.

```
SIGN LANGUAGE → CAMERA → MEDIAPIPE → LSTM → SIGN TOKENS
   → COMMUNICATION ENGINE → CONVERSATION CONTEXT → AI/NLP
   → NATURAL ENGLISH → TEXT-TO-SPEECH → 🔊 SPEECH
```

---

## 0. What you got / project layout

```
ContextAwareHCI/
├── dataset/                  # your training data lives here (one folder per sign)
├── collect_data.py           # collect ONE sign live via webcam
├── collect_all_data.py       # collect the WHOLE vocabulary in one session, live
├── app_upload.py             # ⭐ web page: pick a sign, upload a recorded video, done
├── templates/upload.html     #    (the page app_upload.py serves)
├── check_dataset.py          # verify dataset health before training
├── train_model.py            # train the LSTM and save the model
├── hand_landmarkers.py       # MediaPipe Tasks API wrapper
├── video_to_sequences.py     # turns any video into (30,63) training samples
├── gesture_recognizer.py     # loads trained model, streaming predictions
├── communication_engine.py   # holds current sign tokens (no hardcoded phrases)
├── sentence_generator.py     # tokens -> natural English (AI + offline fallback)
├── conversation_context.py   # conversation history + mode (GENERAL/INTERVIEW)
├── intent_engine.py          # rule-based intent classification (metadata only)
├── tts_engine.py             # text-to-speech wrapper
├── live_sign_test.py         # ⭐ the full end-to-end live demo
├── config.py                 # all shared paths/constants — edit vocabulary here
├── tests/                    # small test scripts for each module
├── requirements.txt
└── .gitignore
```

Two things are **not** included in this download and you must get yourself:
1. **`hand_landmarker.task`** — the MediaPipe hand-tracking model file (step 2 below).
2. **A webcam / recorded videos** — the actual sign data, obviously.

---

## 1. Setup (do this once)

```bash
cd ContextAwareHCI

# create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 2. Download the MediaPipe hand model (do this once)

```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

(No `curl`? Just open that URL in a browser and save the file as
`hand_landmarker.task` in the project root, next to `config.py`.)

## 3. (Optional) Enable AI-generated sentences

The system works fully **offline** with a deterministic sentence builder.
For richer, more natural sentences, set an Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Windows: set ANTHROPIC_API_KEY=sk-ant-...
```

No key? No problem — `sentence_generator.py` automatically falls back to
offline mode (see spec section 34), everything else still works.

---

## 4. Building your dataset — you asked specifically for an upload-based way to do this

You have **two ways** to add training samples. Use whichever is easier for
each contributor:

### Option A — Upload recorded videos (recommended for you: "record and upload")

```bash
python app_upload.py
```

Then open **http://127.0.0.1:5000** in a browser (works on your phone too,
if it's on the same network — use your computer's local IP instead of
127.0.0.1, e.g. `http://192.168.1.23:5000`).

On that page:
1. **Select the sign** you're recording from the dropdown (HELLO, YES, NO, …).
2. Optionally type your name/contributor ID.
3. **Choose/record a video** of yourself doing that sign — any camera app,
   a few seconds, hand clearly visible. You can even repeat the sign 2–3
   times in one clip; the app automatically slides a window over the
   footage and generates *multiple* training samples from a single upload.
4. Click **Upload & Process**. It's saved straight into `dataset/<LABEL>/`
   as ready-to-train `.npy` files. The page shows a running count per sign.

Repeat for each sign, ideally 30+ samples each, ideally from more than one
person (see "Dataset Quality" below).

### Option B — Live webcam collection (if you'd rather record directly in Python)

```bash
# one sign at a time
python collect_data.py --label HELLO --person person1 --samples 30

# the whole vocabulary in one sitting
python collect_all_data.py --person person1 --samples 30
```
Press `SPACE` to record each sample, `q` to quit/skip.

### Check your dataset anytime

```bash
python check_dataset.py
```
This reports per-sign sample counts, missing signs, and any malformed files.

**Dataset quality tip:** don't rely on just one person. A stronger dataset
is 15 signs × multiple people × 30–50 samples each, so the model doesn't
overfit to one person's hand shape.

---

## 5. Train the model

Once `check_dataset.py` looks healthy:

```bash
python train_model.py
```

This trains the LSTM, evaluates it on a held-out test split (prints
accuracy/precision/recall/confusion matrix), and saves:
- `sign_model.keras`
- `label_classes.npy`

**Important:** don't retrain every time one person adds a few files. Merge
everyone's dataset contributions first (see the Git workflow below), then
train once on the complete dataset.

---

## 6. Run the full live demo

```bash
# general conversation mode
python live_sign_test.py

# force offline sentence generation (skip the AI call)
python live_sign_test.py --no-ai
```

Controls while it's running:
- Just sign — recognized tokens accumulate and are shown on screen.
- `c` — clear the current tokens (start a fresh message).
- `ENTER` — generate a natural sentence from the current tokens, show the
  detected intent, and speak it out loud.
- `q` — quit.

---

## 7. Individual module tests

```bash
python tests/camera_test.py
python tests/gesture_test.py
python tests/communication_test.py
python tests/context_test.py
python tests/intent_test.py
```

Recommended order when something's not working: camera → MediaPipe/landmarks
→ dataset → LSTM → tokens → communication → NLP → context → TTS. Debug one
stage at a time rather than the whole pipeline at once.

---

## 8. Extending the vocabulary for the interview demo

`config.py` already lists the interview-specific vocabulary
(`STUDENT`, `COMPUTER_SCIENCE`, `AI`, `PYTHON`, `MACHINE_LEARNING`, etc.)
under `INTERVIEW_VOCABULARY`, but it's **not** active yet — get the core
15-sign pipeline fully working first. When you're ready:

1. In `config.py`, change:
   ```python
   ALL_VOCABULARY = CORE_VOCABULARY  # change to:
   ALL_VOCABULARY = CORE_VOCABULARY + INTERVIEW_VOCABULARY
   ```
2. Re-run `app_upload.py` / `collect_data.py` to gather samples for the new
   signs (the upload dropdown will automatically include them).
3. `python check_dataset.py` → `python train_model.py` again.

---

## 9. Git workflow (multiple people contributing data)

```bash
git clone <your-repo-url>
git checkout -b dataset-person2

# collect data (Option A or B above)
python check_dataset.py

git add dataset/
git commit -m "Add sign language dataset samples"
git push -u origin dataset-person2
# then open a Pull Request on GitHub and merge into main
```

**Very important rule:** don't let different people independently retrain
and commit `sign_model.keras` / `label_classes.npy` while dataset
collection is still in progress — those files aren't tracked by `.gitignore`
by default in some setups, so double check. The dataset is the shared
source of truth; train the final model once, after everyone's data is merged.

---

## 10. Troubleshooting

| Problem | Likely fix |
|---|---|
| `FileNotFoundError: hand_landmarker.task` | Redo step 2 above. |
| Webcam won't open | Close other apps using the camera; try `--label` scripts with camera index changed in code if you have multiple cameras. |
| Upload says "no usable hand movement" | Better lighting, keep hand in frame the whole clip, try a shorter/simpler clip. |
| `train_model.py` complains about missing classes | Run `check_dataset.py`, fill in the 0-sample signs before training. |
| AI sentences not appearing | Check `ANTHROPIC_API_KEY` is set and the `anthropic` package installed; it silently falls back to offline mode otherwise. |
| No sound from TTS | `pyttsx3` needs a working audio backend on your OS; if unavailable the app prints the sentence instead of crashing. |

---

## Architecture principle to keep in mind

```
RECOGNITION ≠ LANGUAGE GENERATION ≠ RESPONSE
```

What did the user sign? → What did they mean? → How should that be phrased
naturally? → (optionally) what should the system say back? Keep these
separate and the project stays easy to explain, test, and extend.
