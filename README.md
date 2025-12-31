# Tiny LLM

This repository contains a compact transformer language model plus the tooling that trains it from raw text and then fine-tunes only its output head with a collaborative particle swarm optimizer (CLPSO). The workflow is intentionally split into two notebooks so that base training and CLPSO head tuning can run independently.

## 2-stage training & CLPSO fine-tuning pipeline

1. **Stage 1 – Base model training (`Network_Train.ipynb`)**
   - Trains the transformer end-to-end on the selected corpus (TinyStories, EMNLP, movie dialogs, etc.).
   - Persists the weights as `checkpoints/mini_llm_checkpoint.pt`, which stores both the model state_dict and the vocabulary metadata required for later runs.
2. **Stage 2 – Head-only CLPSO fine-tuning (`Network_Fine_Tune.ipynb`)**
   - Loads the Stage 1 checkpoint and constructs the base `MiniTransformer` body.
   - Calls `run_clpso()` from `CLPSO_GRAD_script.py`, which freezes the transformer body, initializes a swarm around the current output head, then alternates CLPSO exploration with short gradient refinement steps.
   - Produces `checkpoints/finetuned_llm_clpso.pt`, a head-tuned checkpoint that can be swapped in for inference.

Because CLPSO operates only on the output projection layer (`model.head`), the two-stage design keeps the heavy pre-training reproducible and exposes the faster CLPSO loop for experiments without retraining the whole network.

## How to run

1. **Set up Python** – Use Python 3.10+, create a virtual environment (`python -m venv .venv && source .venv/bin/activate`), and install the dependencies referenced inside the notebooks (PyTorch, numpy, tqdm, Jupyter, etc.).
2. **Prepare data** – Place the text corpora you plan to use (e.g., `TinyStories-train.txt`, `EMNLP_dataset/`, `movie_lines.txt`, `movie_conversations.txt`) in the repository root as shown below, or adjust the paths in the notebooks.
3. **Stage 1: Train the base model**
   - Launch Jupyter (`jupyter lab` or `jupyter notebook`) and open `Network_Train.ipynb`.
   - Configure the dataset + hyperparameters cells you need, then run all cells. The notebook saves the resulting checkpoint to `checkpoints/mini_llm_checkpoint.pt`.
4. **Stage 2: Fine-tune with CLPSO**
   - Open `Network_Fine_Tune.ipynb`.
   - Point the notebook to your base checkpoint path (`checkpoints/mini_llm_checkpoint.pt`).
   - Ensure `CLPSO_GRAD_script.py` is importable, then run the notebook. It loads the checkpoint, invokes `run_clpso(...)`, freezes the backbone, and optimizes only the head parameters until convergence or patience runs out.
   - The fine-tuned head is stored as `checkpoints/finetuned_llm_clpso.pt`.
5. **Resume / inference** – When you want to resume experiments or run inference, load the desired checkpoint from `checkpoints/` and construct `MiniTransformer` with the matching vocab size (see `model.py`).

## Project structure

```
Tiny_LLM/
├─ Network_Train.ipynb           # Stage 1 notebook: trains the base MiniTransformer
├─ Network_Fine_Tune.ipynb       # Stage 2 notebook: CLPSO head tuning via run_clpso()
├─ CLPSO_GRAD_script.py          # CLPSO + gradient refinement implementation
├─ model.py                      # MiniTransformer architecture definition
├─ EMNLP_dataset/                # Optional EMNLP training data
├─ TinyStories-train.txt         # TinyStories corpus (plain text)
├─ movie_lines.txt               # Cornell Movie-Dialogs Corpus (lines)
├─ movie_conversations.txt       # Cornell Movie-Dialogs Corpus (conversations)
├─ checkpoints/                  # Ignored directory for *.pt checkpoints (see below)
└─ ... other helper files / datasets referenced by the notebooks
```

## Checkpoints & artifacts

- The repository keeps learned weights under `checkpoints/` (ignored by Git) so local experiments do not bloat history. Drop any `.pt` snapshots you want to keep or share into that directory.
- To recreate `mini_llm_checkpoint.pt`, rerun Stage 1. To recreate `finetuned_llm_clpso.pt`, point Stage 2 at the base checkpoint and rerun the CLPSO notebook.
- If you need to distribute checkpoints, upload them to an artifact store (e.g., release assets, object storage) and document the download link; otherwise, expect collaborators to regenerate them with the steps above.
