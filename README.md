## Setup

```bash
pip install -r requirements.txt
# or use uv
```

Set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
```

## Usage

1. **Extract your data**
```bash
python extract_data.py
# or: uv run extract_data.py
```

2. **Prepare the dataset**
```bash
python prepare_dataset.py
# or: uv run prepare_dataset.py
```

3. **Train the model**
```bash
python train.py
# or: uv run train.py
```

4. **Run inference**
```bash
python inference.py
# or: uv run inference.py
```

## Output

Trained LoRA adapters are saved to `./outputs/final_model/`
