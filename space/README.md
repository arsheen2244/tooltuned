---
title: ToolTuned
emoji: 🛠️
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
license: apache-2.0
models:
  - asheeee24/tooltuned-qwen2.5-1.5b
short_description: Qwen2.5-1.5B fine-tuned with QLoRA for function calling
---

# ToolTuned demo

Recorded answers from [`asheeee24/tooltuned-qwen2.5-1.5b`](https://huggingface.co/asheeee24/tooltuned-qwen2.5-1.5b) on five examples: which tools it calls, and when it declines.

This Space is a static page (`index.html`) that shows `examples.json`. Hugging Face needs a paid plan to host a Gradio app, so the live version runs on your own computer instead.

## Files

| File | What it does |
| --- | --- |
| `index.html` | The page this Space shows. |
| `examples.json` | The recorded answers, made by `record.py`. |
| `demo.py` | The examples, plus loading the model and reading its answers. |
| `record.py` | Runs the examples and saves `examples.json`: `python record.py [float32\|bfloat16]`. |
| `app.py` | The live Gradio demo: `python app.py`. |

## Run the live demo

On a CPU it needs about 6 GB of free memory, or 3 GB with `DEMO_DTYPE=bfloat16`:

```bash
pip install -r requirements.txt
python app.py
```

The code lives in the `space/` folder of [github.com/arsheen2244/tooltuned](https://github.com/arsheen2244/tooltuned).
