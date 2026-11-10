"""Live Gradio demo: give the fine-tuned model some tools and a request, and see what it calls.

The hosted Space shows recorded answers instead (index.html), because Hugging Face needs a
paid plan to host Gradio. Run this one yourself with: python app.py
Set DEMO_DTYPE=bfloat16 to halve the memory needed (about 3 GB instead of 6 GB).
"""

import json
import os
import threading

import gradio as gr

from demo import EXAMPLES, MODEL_ID, answer, load_model, parse_tool_calls, parse_tools, summarize

tokenizer, model = load_model(os.environ.get("DEMO_DTYPE", "float32"))
lock = threading.Lock()


def generate(tools_text: str, query: str) -> tuple[str, list, str]:
    try:
        tools = parse_tools(tools_text)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    if not query.strip():
        raise gr.Error("Type a request for the model.")
    with lock:
        raw = answer(tokenizer, model, tools, query.strip())
    return summarize(raw), parse_tool_calls(raw), raw


INTRO = f"""
# ToolTuned: function calling with a 1.5B model

[`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) is Qwen2.5-1.5B-Instruct fine-tuned with QLoRA to call tools.
Describe some tools, type a request, and the model either calls tools or answers in plain text.

On BFCL it beats the base model at calling tools (for example 78.0% vs 74.5% on `parallel`), but it is
**worse at noticing when no tool fits** (57.9% vs 72.5% on `irrelevance`). The last example shows this.
Code and full results: [github.com/arsheen2244/tooltuned](https://github.com/arsheen2244/tooltuned).

On a CPU an answer takes about 20–30 seconds.
"""

with gr.Blocks(title="ToolTuned demo") as page:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            tools_box = gr.Code(
                label="Tools (JSON list)",
                language="json",
                value=json.dumps(EXAMPLES[0][1], indent=2),
                lines=14,
            )
            query_box = gr.Textbox(label="Request", value=EXAMPLES[0][2], lines=2)
            run = gr.Button("Run", variant="primary")
        with gr.Column():
            summary_box = gr.Textbox(label="Result", interactive=False)
            calls_box = gr.JSON(label="Tool calls")
            raw_box = gr.Textbox(label="Raw model output", interactive=False, lines=6)
    gr.Examples(
        examples=[[json.dumps(tools, indent=2), query] for _, tools, query in EXAMPLES],
        example_labels=[label for label, _, _ in EXAMPLES],
        inputs=[tools_box, query_box],
    )
    run.click(generate, [tools_box, query_box], [summary_box, calls_box, raw_box])
    query_box.submit(generate, [tools_box, query_box], [summary_box, calls_box, raw_box])

if __name__ == "__main__":
    page.queue(default_concurrency_limit=1).launch()
