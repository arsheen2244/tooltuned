"""The demo's logic, kept free of Gradio so the tests can import it.

The model answers either with <tool_call> blocks holding JSON, or with plain text when
none of the tools fit. These helpers check the user's tools and read the model's answer.
"""

import json
from typing import Any

MODEL_ID = "asheeee24/tooltuned-qwen2.5-1.5b"
MAX_NEW_TOKENS = 256

WEATHER = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"city": {"description": "City name.", "type": "str"}},
}
CONVERT = {
    "name": "convert_currency",
    "description": "Convert an amount of money from one currency to another.",
    "parameters": {
        "amount": {"description": "Amount to convert.", "type": "float"},
        "from_currency": {"description": "Currency code to convert from, e.g. USD.", "type": "str"},
        "to_currency": {"description": "Currency code to convert to, e.g. EUR.", "type": "str"},
    },
}
ROUTE = {
    "name": "plan_route",
    "description": "Plan a driving route between two addresses.",
    "parameters": {
        "start": {"description": "Starting address.", "type": "str"},
        "end": {"description": "Destination address.", "type": "str"},
    },
}

# (description, tools, query). The last two show both sides of the model's known weakness.
EXAMPLES: list[tuple[str, list[dict[str, Any]], str]] = [
    ("One call", [WEATHER], "What's the weather like in Paris right now?"),
    ("Parallel calls", [WEATHER], "Is it raining in Tokyo or in Oslo?"),
    (
        "Picking the right tool",
        [WEATHER, CONVERT],
        "How much is 250 US dollars in euros?",
    ),
    ("No tool fits", [CONVERT], "Who wrote the novel Moby Dick?"),
    (
        "Known weakness: a near-miss tool",
        [ROUTE],
        "Where is the nearest pharmacy?",
    ),
]


def load_model(dtype_name: str = "float32") -> tuple[Any, Any]:
    """Load the tokenizer and model on the CPU. Imported here so the tests needn't load torch."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=getattr(torch, dtype_name))
    return tokenizer, model.eval()


def answer(tokenizer: Any, model: Any, tools: list[dict[str, Any]], query: str) -> str:
    """Return the model's greedy answer to the query, given the tools."""
    import torch

    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_tools(text: str) -> list[dict[str, Any]]:
    """Read the tools box: a JSON list of tool objects, or a single tool object."""
    try:
        tools = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"The tools aren't valid JSON: {error.msg} (line {error.lineno}).") from error
    if isinstance(tools, dict):
        tools = [tools]
    if not isinstance(tools, list) or not tools:
        raise ValueError("Give at least one tool, as a JSON list of objects.")
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise ValueError('Every tool needs to be a JSON object with a "name".')
    return tools


def parse_tool_calls(answer: str) -> list[dict[str, Any] | str]:
    """Return each <tool_call> block in the answer: its JSON, or the raw text if it isn't valid."""
    calls: list[dict[str, Any] | str] = []
    for block in answer.split("<tool_call>")[1:]:
        body = block.split("</tool_call>")[0].strip()
        try:
            calls.append(json.loads(body))
        except json.JSONDecodeError:
            calls.append(body)
    return calls


def summarize(answer: str) -> str:
    """Describe the answer in one line for the page."""
    calls = parse_tool_calls(answer)
    if not calls:
        return "No tool call: the model answered in plain text."
    broken = sum(isinstance(call, str) for call in calls)
    noun = "call" if len(calls) == 1 else "calls"
    if broken:
        return f"{len(calls)} tool {noun}, {broken} of them not valid JSON."
    return f"{len(calls)} tool {noun}."
