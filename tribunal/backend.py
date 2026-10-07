"""Pick a device, load a model, and generate one answer at a time.

The only file that touches the model. run.py calls Backend(...) once, then
generate(...) once per question (or per sample).
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    # bf16 halves memory vs fp32 with little quality loss, but CPUs run it slowly or not at all.
    return torch.float32 if device == "cpu" else torch.bfloat16


class Backend:
    def __init__(self, model_id, device, dtype, device_map="single"):
        start = time.perf_counter()
        # The tokenizer turns text into token ids and back; the model only ever sees ids.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # "CausalLM" = predicts the next token. One class loads Qwen, Llama, Gemma, ... alike.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            # "auto" spreads layers across every visible GPU, for models too big for one.
            device_map="auto" if device_map == "auto" else device,
        )
        self.model.eval()  # turn off training-only behavior such as dropout
        self.load_seconds = time.perf_counter() - start

    @torch.inference_mode()  # no gradient bookkeeping: we never train, so skip it
    def generate(self, messages, max_new_tokens=128, temperature=0.0):
        # Chat models were trained on conversations wrapped in special tokens. The chat template
        # wraps our messages the same way and ends with the cue for the assistant to reply.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,  # no hidden reasoning before the answer; templates without the flag ignore it
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        start = time.perf_counter()
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # temperature 0 means greedy: always take the single most likely next token.
            do_sample=temperature > 0,
            temperature=temperature or None,
            # Some families (Llama, Gemma) ship sampling defaults in generation_config.json; clear them
            # so sampling is plain temperature sampling. Qwen3.5 ships none, so this is a no-op there.
            top_p=None,
            top_k=None,
        )
        # generate() returns the prompt followed by the answer; keep only the answer's tokens.
        new_ids = output[0, inputs["input_ids"].shape[1]:].cpu()
        seconds = time.perf_counter() - start
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return text, len(new_ids), seconds
