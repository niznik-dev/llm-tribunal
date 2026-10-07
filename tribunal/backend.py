"""Pick a device, load a model, and generate one answer at a time."""

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
    return torch.float32 if device == "cpu" else torch.bfloat16


class Backend:
    def __init__(self, model_id, device, dtype, device_map="single"):
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto" if device_map == "auto" else device,
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - start
        # The first generate() pays one-time kernel setup; keep it out of question 1's tok/s.
        self.generate([{"role": "user", "content": "Hi"}], max_new_tokens=4)

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=128):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,  # templates without the flag ignore it
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        start = time.perf_counter()
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # Unset the model's sampling defaults so greedy decoding runs without warnings.
            temperature=None,
            top_p=None,
            top_k=None,
        )
        new_ids = output[0, inputs["input_ids"].shape[1]:].cpu()
        seconds = time.perf_counter() - start
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return text, len(new_ids), seconds
