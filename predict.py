import os
import pathlib
import subprocess
import time
import socket

# model
model = f"Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
model_path = f"./models/{model}"
model_url = f"https://hf-mirror.com/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/{model}?download=true"

import inspect
from cog import BasePredictor, ConcatenateIterator, Input
from llama_cpp import Llama

# This prompt formatting was copied from the original CodeLlama repo:
# https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L44

# These are components of the prompt that should not be changed by the users
PROMPT_TEMPLATE_WITH_SYSTEM_PROMPT = (
    f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {{system_prompt}}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {{prompt}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)


class Predictor(BasePredictor):
    is_instruct = True

    def setup(self) -> None:
        if not os.path.exists(model_path):
            print(f"Downloading model weights from {model_url}....")
            start = time.time()
            subprocess.check_call(["pget", "-f", model_url, model_path], close_fds=True)
            print("Downloading weights took: ", time.time() - start)
        self.llm = Llama(
            model_path, n_ctx=4096, n_gpu_layers=-1, main_gpu=0, n_threads=1
        )

    def predict(
        self,
        prompt: str = Input(description="Prompt", default="Tell me a story about the Cheesecake Kingdom."),
        system_prompt: str = Input(
            description="System prompt to send to CodeLlama. This is prepended to the prompt and helps guide system behavior.",
            default="You are a helpful assistant"
        ),
        max_tokens: int = Input(
            description="Max number of tokens to return", default=512
        ),
        temperature: float = Input(description="Temperature", default=0.8),
        top_p: float = Input(description="Top P", default=0.95),
        top_k: int = Input(description="Top K", default=10),
        frequency_penalty: float = Input(
            description="Frequency penalty", ge=0.0, le=2.0, default=0.0
        ),
        presence_penalty: float = Input(
            description="Presence penalty", ge=0.0, le=2.0, default=0.0
        ),
        repeat_penalty: float = Input(
            description="Repetition penalty", ge=0.0, le=2.0, default=1.1
        )
    ) -> ConcatenateIterator[str]:
        print("Pod hostname:", socket.gethostname())
        # If USE_SYSTEM_PROMPT is True, and the user has supplied some sort of system prompt, we add it to the prompt.
        if self.is_instruct:
            prompt_text = PROMPT_TEMPLATE_WITH_SYSTEM_PROMPT.format(system_prompt=system_prompt, prompt=prompt)

        elif not self.is_instruct:
            prompt_text = prompt

        print("Prompt:\n" + prompt_text)
        output = ""

        for tok in self.llm(
            prompt=prompt_text,
            grammar=None,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            mirostat_mode=0,
        ):
            output += tok["choices"][0]["text"]
            yield tok["choices"][0]["text"]
        print("Output:", output)

    _predict = predict

    def base_predict(self, *args, **kwargs) -> ConcatenateIterator:
        kwargs["system_prompt"] = None
        return self._predict(*args, **kwargs)

    # for the purposes of inspect.signature as used by predictor.get_input_type,
    # remove the argument (system_prompt)
    # this removes system_prompt from the Replicate API for non-chat models.

    if not is_instruct:
        wrapper = base_predict
        sig = inspect.signature(_predict)
        params = [p for name, p in sig.parameters.items() if name != "system_prompt"]
        wrapper.__signature__ = sig.replace(parameters=params)
        predict = wrapper

