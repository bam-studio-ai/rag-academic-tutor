import warnings
warnings.filterwarnings("ignore", message="builtin type SwigPyPacked has no __module__ attribute")
warnings.filterwarnings("ignore", message="builtin type SwigPyObject has no __module__ attribute")

from transformers import pipeline
import torch

class LLMClient:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_name = model_name

        # For tesging purposes, we can use a mock model or a lightweight model
        if self.model_name == "MockModel":
            # Placeholder for a mock model for testing purposes
            self.pipe = lambda prompt, **kwargs: [{"generated_text": "This is a placeholder response from the LLM."}]
            # mock tokenizer = lambda x: x
            self.pipe.tokenizer = lambda x: x
            self.pipe.tokenizer.eos_token_id = 0

        else:
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    def get_response(self, prompt: str) -> str:
        response = self.pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )

        return response[0]['generated_text'].split('[/INST]')[-1].strip()
