import pytest
from src.generation.prompt_template import PromptTemplate

class TestPromptTemplate:
    def test_generate_prompt(self):
        question = "What is the capital of France?"
        relevant_docs = [
            {'content': 'France is a country in Europe.'},
            {'content': 'The capital of France is Paris.'}
        ]
        prompt = PromptTemplate.generate_prompt(question, relevant_docs)
        assert prompt is not None
        assert isinstance(prompt, str)
        assert prompt == (
            f"Use the following context to answer the question:\n\n"
            f"France is a country in Europe.\n\nThe capital of France is Paris.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
