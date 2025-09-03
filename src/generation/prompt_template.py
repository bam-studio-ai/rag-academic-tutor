from typing import List

class PromptTemplate:
    @staticmethod
    def generate_prompt(question: str, relevant_docs: List) -> str:
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        prompt = (
            f"Use the following context to answer the question:\n\n"
            f"{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return prompt

