from langchain.prompts import ChatPromptTemplate
from .base_model import Model
from .errors import InferenceError

class HistorySummarizer:
    """Summarizes conversation history to optimize token usage."""
    def __init__(self, model: Model, max_summary_length: int = 100):
        self.model = model
        self.max_summary_length = max_summary_length
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following conversation history in {max_length} words or less:\n\n{history}"),
            ("human", "Provide a concise summary.")
        ])

    def summarize(self, history: list[dict]) -> str:
        """Summarize a conversation history (list of message dicts with 'role' and 'content')."""
        try:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            chain = self.prompt_template | self.model.model
            summary = chain.invoke({"history": history_text, "max_length": self.max_summary_length}).content
            return summary
        except Exception as e:
            raise InferenceError(f"History summarization failed: {str(e)}")