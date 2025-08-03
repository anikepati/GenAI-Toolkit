from langchain.prompts import ChatPromptTemplate
from .errors import InferenceError

class PromptManager:
    """Manages prompt templates for GenAI models."""
    def __init__(self, system_prompt: str = "You are a helpful AI assistant."):
        self.system_prompt = system_prompt
        self.templates = {}

    def create_template(self, name: str, template: str, input_variables: list):
        """Create and store a prompt template."""
        try:
            self.templates[name] = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", template)
            ])
        except Exception as e:
            raise InferenceError(f"Failed to create prompt template: {str(e)}")

    def get_template(self, name: str) -> ChatPromptTemplate:
        """Retrieve a prompt template by name."""
        if name not in self.templates:
            raise InferenceError(f"Template {name} not found")
        return self.templates[name]

    def format_prompt(self, name: str, **kwargs) -> ChatPromptTemplate:
        """Format a prompt template with input variables."""
        template = self.get_template(name)
        return template.format(**kwargs)