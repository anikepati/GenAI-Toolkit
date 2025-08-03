import unittest
from genai_toolkit import HuggingFaceAdapter, PromptManager, Evaluator, HistorySummarizer, SecurityManager
from genai_toolkit.errors import InferenceError, SecurityError

class TestGenAIToolkit(unittest.TestCase):
    def setUp(self):
        self.enterprise_app_key = "a" * 32  # Dummy 32-byte key for testing
        self.model = HuggingFaceAdapter(enterprise_app_key=self.enterprise_app_key, model_name="gpt2")
        self.prompt_manager = PromptManager()
        self.evaluator = Evaluator(self.model)
        self.summarizer = HistorySummarizer(self.model, max_summary_length=50)

    def test_security_manager(self):
        with self.assertRaises(SecurityError):
            SecurityManager("invalid_key")  # Too short

    def test_prompt_manager(self):
        self.prompt_manager.create_template("test", "Hello {name}", ["name"])
        prompt = self.prompt_manager.get_template("test")
        self.assertIsNotNone(prompt)

    def test_generate(self):
        try:
            result = self.model.generate("Test prompt")
            self.assertIsInstance(result, str)
        except InferenceError:
            pass

    def test_evaluate(self):
        metrics = self.evaluator.evaluate_rag(
            query="What is AI?",
            context="AI is a field of computer science.",
            generated_answer="AI enables machines to think.",
            reference_answer="AI enables machines to mimic intelligence."
        )
        self.assertIn("bleu", metrics)

    def test_summarize_history(self):
        history = [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]
        try:
            summary = self.summarizer.summarize(history)
            self.assertIsInstance(summary, str)
        except InferenceError:
            pass

if __name__ == "__main__":
    unittest.main()