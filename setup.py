from setuptools import setup, find_packages

setup(
    name="genai_toolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.0",
        "langchain-openai>=0.2.0",
        "langchain-google-genai>=1.0.10",
        "langchain-huggingface>=0.3.0",
        "redis>=5.0.8",
        "nltk>=3.8.1",
        "rouge-score>=0.1.2",
        "numpy>=1.26.4",
        "cryptography>=43.0.1",
    ],
    author="Sunil K Anikepati",
    author_email="sunil.anikepati@gmail.com",
    description="A secure LangChain-based toolkit for prompt engineering, model inference, and evaluation in GenAI with Redis caching",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/anikepati/genai_toolkit",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)