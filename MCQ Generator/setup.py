from setuptools import setup, find_packages

setup(
    name="mcq-generator",
    version="0.1",
    author="Ujjwal Kumar",
    author_email="batman.c137@gmail.com",
    description="A tool to generate MCQs from text",
    install_requires=["openai", "langchain", "streamlit", "python-dotenv", "PyPDF2"],
    # It automatically finds all Python packages (i.e., folders containing an __init__.py file)
    
    packages=find_packages()
    )