import os
import json
import pandas as pd
import traceback
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2
import warnings
warnings.filterwarnings(action="ignore")
from dotenv import load_dotenv
from src.mcqgenerator.logger import logging

# Load environment variables from .env file
load_dotenv()

API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN_NEW")

# Defining the LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=API_TOKEN,
    model_kwargs={
        "max_length": 100,  # Limit response length
        "temperature": 0.7,  # Control creativity
        "top_p": 0.9        # Nucleus sampling
    }
)

# Define the prompt template for generating MCQs
TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} MCQs for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide.\
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
)
# Defining the chain
quiz_chain = LLMChain(
    llm=llm,
    prompt=quiz_generation_prompt,
    verbose=True,
    output_key="quiz"
)

# Template for 2nd chain
TEMPLATE2 = """
You are an expert Englist Grammarian and writer. Given the MCQ for {subject} students \ 
Evaluate the MCQ and check if the MCQ is correct or not. \
Make sure to check the MCQ and options are correct and not repeated. \
Update the quiz if necassary. \
{quiz}
"""
quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)

review_chain = LLMChain(
    llm=llm,
    prompt=quiz_evaluation_prompt,
    verbose=True,
    output_key="review"
)

# Combining the chains using SequentialChain
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True
)

