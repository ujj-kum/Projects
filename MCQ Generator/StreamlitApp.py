import warnings
warnings.filterwarnings(action="ignore")
import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file, create_table, clean_json
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

# Loading JSON File
with open('D:\WORK STUDY\PROJECTS\MCQ Generator\Response.json', 'r') as f:
    RESPONSE_JSON = json.load(f)

# Create a title for the app
st.title("MCQ Generator")

# Create a form
with st.form(key='user_inputs'):
    # File uploader for PDF or TXT files
    file = st.file_uploader("Upload a PDF or TXT file", type=['pdf', 'txt'])
    
    # Text input for number of MCQs
    number = st.number_input(label="Number of MCQs", min_value=1, max_value=100, value=5)
    
    # Text input for subject
    subject = st.text_input(label="Subject")
    
    # Text input for tone
    tone = st.text_input(label="Tone")
    
    # Submit button
    submit_button = st.form_submit_button(label='Generate MCQs')

    # Check if the submit button is pressed
    if submit_button and file and number and subject and tone:
        with st.spinner("Generating MCQs..."):
            try:
                # Read the file and extract text
                # from src.mcqgenerator.utils import read_file
                text = read_file(file)
                
                # Generate and evaluate MCQs
                response = generate_evaluate_chain(
                            {'text':text,
                            'number':number,
                            'subject':subject,
                            'tone':tone,
                            'response_json':json.dumps(RESPONSE_JSON)
                            }
                        )
                
                # Convert the quiz response to a tabular format
                # from src.mcqgenerator.utils import clean_json, create_table
                quiz = response['quiz']
                if isinstance(quiz, str):
                    quiz_dict_proper = clean_json(quiz)
                    quiz_table_df = create_table(quiz_dict_proper)
                
                    # Display the quiz table
                    st.write("Generated MCQs:")
                    st.table(quiz_table_df)
            except Exception as e:
                logging.error(f"Error: {e}")
                st.error(f"An error occurred: {e}")
            else:
                st.success("MCQs generated successfully!")
                st.balloons()
                # Display the quiz table