import os
import PyPDF2
import json
import traceback
import pandas as pd

# Read from a PDF file and return the text content
def read_file(file):
    if file.name.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF file: {e}")
    elif file.name.endswith('.txt'):
        try:
            text = file.read().decode('utf-8')
            return text
        except Exception as e:
            raise Exception(f"Error reading TXT file: {e}")
    else:
        raise Exception("Unsupported file format. Please upload a PDF or TXT file.")
    

# Clean the response to make suitable json
def clean_json(quiz):
    ind = quiz.index("### RESPONSE_JSON")
    quiz_dict = quiz[ind+13:].split("RESPONSE_JSON")[1]
    quiz_dict_new = quiz_dict.replace("### RESPONSE_JSON_FOR_MCQ_ON_PROFESSIONAL_QUOTATION_FOR_EDUCATIONAL_CONSULTANCY_WEBSITE", "")
    quiz_dict_proper = json.loads(quiz_dict_new)
    return quiz_dict_proper


# Convert the quiz response to a tabular format
def create_table(quiz_dict):
    try:
        quiz_table_data = []
        for key, value in quiz_dict.items():
            mcq = value['MCQ']
            options = " | ".join(
                [
                    f"{option}: {value['Options'][option]}" for option in value['Options']
                ]
                )
            correct = value['Correct']
            quiz_table_data.append({
                'MCQ': mcq,
                'Options': options,
                'Correct': correct
            })
        # Convert to DataFrame or any other format as needed
        quiz_df = pd.DataFrame(data=quiz_table_data)
        return quiz_df
    except json.JSONDecodeError as e:
        raise Exception(f"Error decoding JSON: {e}")