import streamlit as st
import os
import tempfile
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with secrets
openai_api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=openai_api_key)

# Set up text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def process_pdf(pdf_file):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            # Write the uploaded file content to the temporary file
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name

        # Load PDF directly from file object
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()

        # Split the document into chunks
        docs = text_splitter.split_documents(pages)

        # Combine all text from docs
        full_text = " ".join([doc.page_content for doc in docs])

        # Summarize the document using OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the specified model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                {"role": "user", "content": f"Please summarize the following document:\n\n{full_text}"}
            ],
            max_tokens=1000  # Adjust as needed
        )

        summary = response.choices[0].message.content

        logging.info("Summary generated successfully")

        # Process for character names and other details
        try:
            processed_text = process_text(summary)
            logging.info("Text processed successfully")
        except Exception as e:
            logging.error(f"Error in process_text: {str(e)}")
            raise

        # Log the response from the OpenAI API
        logger.debug("OpenAI API Response: %s", summary)

        # Don't forget to remove the temporary file
        os.unlink(temp_file_path)

        return summary, processed_text
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        raise

def process_text(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in literary analysis."},
            {"role": "user", "content": f"""Analyze the following text from a novel and provide:
            1. A brief summary (2-3 sentences)
            2. Key events (bullet points)
            3. Character mentions
            4. Any notable time references

            Text: {text}"""}
        ]
    )
    return response.choices[0].message.content

def extract_names_llm(processed_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in literary analysis and character identification."},
            {"role": "user", "content": f"I have a text and I need your assistance in identifying all the characters from the text. The names should be presented in a clear and organized list. Here is the text: {processed_t










