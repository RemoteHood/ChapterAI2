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
            {"role": "user", "content": f"I have a text and I need your assistance in identifying all the characters from the text. The names should be presented in a clear and organized list. Here is the text: {processed_text}."}
        ]
    )
    return response.choices[0].message.content

def validate_names_llm(names):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in verifying and simplifying character names in literature."},
            {"role": "user", "content": f"""Here is the list of character names: {names}. Please provide me with a clear and simplified version of this list without writing anything before or after the names. Put this separation "/n" between full names."""}
        ]
    )
    return response.choices[0].message.content

def generate_summary(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing novels."},
            {"role": "user", "content": f"""Analyze the following processed information from the beginning of a novel and provide:
            1. An overall summary of the story so far (5-7 sentences)
            2. Main themes and motifs identified
            3. The author's writing style and narrative techniques

            Processed information:
            {text}"""}
        ]
    )
    return response.choices[0].message.content

def generate_chapter_title(chapter_content):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in creating engaging chapter titles for novels."},
            {"role": "user", "content": f"""Given the following chapter content, generate an appropriate and engaging title for this chapter. The title should be concise (no more than 10 words) and reflect the main theme or event of the chapter.

            Chapter content:
            {chapter_content}

            Please provide only the title, without any additional text or explanation."""}
        ]
    )
    return response.choices[0].message.content

def generate_chapter(selected_characters, selected_genres, process_text, overall_summary):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a skilled novelist, able to create engaging and well-structured chapters."},
            {"role": "user", "content": f"""Generate a new chapter of a story that focuses on the following characters: {', '.join(selected_characters)}.
            The genres of the story should be: {', '.join(selected_genres)}. If one of the genres is 'same', maintain the original style and genre of the author.

            Follow these guidelines:
            1. Write a cohesive narrative of about 1500-2000 words.
            2. Do not include a chapter title or number.
            3. Ensure the writing style and elements align with the chosen genres.

            Here is the processed text and overall summary for context:
            Processed Text: {process_text}
            Overall Summary: {overall_summary}

            Begin the chapter now:"""}
        ]
    )
    return response.choices[0].message.content

def generate_next_chapter(previous_chapter, selected_genres, process_text, overall_summary):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a skilled novelist, able to create engaging and well-structured chapters."},
            {"role": "user", "content": f"""Generate the next chapter of a story that continues from the following chapter.
            The genres of the story should be: {', '.join(selected_genres)}. If one of the genres is 'same', maintain the original style and genre of the author.

            Follow these guidelines:
            1. Write a cohesive narrative of about 1500-2000 words.
            2. Do not include a chapter title or number.
            3. Ensure the writing style and elements align with the chosen genres.

            Here is the previous chapter for context:
            Previous Chapter: {previous_chapter}
            Processed Text: {process_text}
            Overall Summary: {overall_summary}

            Begin the next chapter now:"""}
        ]
    )
    return response.choices[0].message.content

def generate_chapter_summary(chapter_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a skilled novelist, able to create engaging and well-structured chapters."},
            {"role": "user", "content": f"""Generate a summary of the following chapter: {chapter_text}.

            Follow these guidelines:
            1. Write a brief summary (2-3 sentences) of the chapter.
            2. Include key events and character development.
            3. Ensure the summary is cohesive and concise.

            Chapter Text: {chapter_text}

            Begin the summary now:"""}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Chapter Writer")

# Sidebar for character selection and genre selection
st.sidebar.title("Chapter Writer")

# Initialize session state
if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = {}
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""
if "validated_name_list" not in st.session_state:
    st.session_state.validated_name_list = []
if "overall_summary" not in st.session_state:
    st.session_state.overall_summary = ""
if "selected_characters" not in st.session_state:
    st.session_state.selected_characters = []
if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []

# File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Check if the file size is below 200 MB
if pdf_file:
    file_size = pdf_file.size
    if file_size < 200 * 1024 * 1024:  # 200 MB
        st.sidebar.write("Processing PDF...")
        st.session_state.summary, st.session_state.processed_text = process_pdf(pdf_file)
        st.session_state.potential_names = extract_names_llm(st.session_state.processed_text)
        st.session_state.validated_names = validate_names_llm(st.session_state.potential_names)
        st.session_state.validated_name_list = st.session_state.validated_names.split('/n')
        st.session_state.overall_summary = generate_summary(st.session_state.summary)

        st.sidebar.write("PDF processed successfully.")

# Display character list
st.sidebar.subheader("Characters")
for char in st.session_state.validated_name_list:
    if st.sidebar.checkbox(char, key=char):
        if char not in st.session_state.selected_characters:
            st.session_state.selected_characters.append(char)
    else:
        if char in st.session_state.selected_characters:
            st.session_state.selected_characters.remove(char)

# Genre selection
st.sidebar.subheader("Select Genres")
genres = ["Romance", "Mystery", "Thriller", "Crime", "Fantasy", "Science Fiction", "Historical Fiction", "Horror", "Paranormal", "Dystopian", "Adventure", "Humor", "same"]
st.session_state.selected_genres = st.sidebar.multiselect("Genres", genres, key="genres")

# Generate new chapter button
if st.sidebar.button("Generate new chapter"):
    if not st.session_state.selected_characters:
        st.sidebar.error("Please select at least one character.")
    elif not st.session_state.selected_genres:
        st.sidebar.error("Please select at least one genre.")
    else:
        st.sidebar.write("Generating new chapter...")
        new_chapter = generate_chapter(st.session_state.selected_characters, st.session_state.selected_genres, st.session_state.processed_text, st.session_state.overall_summary)
        chapter_title = generate_chapter_title(new_chapter)
        st.sidebar.write(f"Chapter Title: {chapter_title}")
        st.write(new_chapter)

# Generate next chapter button
if st.sidebar.button("Generate next chapter"):
    if not st.session_state.selected_characters:
        st.sidebar.error("Please select at least one character.")
    elif not st.session_state.selected_genres:
        st.sidebar.error("Please select at least one genre.")
    else:
        st.sidebar.write("Generating next chapter...")
        chapter_summary = generate_chapter_summary(new_chapter)
        new_chapter = generate_next_chapter(chapter_summary, st.session_state.selected_genres, st.session_state.processed_text, st.session_state.overall_summary)
        chapter_title = generate_chapter_title(new_chapter)
        st.sidebar.write(f"Chapter Title: {chapter_title}")
        st.write(new_chapter)








