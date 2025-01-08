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
import streamlit.components.v1 as components

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

def handle_chunked_upload(chunk, chunk_number, total_chunks, upload_dir):
    chunk_path = os.path.join(upload_dir, f"chunk_{chunk_number}.part")
    with open(chunk_path, "wb") as chunk_file:
        chunk_file.write(chunk)
    st.session_state.uploaded_chunks[chunk_number] = chunk_path
    if len(st.session_state.uploaded_chunks) == total_chunks:
        combine_chunks(upload_dir, total_chunks)

def combine_chunks(upload_dir, total_chunks):
    final_file_path = os.path.join(upload_dir, "final_file.pdf")
    with open(final_file_path, "wb") as final_file:
        for i in range(total_chunks):
            chunk_path = st.session_state.uploaded_chunks[i]
            with open(chunk_path, "rb") as chunk_file:
                final_file.write(chunk_file.read())
            os.remove(chunk_path)
    st.success("File uploaded and combined successfully!")
    st.session_state.uploaded_chunks = {}
    return final_file_path

def process_pdf(pdf_file_path):
    try:
        loader = PyPDFLoader(pdf_file_path)
        pages = loader.load_and_split()
        docs = text_splitter.split_documents(pages)
        full_text = " ".join([doc.page_content for doc in docs])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                {"role": "user", "content": f"Please summarize the following document:\n\n{full_text}"}
            ],
            max_tokens=1000
        )
        summary = response.choices[0].message.content
        logging.info("Summary generated successfully")
        processed_text = process_text(summary)
        logging.info("Text processed successfully")
        logger.debug("OpenAI API Response: %s", summary)
        os.unlink(pdf_file_path)
        return summary, processed_text
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        raise

def process_text(text):
    try:
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
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        st.error(f"Error processing text: {str(e)}")
        raise

def extract_names_llm(processed_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in literary analysis and character identification."},
                {"role": "user", "content": f"I have a text and I need your assistance in identifying all the characters from the text. The names should be presented in a clear and organized list. Here is the text: {processed_text}."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error extracting names: {str(e)}")
        st.error(f"Error extracting names: {str(e)}")
        raise

def validate_names_llm(names):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in verifying and simplifying character names in literature."},
                {"role": "user", "content": f"""Here is the list of character names: {names}. Please provide me with a clear and simplified version of this list without writing anything before or after the names. Put this separation "/n" between full names."""}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error validating names: {str(e)}")
        st.error(f"Error validating names: {str(e)}")
        raise

def generate_summary(text):
    try:
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
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        st.error(f"Error generating summary: {str(e)}")
        raise

def generate_chapter_title(chapter_content):
    try:
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
    except Exception as e:
        logging.error(f"Error generating chapter title: {str(e)}")
        st.error(f"Error generating chapter title: {str(e)}")
        raise

def generate_chapter(selected_characters, selected_genres, process_text, overall_summary):
    try:
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
    except Exception as e:
        logging.error(f"Error generating chapter: {str(e)}")
        st.error(f"Error generating chapter: {str(e)}")
        raise

def generate_next_chapter(previous_chapter, selected_genres, process_text, overall_summary):
    try:
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
    except Exception as e:
        logging.error(f"Error generating next chapter: {str(e)}")
        st.error(f"Error generating next chapter: {str(e)}")
        raise

def generate_chapter_summary(chapter_text):
    try:
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
    except Exception as e:
        logging.error(f"Error generating chapter summary: {str(e)}")
        st.error(f"Error generating chapter summary: {str(e)}")
        raise

# Streamlit UI
st.title("Chapter Writer")

# Sidebar for character selection and genre selection
st.sidebar.title("Chapter Writer")

# Initialize session state
if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = {}

# File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Check if the file size is below 200 MB
if pdf_file:
    file_size = pdf_file.size
    if file_size < 200 * 1024 * 1024:  # 200 MB
        st.sidebar.write("Processing PDF...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name
        summary, processed_text = process_pdf(temp_file_path)
        potential_names = extract_names_llm(processed_text)
        validated_names = validate_names_llm(potential_names)
        validated_name_list = validated_names.split('/n')
        overall_summary = generate_summary(summary)

        st.sidebar.write("PDF processed successfully.")

        # Display overall summary
        st.sidebar.subheader("Overall Summary")
        st.sidebar.write(overall_summary)

        # Display character list
        st.sidebar.subheader("Characters")
        selected_characters = []
        for char in validated_name_list:
            if st.sidebar.checkbox(char):
                selected_characters.append(char)

        # Display processed text
        st.sidebar.subheader("Processed Text")
        st.sidebar.write(processed_text)

        # Genre selection
        st.sidebar.subheader("Select Genres")
        genres = ["Romance", "Mystery", "Thriller", "Crime", "Fantasy", "Science Fiction", "Historical Fiction", "Horror", "Paranormal", "Dystopian", "Adventure", "Humor", "same"]
        selected_genres = st.sidebar.multiselect("Genres", genres)

        # Generate new chapter button
        if st.sidebar.button("Generate new chapter"):
            if not selected_characters:
                st.sidebar.error("Please select at least one character.")
            elif not selected_genres:
                st.sidebar.error("Please select at least one genre.")
            else:
                st.sidebar.write("Generating new chapter...")
                new_chapter = generate_chapter(selected_characters, selected_genres, processed_text, overall_summary)
                chapter_title = generate_chapter_title(new_chapter)
                st.sidebar.write(f"Chapter Title: {chapter_title}")
                st.write(new_chapter)

        # Generate next chapter button
        if st.sidebar.button("Generate next chapter"):
            if not selected_characters:
                st.sidebar.error("Please select at least one character.")
            elif not selected_genres:
                st.sidebar.error("Please select at least one genre.")
            else:
                st.sidebar.write("Generating next chapter...")
                chapter_summary = generate_chapter_summary(new_chapter)
                new_chapter = generate_next_chapter(chapter_summary, selected_genres, processed_text, overall_summary)
                chapter_title = generate_chapter_title(new_chapter)
                st.sidebar.write(f"Chapter Title: {chapter_title}")
                st.write(new_chapter)
    else:
        st.sidebar.write("File size exceeds 200 MB. Please use the chunked upload method.")

# HTML and JavaScript for chunked uploads
js_file_path = os.path.join(os.path.dirname(__file__), 'chunk_upload.js')
with open(js_file_path, 'r') as file:
    js_content = file.read()

chunk_upload_html = f"""
<input type="file" id="fileInput">
<button id="uploadButton">Upload</button>
<script>
{js_content}
</script>
"""

components.html(chunk_upload_html, height=200)

# Check if the final file is combined
upload_dir = "uploads"
if os.path.exists(os.path.join(upload_dir, "final_file.pdf")):
    pdf_file_path = os.path.join(upload_dir, "final_file.pdf")
    st.sidebar.write("Processing PDF...")
    summary, processed_text = process_pdf(pdf_file_path)
    potential_names = extract_names_llm(processed_text)
    validated_names = validate_names_llm(potential_names)
    validated_name_list = validated_names.split('/n')
    overall_summary = generate_summary(summary)

    st.sidebar.write("PDF processed successfully.")

    # Display overall summary
    st.sidebar.subheader("Overall Summary")
    st.sidebar.write(overall_summary)

    # Display character list
    st.sidebar.subheader("Characters")
    selected_characters = []
    for char in validated_name_list:
        if st.sidebar.checkbox(char):
            selected_characters.append(char)

    # Display processed text
    st.sidebar.subheader("Processed Text")
    st.sidebar.write(processed_text)

    # Genre selection
    st.sidebar.subheader("Select Genres")
    genres = ["Romance", "Mystery", "Thriller", "Crime", "Fantasy", "Science Fiction", "Historical Fiction", "Horror", "Paranormal", "Dystopian", "Adventure", "Humor", "same"]
    selected_genres = st.sidebar.multiselect("Genres", genres)

    # Generate new chapter button
    if st.sidebar.button("Generate new chapter"):
        if not selected_characters:
            st.sidebar.error("Please select at least one character.")
        elif not selected_genres:
            st.sidebar.error("Please select at least one genre.")
        else:
            st.sidebar.write("Generating new chapter...")
            new_chapter = generate_chapter(selected_characters, selected_genres, processed_text, overall_summary)
            chapter_title = generate_chapter_title(new_chapter)
            st.sidebar.write(f"Chapter Title: {chapter_title}")
            st.write(new_chapter)

    # Generate next chapter button
    if st.sidebar.button("Generate next chapter"):
        if not selected_characters:
            st.sidebar.error("Please select at least one character.")
        elif not selected_genres:
            st.sidebar.error("Please select at least one genre.")
        else:
            st.sidebar.write("Generating next chapter...")
            chapter_summary = generate_chapter_summary(new_chapter)
            new_chapter = generate_next_chapter(chapter_summary, selected_genres, processed_text, overall_summary)
            chapter_title = generate_chapter_title(new_chapter)
            st.sidebar.write(f"Chapter Title: {chapter_title}")
            st.write(new_chapter)





