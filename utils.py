import re
import hashlib
import tiktoken
from langchain.document_loaders import PyPDFLoader

def hash_text(text):
    """Hashes a text using SHA256 algorithm.

    Args:
        text_to_hash (str): The text to hash.

    Returns:
        str: The hexadecimal representation of the hash.
    """
    text_bytes = text.encode('utf-8')

    # Step 3: Create a new SHA256 hash object
    hash_object = hashlib.sha256()

    # Step 4: Update the hash object with the encoded text
    hash_object.update(text_bytes)

    # Step 5: Generate the hexadecimal representation of the hash
    hex_digest = hash_object.hexdigest()

    return hex_digest


def extract_text_from_pdf(file_name: str) -> str:
    """Extracts the text from the given PDF file.

    Args:
        file_name (str): The name of the PDF file to extract the text from.

    Returns:
        str: The extracted text.
    """
    loader = PyPDFLoader(file_name)
    pages = loader.load_and_split()
    pages = [page.page_content for page in pages]
    content = " ".join(pages)
    return content

def count_gpt_tokens(text: str) -> int:
    """Counts the number of tokens in the given text.

    Args:
        text (str): The text to count the tokens of.

    Returns:
        int: The number of tokens in the given text.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def clean_text(text: str):
    # Remove Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Replace multiple spaces and combinations of new lines and spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text