import re
import logging

class DocumentPreprocessor:
    """A class to preprocess and clean text documents."""

    def clean_document(text: str) -> str:
        """Clean the content of a Document by removing unwanted characters and extra spaces."""
        cleaned_text = __class__.normalize_whitespace(text)
        cleaned_text = __class__.remove_page_numbers(cleaned_text)
        cleaned_text = __class__.fix_encoding_issues(cleaned_text)
        cleaned_text = __class__.merge_hyphenated_words(cleaned_text)
        cleaned_text = __class__.remove_citations(cleaned_text)
        cleaned_text = __class__.remove_headers_and_footers(cleaned_text)
        cleaned_text = __class__.normalize_quotes(cleaned_text)

        return cleaned_text.strip()

    def normalize_whitespace(text: str) -> str:
        """Replace multiple spaces, tabs, and newlines with a single space."""
        return re.sub(r'\s+', ' ', text)

    def remove_page_numbers(text: str) -> str:
        """Remove page numbers that are on their own line."""
        return re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)


    def fix_encoding_issues(text: str) -> str:
        """Fix common encoding issues."""
        text = text.replace('\u2013', '-')  # En dash to hyphen
        text = text.replace('\u2014', '-')  # Em dash to hyphen
        text = text.replace('\u2018', "'")  # Left single quote to apostrophe
        text = text.replace('\u2019', "'")  # Right single quote to apostrophe
        text = text.replace('\u201c', '"')  # Left double quote to standard double quote
        text = text.replace('\u201d', '"')  # Right double quote to standard double quote
        return text

    def merge_hyphenated_words(text: str) -> str:
        """Merge words that are split by hyphens at line breaks."""
        return re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    def remove_citations(text: str) -> str:
        """Remove in-text citations like [1], (Smith, 2020), etc."""
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\w+, \d{4}\)', '', text)
        return text

    def remove_headers_and_footers(text: str) -> str:
        """Remove common headers and footers."""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if re.match(r'^\s*Page \d+\s*$', line):
                continue
            if re.match(r'^\s*Confidential\s*$', line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def normalize_quotes(text: str) -> str:
        """Normalize different types of quotes to standard quotes."""
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace("‘", "'").replace("’", "'")
        return text

