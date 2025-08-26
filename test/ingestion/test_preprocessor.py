import pytest
import os
import tempfile
from src.ingestion.preprocessor import DocumentPreprocessor

class TestPreprocessor:
    def test_normalize_whitespace(self):
        text = "This  is   a \n test.\tWith  multiple spaces."
        expected = "This is a test. With multiple spaces."
        assert DocumentPreprocessor.normalize_whitespace(text) == expected

    def test_remove_page_numbers(self):
        text = "This is a test.\n\n1\n\nThis is another page.\n\n2"
        expected = "This is a test.\n\nThis is another page.\n"
        assert DocumentPreprocessor.remove_page_numbers(text) == expected

    def test_fix_encoding_issues(self):
        text = "This is an example with an en dash – and an em dash —."
        expected = "This is an example with an en dash - and an em dash -."
        assert DocumentPreprocessor.fix_encoding_issues(text) == expected

    def test_merge_hyphenated_words(self):
        text = "This is a hyphen-\nated word."
        expected = "This is a hyphenated word."
        assert DocumentPreprocessor.merge_hyphenated_words(text) == expected

    def test_remove_citations(self):
        text = "This is a test [1]. Another test (Smith, 2020)."
        expected = "This is a test . Another test ."
        assert DocumentPreprocessor.remove_citations(text) == expected

    def test_remove_headers_and_footers(self):
        text = "Confidential\nThis is a test.\nPage 1\nAnother line."
        expected = "This is a test.\nAnother line."
        assert DocumentPreprocessor.remove_headers_and_footers(text) == expected

    def test_normalize_quotes(self):
        text = '“This is a test.” ‘Single quotes’'
        expected = '"This is a test." \'Single quotes\''
        assert DocumentPreprocessor.normalize_quotes(text) == expected
