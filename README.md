# RAG-Powered Chatbot with Evaluation

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) powered chatbot that can answer questions based on a provided PDF document. The chatbot is built using LangChain and OpenAI's GPT-3.5-turbo, with a Streamlit frontend for user interaction and evaluation.

## Features

- **PDF Document Processing**: Loads and processes a PDF document to create a searchable index.
- **RAG Model**: Combines retrieval and generation to provide accurate answers based on the document content.
- **Streamlit Frontend**: A web-based interface for querying the chatbot and uploading evaluation datasets.
- **Evaluation Metrics**: Grades responses based on factual accuracy and provides detailed feedback.


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Akash265/RAG-Evaluation.git
    cd rag-chatbot-evaluation
    ```

2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

3. Set your OpenAI API key:
    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"
    ```

## Usage

1. **Start the Streamlit Application**:
    ```sh
    streamlit run app.py
    ```

2. **Interacting with the Chatbot**:
    - Enter a question in the input box and get a response based on the PDF document.
    - Upload an evaluation dataset (Excel format) to evaluate the chatbot's accuracy.

3. **Evaluation Dataset Format**:
    The evaluation dataset should be an Excel file with two columns:
    - `Query`: The question to be asked.
    - `Response`: The correct answer based on the PDF document.

## Code Overview

- **app.py**: The main Streamlit application file that handles user input, document processing, and evaluation.
- **requirements.txt**: Lists all the required Python libraries for the project.
