import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAEvalChain
from typing import List, Dict
import pandas as pd
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-lwhyaPhdhtnQt3xsi2kMT3BlbkFJLz9ibUxbY1FeMmcx7RqF"

# Load the PDF document and create the vector store index
loader = PyPDFLoader("policy-booklet-0923.pdf")
document = loader.load_and_split()

# Create the vector index
embeddings = OpenAIEmbeddings()
index = VectorstoreIndexCreator(embedding=embeddings).from_documents(document)

# Initialize the RAG model and language model
llm  = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=index.vectorstore.as_retriever())

# Define the grading prompt template
template = """You are a teacher evaluating answers. 
You are given a question, my answer, and the true answer, and are asked to score  my answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
MY ANSWER: my answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade my answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between my answer and true answer. It is OK if my answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
MY ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:
"""

GRADE_ANSWER_PROMPT = PromptTemplate(input_variables=["query", "result", "answer"], template=template)

def grade_model_answer(predicted_dataset: List[Dict], predictions: List[Dict]) -> List[Dict]:
    """
    Grades the distilled answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @return: A list of scores for the distilled answers.
    """
    # Create an evaluation chain
    eval_chain = QAEvalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        prompt=GRADE_ANSWER_PROMPT
    )

    for pred in predictions:
        if not pred['result']:
            pred['result'] = 'No answer provided'

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        predicted_dataset,
        predictions,
        question_key="query",
        prediction_key="result"
    )

    return graded_outputs

# Streamlit UI
st.title("RAG-Powered Chatbot with Evaluation")
st.write("Ask a question related to the policy document:")

# Input question
user_query = st.text_input("Enter your question:")

if user_query:
    # Get the chatbot response
    response = qa.invoke({"query": user_query})
    st.write("Chatbot Response:", response['result'])

# Upload evaluation dataset
uploaded_file = st.file_uploader("Upload Evaluation Dataset (Excel)")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    dataset = []
    for i in range(len(df)):   
        question = df.loc[i, "Query"]
        answer = df.loc[i, "Response"]
        dataset.append({"query": question, "answer": answer})
    
    # Make predictions on the dataset
    prediction_dataset = []
    for i in range(len(dataset)):
        query = dataset[i]['query']
        result = qa.invoke({"query": query})
        prediction_dataset.append({"query": query, "result": result['result']})

    # Grade the model answers
    graded_outputs = grade_model_answer(dataset, prediction_dataset)

    # Calculate accuracy
    correct_count = sum(1 for output in graded_outputs if output['results'].split(':')[1].strip() == 'CORRECT')
    accuracy = round(correct_count / len(graded_outputs) * 100, 2)
    st.write(f"Accuracy: {accuracy}%")

    # Display detailed grading
    st.write("Detailed Grading Results:")
    for output in graded_outputs:
        st.write(output)
