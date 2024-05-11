import databutton as db
import streamlit as st
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
import os
import random
import textwrap as tr
from text_load_utils import parse_txt, text_to_docs, parse_pdf, load_default_pdf
from df_chat import user_message, bot_message


st.set_page_config("Multilingual Chat Bot 🤖", layout="centered")

cohere_api_key = db.secrets.get(name="secret-key")

st.title("Multilingual Chat Bot 🤖")
st.info(
    "For your personal data!"
)

opt = st.radio("--", options=["Try the demo!", "Upload-own-file"])

pages = None
if opt == "Upload-own-file":

    uploaded_file = st.file_uploader(
        "**Upload a pdf or txt file :**",
        type=["pdf", "txt"],
    )
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            doc = parse_txt(uploaded_file)
        else:
            doc = parse_pdf(uploaded_file)
        pages = text_to_docs(doc)
else:
    st.text("Quick Prompts to try (English | Korean):")
    
    st.code("What is the key lesson from this article?")
    st.code("이 기사의 핵심은 무엇입니까?")
    pages = load_default_pdf()


page_holder = st.empty()
prompt_template = """Text: {context}

Question: {question}

Answer the question based on the pdf."""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

prompt = st.session_state.get("prompt", None)

if prompt is None:
    prompt = [{"role": "system", "content": prompt_template}]

for message in prompt:
    if message["role"] == "user":
        user_message(message["content"])
    elif message["role"] == "assistant":
        bot_message(message["content"], bot_name="Multilingual Personal Chat Bot")

if pages:
    with page_holder.expander("File Content", expanded=False):
        pages
    embeddings = CohereEmbeddings(
        model="multilingual-22-12", cohere_api_key=cohere_api_key
    )
    store = Qdrant.from_documents(
        pages,
        embeddings,
        location=":memory:",
        collection_name="my_documents",
        distance_func="Dot",
    )
    messages_container = st.container()
    question = st.text_input(
        "", placeholder="Type your message here", label_visibility="collapsed"
    )

    if st.button("Run", type="secondary"):
        prompt.append({"role": "user", "content": question})
        chain_type_kwargs = {"prompt": PROMPT}
        with messages_container:
            user_message(question)
            botmsg = bot_message("...", bot_name="Multilingual Personal Chat Bot")

        qa = RetrievalQA.from_chain_type(
            llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
            chain_type="stuff",
            retriever=store.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )

        answer = qa({"query": question})
        result = answer["result"].replace("\n", "").replace("Answer:", "")

        with st.spinner("Loading response .."):
            botmsg.update(result)

        prompt.append({"role": "assistant", "content": result})

    st.session_state["prompt"] = prompt
else:
    st.session_state["prompt"] = None
    st.warning("No file found. Upload a file to chat!")