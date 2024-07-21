import streamlit as st
from pdf_chat import PDFChatBot

st.set_page_config(page_title="PDF Chat App", page_icon="📚")

@st.cache_resource
def get_pdf_chatbot():
    return PDFChatBot()

st.title("PDF Chat Application")

pdf_chatbot = get_pdf_chatbot()

uploaded_files = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    pdf_docs = [file.read() for file in uploaded_files]
    with st.spinner("Processing PDFs..."):
        pdf_chatbot.process_pdfs(pdf_docs)
    st.success("PDFs processed. You can now ask questions.")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        with st.spinner("Generating response..."):
            response = pdf_chatbot.ask_question(user_question)
        st.write("Answer:", response)

st.sidebar.title("About")
st.sidebar.info("This app allows you to chat with your PDF documents. Upload PDFs and ask questions to get relevant answers.")