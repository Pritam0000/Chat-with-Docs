import io
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_content in pdf_docs:
        pdf_stream = io.BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    chunk_size = 1000
    chunk_overlap = 200
    text_chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        text_chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return text_chunks

def get_vectorstore(text_chunks, model):
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def get_relevant_chunk(query, index, text_chunks, embeddings, model):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=1)
    return text_chunks[I[0][0]], I[0][0]

def generate_response(context, query, tokenizer, model):
    input_text = f"Context: {context} Question: {query}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

class PDFChatBot:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        #self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        #self.generation_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        # Load the tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
        model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')

        self.index = None
        self.embeddings = None
        self.text_chunks = None

    def process_pdfs(self, pdf_docs):
        raw_text = get_pdf_text(pdf_docs)
        self.text_chunks = get_text_chunks(raw_text)
        self.index, self.embeddings = get_vectorstore(self.text_chunks, self.embedding_model)

    def ask_question(self, question):
        context, idx = get_relevant_chunk(question, self.index, self.text_chunks, self.embeddings, self.embedding_model)
        response = generate_response(context, question, self.tokenizer, self.generation_model)
        return response
