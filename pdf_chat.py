import io
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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

def get_relevant_chunks(query, index, text_chunks, embeddings, model, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=k)
    return [text_chunks[i] for i in I[0]], I[0]

def generate_response(contexts, query, tokenizer, model):
    combined_context = " ".join(contexts)
    
    input_text = f"""Instructions: Provide a factual answer based solely on the given context. Do not make assumptions or add information not present in the context.

Context: {combined_context}

Question: {query}

Answer:"""

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,  # Control the length of the generated response
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()
    combined_context = " ".join(contexts)
    
    input_text = f"""Instructions: Provide a factual answer based solely on the given context. Do not make assumptions or add information not present in the context.

Context: {combined_context}

Question: {query}

Answer:"""

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

class PDFChatBot:
    def __init__(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.generation_model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.index = None
        self.embeddings = None
        self.text_chunks = None

    def process_pdfs(self, pdf_docs):
        raw_text = get_pdf_text(pdf_docs)
        self.text_chunks = get_text_chunks(raw_text)
        self.index, self.embeddings = get_vectorstore(self.text_chunks, self.embedding_model)

    def ask_question(self, question):
        contexts, idxs = get_relevant_chunks(question, self.index, self.text_chunks, self.embeddings, self.embedding_model, k=3)
        response = generate_response(contexts, question, self.tokenizer, self.generation_model)
        return response

# Example usage
if __name__ == "__main__":
    chatbot = PDFChatBot()
    
    # Assume pdf_docs is a list of PDF file contents
    pdf_docs = [...]  # Load your PDF documents here
    chatbot.process_pdfs(pdf_docs)
    
    while True:
        question = input("Ask a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer = chatbot.ask_question(question)
        print(f"Answer: {answer}")
