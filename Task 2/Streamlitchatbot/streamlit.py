import streamlit as st
from importnb import Notebook
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import GooglePalm
from langchain.vectorstores import Chroma
from langchain.embeddings import GooglePalmEmbeddings



st.title("Chatbot using GPT-3")
st.write("Hello! I am a chatbot powered by GPT-3. How can I assist you today?")
def process_qa_retrieval_chain(chain, query):
    response = chain.invoke({'query': query})
    
    result_str = f'Query: {response["query"]}\n\n'
    result_str += f'Result: {response["result"]}\n\n'
    
    relevant_docs = response['source_documents']
    for i in range(len(relevant_docs)):
        result_str += f'Relevant Doc {i+1}:\n'
        result_str += relevant_docs[i].page_content + '\n'
        result_str += str(relevant_docs[i].metadata) + '\n\n'
    
    return result_str
def main():
    
    llm = GooglePalm(google_api_key='AIzaSyDnK2exZbDeb1BT36mv5-EaPm2afNzTMUI', temperature=0)
    embedding = GooglePalmEmbeddings(google_api_key='AIzaSyDnK2exZbDeb1BT36mv5-EaPm2afNzTMUI')
    vectordb = Chroma(
    persist_directory='../data/vector_db/chroma_pdfs/',
    embedding_function=embedding  # Pass the same embedding function used during creation
)
    if 'conversation' not in st.session_state:
        st.session_state.messages = []

    # Chat input box
    user_input = st.text_input("You: ", "")
    
    if st.button("Send") and user_input:
        # Add user message to chat history
        st.session_state.messages.append(f"You: {user_input}")

        template = """Based on the provided context, answer the following question to the best of your ability. 
Ensure that your response demonstrates a clear understanding of the context and provides a thoughtful, creative, and useful answer. 
If the context does not contain enough information to fully answer the question, please indicate that and explain why. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={'prompt': QA_CHAIN_PROMPT}
)
        # Get GPT-3 response
        result = process_qa_retrieval_chain(qa_chain, user_input)
        print(result)
        
        # Add GPT-3 response to chat history
        st.session_state.messages.append(f"Bot: {result}")
        
    # Display chat history
    for message in st.session_state.messages:
        st.write(message)

if __name__ == "__main__":
    main()
