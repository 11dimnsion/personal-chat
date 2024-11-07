import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from agent.rag_prompt import rag_prompt
from langchain_core.tools import BaseTool, tool
from langchain import hub


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class JSONRAGTool:
    def __init__(self, folder_paths):
        """
        Initializes the JSON RAG tool.
        Args:
            folder_paths (list): List of folder paths to scan for JSON files.
        """
        self.folder_paths = folder_paths
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.rag_chain = None

    def load_jsons_from_folders(self):
        """
        Loads JSON documents from the specified folders and returns a list of Document objects.
        Returns:
            List[Document]: A list of Document objects extracted from JSON files.
        """
        documents = []

        # Iterate through each folder path
        for folder_path in self.folder_paths:
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path, file_name)

                    # Use JSONLoader to read the JSON file
                    try:
                        loader = JSONLoader(file_path, jq_schema='.[].question + " " + .[].answer')
                        print("Loader complete")
                        docs = loader.load()
                        print("Docs created")
                        documents.extend(docs)
                        print("Docs extended")
                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")

        return documents

    def build_vector_store(self, documents):
        """
        Builds a vector store from the given documents.
        Args:
            documents (list): List of Document objects to vectorize.
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def setup_chain(self):
        """
        Sets up the LCEL chain using the vector store.
        """
        if self.vector_store is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            relative_path = os.path.join(script_dir, 'FAQFiles')
            folder_paths = [relative_path]

            print(f"Folder paths: {folder_paths}")

            # Initialize the JSONRAGTool
            json_rag = JSONRAGTool(folder_paths)
            documents = json_rag.load_jsons_from_folders()
            if documents:
                print(f"Loaded {len(documents)} documents.")
                # Step 2: Build the vector store from loaded documents
                json_rag.build_vector_store(documents)

                # Step 3: Set up the LCEL chain
                json_rag.setup_chain()
            else:
                print("No documents loaded.")
            
                        #raise ValueError("Vector store is not initialized. Load documents and build the vector store first.")

        # Define the retriever
        retriever = self.vector_store.as_retriever()

        # Initialize the language model
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = hub.pull("rlm/rag-prompt")

        # Create the LCEL chain
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
        )

        #print(self.rag_chain.invoke("What is Task Decomposition?"))

    def run(self, query: str):
        """
        Runs a query through the LCEL chain.
        Args:
            query (str): The query to run.
        Returns:
            str: The response from the LCEL chain.
        """
        if self.rag_chain is None:
            # If the chain is not set up, set it up
            self.setup_chain()
            #raise ValueError("Chain is not set up. Build the vector store and set up the chain first.")
        #return self.rag_chain.invoke({"input": query})["output"]
        return self.rag_chain.invoke(query)

# Set up the relative path to the local directory 'RAGFiles'
script_dir = os.path.dirname(os.path.realpath(__file__))
relative_path = os.path.join(script_dir, 'FAQFiles')
folder_paths = [relative_path]

# Initialize the JSONRAGTool
json_rag = JSONRAGTool(folder_paths)
documents = json_rag.load_jsons_from_folders()

if documents:
    print(f"Loaded {len(documents)} documents.")
    # Step 2: Build the vector store from loaded documents
    json_rag.build_vector_store(documents)

    # Step 3: Set up the LCEL chain
    json_rag.setup_chain()
else:
    print("No documents loaded.")


json_rag_tool: BaseTool = tool(json_rag.run)
json_rag_tool.name = "JSONRAGTool"

def main():
    # Step 4: Test the tool with a sample query
    query = "Do you offer gift cards?"
    response = json_rag.run(query)
    print("Response:", response)


if __name__ == "__main__":
    main()