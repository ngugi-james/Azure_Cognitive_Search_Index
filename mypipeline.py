import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Step 3: Configure Azure Cognitive Search Client
index_name = "azure-james-demo"
endpoint = "the endpoint seen on the seaech service on Azure portal"
key = "Go to Setting > Keys"

credential = AzureKeyCredential(key)
client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

# Step 4: Configure LLM Model
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_KEY"] = "Use the API key 1 on the home tab or the API key on the model deployment tab"
os.environ["AZURE_OPENAI_ENDPOINT"] = "Use the OpenAI endpoint on the home tab not the endpoint on the deployed model. The one on the hometab should be shorter than the one on the deployed model"

llm = AzureOpenAI(deployment_name="cst-gpt-35-turbo-instruct", model="gpt-35-turbo-instruct", temperature=1)

# Step 5: Load and Process PDF Data
pdf_link = "DeepLearning.pdf"  # Ensure this file is in the same directory
loader = PyPDFLoader(pdf_link, extract_images=False)
data = loader.load_and_split()

# Split data into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap  = 20,
    length_function = len
)
chunks = text_splitter.split_documents(data)

# Step 6: Store Data in Azure Search Index
for index, chunk in enumerate(chunks):
    data = {
        "id": str(index + 1),
        "data": chunk.page_content
    }  # Removed 'source' field

    result = client.upload_documents(documents=[data])

# Step 7: Create a Function for Retrieval Augmented Generation (RAG)
def generate_response(user_question):
    # Fetch relevant chunks from Azure Search Index
    context = ""
    results = client.search(search_text=user_question, top=2)

    for doc in results:
        context += "\n" + doc['data']

    print("Context Retrieved:\n", context)

    # Construct prompt for LLM
    qna_prompt_template = f"""You will be provided with a question and related context. Answer the question using only the context.

Context:
{context}

Question:
{user_question}

If the context does not contain the answer, respond with "I don't have enough information to answer this question."

Answer:"""

    # Generate response using LLM
    response = llm.invoke(qna_prompt_template)
    return response

# Step 8: Test the System
if __name__ == "__main__":
    user_question = "What is the general focus of machine learning?"
    response = generate_response(user_question)
    print("Answer:", response)