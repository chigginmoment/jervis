import os
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate
import time

start = time.time()

chroma_db_path = "chroma_db"
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_function)

template="""
<｜begin▁of▁sentence｜>
You are a helpful digital assistant. You provide answers as accurately as you can, but you never lie.
If unsure at all about something, you answer "I am unsure".
You are tasked with receiving context and extracting the information that is relevant to the user's query.
Think about what the user is asking and the best way to understand the context before constructing an answer.
Once you're done thinking, answer in a single sentence as concisely as possible. If the user query is unrelated
to your provided context, refuse to answer the question.
You are not asked to extrapolate any hidden information from the context. Simply state the most relevant point.

You will mainly be reading wiki pages. Here are some helpful hints:
# For World of Tanks wiki pages:
- The penetration format is in [standard round penetration]/[premium round penetration]/[explosive round penetration] format, in millimeters.
- When reading about stats in the Performance section, they are often presented backwards. E.g. 6,100,000\nCost means Cost: 6,100,000.
- Acronyms for rounds are as follows: AP=>Armor Piercing, APCR=>Armor Piercing Composite Rigid, HEAT=>High Explosive Anti Tank, HE=>High Explosive, HESH=>High Explosive Squash Head
- You may receive multiple documents. Remember that each document begins with a roman numeral from I to X indicating tier, and an integer indicating cost. THESE CONTEXTS ARE NOT RELATED TO EACH OTHER. 
- If answering a question about a specific vehicle, only one context will apply - because the individual contexts are about individual vehicles.

<context>
{context}
</context>

<｜User｜>
{question}

<｜Assistant｜>
"""

custom_prompt = PromptTemplate(input_variables=['context', 'question'], 
                               input_types={}, 
                               partial_variables={}, 
                               template=template)

n_gpu_layers = -1
n_batch = 512 

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    # callback_manager=callback_manager,
    max_tokens=10000,
    n_ctx = 4096,
    # verbose=True,  # Verbose is required to pass to the callback manager
)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=2)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = custom_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    print(docs_content)
    return {"answer": response}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

question = input()
response = graph.invoke({"question": question})
print(response["answer"])
end = time.time() - start
print("runtime:", end)