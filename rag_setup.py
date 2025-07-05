# setup.py
import os
import sys
print("Using Python:", sys.executable)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings




import pickle

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db = None
retriever = None
import shutil

shutil.rmtree("chroma_store", ignore_errors=True)  # or your actual directory name

from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(chunks, embedding=embedder,persist_directory="chroma_store")
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})


from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up Gemini (gpt-4o equivalent from Google)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

template = """Answer the question based on the following context and the Chathistory. Especially take the latest question into consideration:

Chathistory: {history}

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm


from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI


class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage


class GradeQuestion(BaseModel):
    score: str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    )


def question_rewriter(state: AgentState):
    print(f"Entering question_rewriter with following state: {state}")

    # Reset state variables except for 'question' and 'messages'
    state["documents"] = []
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        current_question = state["question"].content
        messages = [
            SystemMessage(
                content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."
            )
        ]
        messages.extend(conversation)
        messages.append(HumanMessage(content=current_question))
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        #llm = ChatOpenAI(model="gpt-4o-mini")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        prompt = rephrase_prompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        print(f"question_rewriter: Rephrased question: {better_question}")
        state["rephrased_question"] = better_question
    else:
        state["rephrased_question"] = state["question"].content
    return state

def question_classifier(state: AgentState):
    # print("Entering question_classifier")
    # system_message = SystemMessage(
    #     content=""" You are a classifier that determines whether a user's question is about one of the following topics 
    
    
    # 1. Building Manager Team  
    # 2. Learning Outcomes and Objectives  
    # 3. Dress Code  
    # 4. Management Hierarchy  
    # 5. Who Ya Gonna Call?  
    # 6. Shift Scheduling  
    # 7. Find a Cover/Trade  
    # 8. Discipline  
    # 9. Shift Reports  
    # 10. Shift Change Chats  
    # 11. Radio Etiquette  
    # 12. Timesheet Edits Form  
    # 13. The Event Schedule  
    # 14. 7Point Ops  
    # 15. Set Up Notes  
    # 16. Determining AV Needs  
    # 17. Function Housekeeper Info  
    # 18. Building Rounds  
    # 19. Locking Rooms  
    # 20. Opening and Closing the Buildings  
    # 21. Opening Checklist  
    # 22. Closing Checklist  
    # 23. Mandel Hall Closing  
    # 24. Which Spaces Are Open to the Public  
    # 25. Building Access and Double Tap  
    # 26. Building Partners  
    # 27. Managing Client Reservations  
    # 28. Notes for RSO Events  
    # 29. Notes for Departmental and External Events  
    # 30. Audio Visual Equipment and Set Ups  
    # 31. Loaning Out Miscellaneous Items  
    # 32. Keys  
    # 33. Moving Furniture  
    # 34. Furniture Storage Locations  
    # 35. Managing and Requesting Custodial/Maintenance Services  
    # 36. Function Housekeeper  
    # 37. The Mail  
    # 38. Packages at Ida Noyes  
    # 39. Lost and Found  
    # 40. Answering the Phone(s)  
    # 41. Transferring Calls  Reynolds Club  
    # 42. Transferring Calls Ida Noyes Hall  
    # 43. Emergency Procedures  
    # 44. Troubleshooting  
    # 45. Appendices  Key Lists  
    # 46. General Shift Expectation Cheat Sheet  
    # 47. Campus Resources
    # 48. Desk Expectations 
    # 49. Workday
    # 50. Fire Safety
    # 60. Children and Family Services
    # 61. Building Manager Dress Code
    
    
    # If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'.

    # """
    # )

    # human_message = HumanMessage(
    #     content=f"User question: {state['rephrased_question']}"
    # )
    # grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    # #llm = ChatOpenAI(model="gpt-4o")
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    # structured_llm = llm.with_structured_output(GradeQuestion)
    # grader_llm = grade_prompt | structured_llm
    # result = grader_llm.invoke({})
    #state["on_topic"] = result.score.strip()
    state["on_topic"] = 'Yes'
    print(f"question_classifier: on_topic = {state['on_topic']}")
    return state

def on_topic_router(state: AgentState):
    print("Entering on_topic_router")
    on_topic = state.get("on_topic", "").strip().lower()
    if on_topic == "yes":
        print("Routing to retrieve")
        return "retrieve"
    else:
        print("Routing to off_topic_response")
        return "off_topic_response"


def retrieve(state: AgentState):
    print("Entering retrieve")
    documents = retriever.invoke(state["rephrased_question"])
    print(f"retrieve: Retrieved {len(documents)} documents")
    state["documents"] = documents
    return state


class GradeDocument(BaseModel):
    score: str = Field(
        description="Document is relevant to the question? If yes -> 'Yes' if not -> 'No'"
    )

def retrieval_grader(state: AgentState):
    print("Entering retrieval_grader")
    system_message = SystemMessage(
        content="""You are a grader assessing the relevance of a retrieved document to a user question.
Only answer with 'Yes' or 'No'.

If the document contains information relevant to the user's question, respond with 'Yes'.
Otherwise, respond with 'No'."""
    )

    #llm = ChatOpenAI(model="gpt-4o")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    structured_llm = llm.with_structured_output(GradeDocument)

    relevant_docs = []
    for doc in state["documents"]:
        human_message = HumanMessage(
            content=f"User question: {state['rephrased_question']}\n\nRetrieved document:\n{doc.page_content}"
        )
        grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        grader_llm = grade_prompt | structured_llm
        result = grader_llm.invoke({})
        print(
            f"Grading document: {doc.page_content[:30]}... Result: {result.score.strip()}"
        )
        if result.score.strip().lower() == "yes":
            relevant_docs.append(doc)
    state["documents"] = relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) > 0
    print(f"retrieval_grader: proceed_to_generate = {state['proceed_to_generate']}")
    return state

def proceed_router(state: AgentState):
    print("Entering proceed_router")
    rephrase_count = state.get("rephrase_count", 0)
    if state.get("proceed_to_generate", False):
        print("Routing to generate_answer")
        return "generate_answer"
    elif rephrase_count >= 5:
        print("Maximum rephrase attempts reached. Cannot find relevant documents.")
        return "cannot_answer"
    else:
        print("Routing to refine_question")
        return "refine_question"
    
def refine_question(state: AgentState):
    print("Entering refine_question")
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        print("Maximum rephrase attempts reached")
        return state
    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(
        content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
Provide a slightly adjusted version of the question."""
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
    )
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    #llm = ChatOpenAI(model="gpt-4o")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = refine_prompt.format()
    response = llm.invoke(prompt)
    refined_question = response.content.strip()
    print(f"refine_question: Refined question: {refined_question}")
    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1
    return state

def generate_answer(state: AgentState):
    print("Entering generate_answer")
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer.")

    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    response = rag_chain.invoke(
        {"history": history, "context": documents, "question": rephrased_question}
    )

    generation = response.content.strip()

    state["messages"].append(AIMessage(content=generation))
    print(f"generate_answer: Generated response: {generation}")
    return state

def cannot_answer(state: AgentState):
    print("Entering cannot_answer")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(
            content="I'm sorry, but I cannot find the information you're looking for."
        )
    )
    return state


def off_topic_response(state: AgentState):
    print("Entering off_topic_response")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question!"))
    return state

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

# Workflow
workflow = StateGraph(AgentState)
workflow.add_node("question_rewriter", question_rewriter)
workflow.add_node("question_classifier", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieval_grader", retrieval_grader)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("refine_question", refine_question)
workflow.add_node("cannot_answer", cannot_answer)

workflow.add_edge("question_rewriter", "question_classifier")
workflow.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {
        "retrieve": "retrieve",
        "off_topic_response": "off_topic_response",
    },
)
workflow.add_edge("retrieve", "retrieval_grader")
workflow.add_conditional_edges(
    "retrieval_grader",
    proceed_router,
    {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer",
    },
)
workflow.add_edge("refine_question", "retrieve")
workflow.add_edge("generate_answer", END)
workflow.add_edge("cannot_answer", END)
workflow.add_edge("off_topic_response", END)
workflow.set_entry_point("question_rewriter")
graph = workflow.compile(checkpointer=checkpointer)







# def load_and_embed_documents(pdf_folder="docs", persist_directory="chromadb"):
#     loaders = [PyPDFLoader(os.path.join(pdf_folder, file)) 
#                for file in os.listdir(pdf_folder) if file.endswith(".pdf")]
    
#     documents = []
#     for loader in loaders:
#         documents.extend(loader.load())
    
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = text_splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
#     db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_directory)
#     return db
