import os, sys, pickle, shutil
print("Using Python:", sys.executable)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.schema import Document
from typing import TypedDict, List
from pydantic import BaseModel, Field


def setup_rag():
    # Load persisted DB instead of building it
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    db = Chroma(
        embedding_function=embedder,
        persist_directory="chroma_store"
    )
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the following context and the Chathistory.
        Especially take the latest question into consideration:

        Chathistory: {history}
        Context: {context}
        Question: {question}
    """)
    rag_chain = prompt | llm

    class AgentState(TypedDict):
        messages: List[BaseMessage]
        documents: List[Document]
        on_topic: str
        rephrased_question: str
        proceed_to_generate: bool
        rephrase_count: int
        question: HumanMessage

    class GradeQuestion(BaseModel):
        score: str = Field(description="Is the question on topic? Answer 'Yes' or 'No'")

    class GradeDocument(BaseModel):
        score: str = Field(description="Is the document relevant? Answer 'Yes' or 'No'")

    def question_rewriter(state: AgentState):
        state.update({"documents": [], "on_topic": "", "rephrased_question": "",
                      "proceed_to_generate": False, "rephrase_count": 0})

        if "messages" not in state or state["messages"] is None:
            state["messages"] = []

        if state["question"] not in state["messages"]:
            state["messages"].append(state["question"])

        if len(state["messages"]) > 1:
            conversation = state["messages"][:-1]
            current_question = state["question"].content
            messages = [SystemMessage(content="You rephrase questions to be standalone.")]
            messages.extend(conversation)
            messages.append(HumanMessage(content=current_question))
            rephrase_prompt = ChatPromptTemplate.from_messages(messages)
            response = llm.invoke(rephrase_prompt.format())
            state["rephrased_question"] = response.content.strip()
        else:
            state["rephrased_question"] = state["question"].content
        return state

    def question_classifier(state: AgentState):
        state["on_topic"] = 'Yes'
        return state

    def on_topic_router(state: AgentState):
        return "retrieve" if state.get("on_topic", "").lower() == "yes" else "off_topic_response"

    def retrieve(state: AgentState):
        state["documents"] = retriever.invoke(state["rephrased_question"])
        return state

    def retrieval_grader(state: AgentState):
        structured_llm = llm.with_structured_output(GradeDocument)
        relevant_docs = []
        for doc in state["documents"]:
            msg = HumanMessage(content=f"User question: {state['rephrased_question']}\n\nRetrieved document:\n{doc.page_content}")
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Is the document relevant?"),
                msg
            ])
            result = (prompt | structured_llm).invoke({})
            if result.score.strip().lower() == "yes":
                relevant_docs.append(doc)
        state["documents"] = relevant_docs
        state["proceed_to_generate"] = len(relevant_docs) > 0
        return state

    def proceed_router(state: AgentState):
        if state.get("proceed_to_generate", False):
            return "generate_answer"
        elif state.get("rephrase_count", 0) >= 2:
            return "cannot_answer"
        else:
            return "refine_question"

    def refine_question(state: AgentState):
        refine_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Slightly refine this question to improve results."),
            HumanMessage(content=f"Original question: {state['rephrased_question']}")
        ])
        response = llm.invoke(refine_prompt.format())
        state["rephrased_question"] = response.content.strip()
        state["rephrase_count"] += 1
        return state

    def generate_answer(state: AgentState):
        response = rag_chain.invoke({
            "history": state["messages"],
            "context": state["documents"],
            "question": state["rephrased_question"]
        })
        state["messages"].append(AIMessage(content=response.content.strip()))
        return state

    def cannot_answer(state: AgentState):
        state["messages"].append(AIMessage(content="I'm sorry, I couldn’t find an answer."))
        return state

    def off_topic_response(state: AgentState):
        state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question."))
        return state

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

    graph = workflow.compile(checkpointer=MemorySaver())
    return graph, retriever
