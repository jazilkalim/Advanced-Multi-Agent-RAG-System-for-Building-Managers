

def ragoutput(query):
    input_data = {"question": HumanMessage(content=query)}
    response=graph.invoke(input=input_data, config={"configurable": {"thread_id": 9}})
    return response['messages'][1].content
