from langchain.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the user mentions that the situation is critical or an emergency, list out remedies in simple steps. "
    "Otherwise, provide a normal answer. Always format the answer clearly. "
    "If you don't know the answer, say that you don't know.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
