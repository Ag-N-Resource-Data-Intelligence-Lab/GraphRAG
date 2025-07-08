import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA, GraphCypherQAChain


def run_rag(
    url: str = "neo4j+s://44e850f8.databases.neo4j.io",
    username: str = "neo4j",
    password: str = "f1nRQlWKd6LriVlCL4UZt_k_MqJtQdtZE30z_Pq5AYA",
    openai_api_key: str = None,
    model_name: str = "gpt-4-turbo",
    prompt: str = None
) -> str:
    # Set OpenAI API Key
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model_name=model_name)

    # Setup Neo4j vector index
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        url=url,
        username=username,
        password=password,
        index_name='Vector',
        node_label="Vector",
        text_node_properties=['name'],
        embedding_node_property='embedding',
    )

    vector_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_index.as_retriever()
    )

    # Create GraphCypherQAChain using Neo4jVector's internal graph
    graph = Neo4jGraph(
    url=url,
    username=username,
    password=password,
    )
    graph.refresh_schema()

    cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        verbose=False,
        allow_dangerous_requests=True
    )

    tools = [
        Tool(name="Vector", func=vector_qa.run, description='Vector search'),
        Tool(name="Graph", func=cypher_chain.run, description='Cypher search'),
    ]

    mrkl = initialize_agent(
        tools, 
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False
    )

    # Prepare prompt if not provided
    if prompt is None:
        with open('../data/doc1.txt', "r", encoding="utf-8") as f:
            doc = f.read()
        prompt = (
            "Explain the data rights related terms in this document, "
            "including the data type, relations between stakeholders involving data right: "
            + doc
        )

    # Run LLM agent
    response = mrkl.run(prompt)
    return response
