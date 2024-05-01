from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from os import listdir
from os.path import isfile, join
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

os.environ["NEO4J_URI"] = os.getenv("neo4j_uri")
os.environ["NEO4J_USERNAME"] = os.getenv("neo4j_username")
os.environ["NEO4J_PASSWORD"] = os.getenv("neo4j_password")
os.environ["Database"] = os.getenv("database")
os.environ["OPEAI_API_KEY"] = os.getenv("openai_api_key")
os.environ["AZURE_ENDPOINT"] = os.getenv("azure_endpoint")

class Test:
    def  __init__ (self):
        self.url="neo4j+s://8c62a2a5.databases.neo4j.io:7687",
        self.username="neo4j",
        self.password="qmsm0-kk7MgsJoJCllahRYawJ1o9WwMdSfFnLi-YMoM",
        self.database="neo4j",
        self.graph = Neo4jGraph()
        print(self.graph)

    def get_pdf_text(self,pdf_docs):
        print(pdf_docs)
        documents=[]
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        for pdffile in pdf_docs:
            print(os.getcwd())
            tmp_location = os.path.join('./tmpDir', pdffile.name)
            print(f"------------{tmp_location}-----------------")
            pdf_document=PyPDFLoader(file_path=tmp_location).load()
            document = text_splitter.split_documents(pdf_document)
            documents.append(document)
        return documents

    def load_embedding_model(self):
        embeddings = AzureOpenAIEmbeddings(
        azure_deployment="TextEmbeddingAda002",
        model="text-embedding-ada-002",
        openai_api_key="07b579590a6e48f2993bb34d2fd905f4",
        azure_endpoint="https://instgenaipoc.openai.azure.com/",
        openai_api_type="azure",)
        dimension = 1536
        return embeddings

    def load_llm(self):
        llm = AzureChatOpenAI(deployment_name="chatgpt45turbo",
        model_name="gpt-4",
        azure_endpoint="https://instgenaipoc.openai.azure.com/",
        api_version="2023-05-15",
        openai_api_key="07b579590a6e48f2993bb34d2fd905f4",
        openai_api_type="azure")
        return llm

    def add_graph_db(self,llm,documents):
        llm_transformer = LLMGraphTransformer(llm=llm)
        for doc in documents:
            graph_documents = llm_transformer.convert_to_graph_documents(doc)
            self.graph.add_graph_documents(graph_documents,baseEntityLabel=True,include_source=True)

    def create_vector_index(self,embeddings):
        vector_index = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        #url=NEO4J_URI,
        #username=NEO4J_USERNAME,
        #password=NEO4J_PASSWORD,
        #database=Database,
        url="neo4j+s://8c62a2a5.databases.neo4j.io",
        username="neo4j",
        password="qmsm0-kk7MgsJoJCllahRYawJ1o9WwMdSfFnLi-YMoM",
        database="neo4j",
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
        )
        return vector_index

    def entity_retriever(self):
        llm=self.load_llm()
        #self.graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
        # Extract entities from text
        class Entities(BaseModel):
            """Identifying information about entities."""
            names: List[str] = Field(
                ...,
                description="All the person, organization, or business entities that "
                "appear in the text",
            )
        prompt = ChatPromptTemplate.from_messages(
            [(
                "system",
                "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),])
        entity_chain = prompt | llm.with_structured_output(Entities)
        return entity_chain

    def generate_full_text_query(self,input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # Fulltext index query
    def structured_retriever(self,question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entity_chain=self.entity_retriever()
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    def retriever(self,question: str):
        print(f"Search query: {question}")
        structured_data = self.structured_retriever(question)
        embeddings=self.load_embedding_model()
        vector_index=self.create_vector_index(embeddings)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        final_data = f"""Structured data: {structured_data} Unstructured data: {"#Document ". join(unstructured_data)}"""
        return final_data

    def _format_chat_history(self,chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    def search(self,llm):
        # Condense a chat history and follow-up question into a standalone question
        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
        in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""  # noqa: E501
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
        _search_query = RunnableBranch(
            # If input includes chat_history, we condense it with the follow-up question
            (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
            chat_history=lambda x: self._format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | AzureChatOpenAI(temperature=0,model_name="gpt-4",azure_endpoint="https://instgenaipoc.openai.azure.com/",api_version="2023-05-15",openai_api_key="07b579590a6e48f2993bb34d2fd905f4",openai_api_type="azure")
            | StrOutputParser(),
            ),
            # Else, we have no chat history, so just pass through the question
            RunnableLambda(lambda x : x["question"]),
            )
        template = """Answer only from the following context provided:
        {context}

        Question: {question}
        Use natural language to answer.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            RunnableParallel(
                {"context": _search_query | self.retriever,"question": RunnablePassthrough(),})
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain