import numpy as np
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory, CombinedMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.output_parsers import PydanticOutputParser

import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import typing

import os
import sys
import pandas as pd
import re

import uvicorn

# import nltk 
# nltk.download('punkt')



class LoaderCSV:
    def __init__(self, path):
        self.path = path

    def load(self):
        csv_file = pd.read_csv("./tinkoff-terms/cards.csv")
        csv_chunks = []
        for i in range(csv_file.shape[0]):
            row = csv_file.iloc[i]
            sen = f"{row.Service} {row.Condition} - {row.Tariff}"
            csv_chunks.append(Document(page_content=sen))
        return csv_chunks

# @app.get("/")
# def get_question():
#     return {"message": input()}


def load_model():
    loader = DirectoryLoader('./tinkoff-terms/', glob="**/*.pdf")
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200,
                                          chunk_overlap=10, length_function=len,
                                          is_separator_regex=False)
    text = loader.load()
    chunks = text_splitter.split_documents(text)

    csv_chunks = LoaderCSV("./tinkoff-terms/cards.csv").load()

    model_name = "ai-forever/sbert_large_nlu_ru"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("DB CREATION")
    db = Chroma.from_documents(csv_chunks + chunks, embeddings_model)

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    print("LLAMA LOAD")
    llm = LlamaCpp(
        model_path="./llama-2-7b-chat.Q4_K_M.gguf",
        temperature=0.01,
        max_tokens=200,
        n_ctx=2048,
        # ctx_size=2048,
        # repeat_last_n=2048,
        thread=10,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    class PlaceOrder(BaseModel):
        service: str = Field(description='Ответ: ')

    parser = PydanticOutputParser(pydantic_object=PlaceOrder)
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                'Ты сотрудник-консультант банка Тинькофф.\n'
                'Ты коротко отвечаешь только на русском языке на вопросы пользователей об услугах, условиях и тарифах банка\n'
                'Используй только информацию приведенную ниже\n'
                '{context}\n'
                'Если информации по вопросу нет, ответь: _НЕ_НАЙДЕНО_ \n',),
            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate.from_template('{question}.')
        ],
        partial_variables={
            'format_instructions': parser.get_format_instructions()
        }
    )
    return llm, prompt, parser, db


class Message(BaseModel):
    message: str
    user_id : str

llm, prompt, parser, db = load_model()
redis_backed_dict = {}

print("APP creation")
app = FastAPI()

@app.post('/message')
async def message(message: Message):
    user_id, message = message.user_id, message.message

    memory = redis_backed_dict.get(
        user_id,
        ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
    )
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    docs = db.similarity_search(message)

    cum_len = np.array([len(d.page_content) for d in docs[:]]).cumsum()
    # n_doc = (cum_len > 1000).argmax() + \
    #     1 if (cum_len <= 1000).any() else None
    # if n_doc >= len(docs):
    #     n_doc = None

    docs_page_content = " ".join([d.page_content for d in docs[:]])
    ai_message = conversation(
        {'question': message, 'context': docs_page_content})
    if "НЕ_НАЙДЕНО" in ai_message['text']:
        return {'message': "К сожалению не могу ответить на Ваш вопрос."
                "Обратитесь к оператору по горячей линии: 8(800)555 35-35"}
    try:
        command: Optional[PlaceOrder] = parser.parse(ai_message['text'])
        if command is not None:
            return {'message': 'Условия по интересующей вас услуге предоставлены.'}
    except:
        pass

    return {'message': ai_message['text']}




    #uvicorn.run(app, host="localhost", port=8010)

    # print("Ctrl+C чтобы закончить диалог")
    # print("Введите ваш id (например 0): ")
    # user_id = int(input())
    # while True:
    #     print('Введите Ваш вопрос: ')
    #     question = input()
    #     message(user_id, question)
