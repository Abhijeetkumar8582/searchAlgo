import langchain as lc
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

app = Flask(__name__)

load_dotenv('file.env')

loader = PyPDFLoader("Abhijeet_Kumar_CV.pdf")
pages = loader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
texts = text_splitter.split_documents(pages)


prompt_template = """Your goal is to answer user question from data you are getting and if you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in english:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))


db = Chroma.from_documents(texts, OpenAIEmbeddings())

@app.route('/')
def home():
    return "Hello, Flask!"


@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        data = request.json
        user_input = data['user_input']
        print(os.getenv('OPENAI_API_KEY'))

        # Generate a question based on user input
        openai.api_key = os.getenv('OPENAI_API_KEY')
        completion = openai.Completion.create(
            model="text-davinci-003",
            max_tokens=150,
            temperature=0.1,
            top_p=1,
            prompt=f"Generate only one of insightful and detailed questions based on the following topic: {user_input}.Don't change the meaning of the question what user asked just optimize that."
        )
        optimize_query = completion.choices[0].text
        llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'),
                     temperature=0.1, top_p=1, model='text-embedding-ada-002')

        chain = RetrievalQA.from_chain_type(llm=OpenAI(
        ), chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)

        # Retrieve answer using the question
        result = chain.run(optimize_query)
        vectorizer = CountVectorizer().fit_transform([optimize_query, result])
        cosine_sim = cosine_similarity(vectorizer)
        similarity_score = cosine_sim[0, 1]

        return jsonify({'question': optimize_query, 'answer': result,'similarity_score':similarity_score})

    except Exception as e:
        return jsonify({'error': str(e)})


def generate_answer(question):
   
    return "This is a placeholder answer."


if __name__ == '__main__':
    app.run(debug=True)