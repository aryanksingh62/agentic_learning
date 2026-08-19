from sentence_transformers import SentenceTransformer
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

input_file="data/article.txt"

def chunking():
    with open(input_file,"r") as f:
        data= f.read()

        chunks=[]
        i=0
        while i<len(data):
            chunks.append(data[i:i+100])
            i+=100
        return chunks

def retrieval(query_vector,vectors,sentences):
    store=[]
    for i in range(len(sentences)):
        store.append((cosine_similarity(query_vector,vectors[i]),sentences[i]))

    final_store= sorted(store)

    text_chunks=""
    for i in range(1,3):
        text_chunks = text_chunks + final_store[-i][1] +"\n"

    return text_chunks
        
def cosine_similarity(a,b):
    dot=0
    sqr_a=0
    sqr_b=0
    for i in range(len(a)):
        dot+= a[i]*b[i]
        sqr_a+= a[i]**2
        sqr_b+= b[i]**2
    sum_of_mag = math.sqrt(sqr_a)* math.sqrt(sqr_b)
    cosine= dot/sum_of_mag
    return cosine

query= input("tell me whats's the problem: ").strip()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences= chunking()
vectors= model.encode(sentences)
query_vector= model.encode(query)

context= retrieval(query_vector,vectors,sentences)
print("retrived context\n",context)

augmented_prompt = f"""Answer the question using ONLY the information in the context below.
Do not use any outside knowledge, even if you know the answer.
If the answer is not contained in the context, respond exactly with: "I don't know based on the provided context."

Context:
{context}

Question: {query}
"""

response = client.responses.create(
    model="gpt-5.4-mini",
    input= augmented_prompt
)
print("-"*15)
print(response.output_text)