from sentence_transformers import SentenceTransformer
import math

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

sentences = [f"{query}",
            "The Sun is a star located at the center of our solar system.",
            "Jupiter is the largest planet in the solar system.",
            "The Moon orbits the Earth approximately every 27 days.",
            "Mars is often called the Red Planet because of its iron oxide surface.",
            "Elephants are the largest land animals on Earth.",
            "Dolphins are highly intelligent marine mammals.",
            "Cheetahs are the fastest land animals, reaching speeds over 100 km/h.",
            "Owls can rotate their heads almost 270 degrees.",
            "Pizza originated in Naples, Italy.",
            "Honey never spoils if stored properly.",
            "Coffee beans are actually seeds from a fruit called a coffee cherry.",
            "Basketball was invented by James Naismith in 1891.",
            "A marathon is 42.195 kilometers long.",
            "Soccer is the most popular sport in the world.",
            "The first computer mouse was made of wood.",
            "Smartphones typically contain more computing power than early spacecraft.",
            "The internet was originally developed for military and academic research."
]
vector = model.encode(sentences)
store=[]
print(vector.shape)
for i in range(len(sentences)-1):
    store.append((cosine_similarity(vector[0],vector[i+1]),sentences[i+1]))

final_store= sorted(store)
for i in range(1,4):
    print(final_store[-i])