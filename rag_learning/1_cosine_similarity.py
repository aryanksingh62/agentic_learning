import math


documents = {
    "A": [1.0, 0.0],
    "B": [0.9, 0.1],
    "C": [0.0, 1.0],
}

query = [1.0, 0.0]

def cosine_similarity(a,b):
    dot= a[0]*b[0] + a[1]*b[1]
    mag_a= math.sqrt((a[0])**2 + a[1]**2)
    mag_b= math.sqrt((b[0])**2 + b[1]**2)
    sum_of_mag= mag_a * mag_b
    cosine= dot/sum_of_mag
    return cosine

print(cosine_similarity(query, documents["A"]))
print(cosine_similarity(query, documents["B"]))
print(cosine_similarity(query, documents["C"]))