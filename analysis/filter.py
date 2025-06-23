'''filter for education-related documents'''

import os
import json

indir = "../final_data"
outdir = "../edu_data"

keywords = [
    "education", "educational", "school", "schooling", "university", "college",
    "degree", "diploma", "graduation", "student", "teacher", "professor",
    "academic", "studies", "learning", "classroom", "curriculum", "homework",
    "scholarship", "tuition", "admissions", "grade", "grades", "tutor", "tutoring"
]
keywords = [k.lower() for k in keywords]

def contains_keyword(s):
    s_lower = s.lower()
    return any(k in s_lower for k in keywords)

total_docs = 0
total_education_docs = 0

for filename in os.listdir(indir):
    if filename.endswith('.json'):
        filepath = os.path.join(indir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        education_docs = [doc for doc in data if contains_keyword(doc)]

        with open(os.path.join(outdir, filename), 'w', encoding='utf-8') as out_f:
            json.dump(education_docs, out_f, ensure_ascii=False, indent=2)

        num_edu = len(education_docs)
        num_total = len(data)
        ratio = 100 * num_edu / num_total if num_total > 0 else 0

        print(f"{filename}: {num_edu} education documents ({ratio:.2f}% of {num_total})")

        total_docs += num_total
        total_education_docs += num_edu
