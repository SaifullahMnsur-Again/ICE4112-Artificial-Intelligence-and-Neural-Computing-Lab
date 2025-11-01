knowledge_base = {
    'cold': ['sneezing', 'runny_nose'],
    'flu': ['fever', 'body_ache', 'fatigue']
}

symptoms = {
    'john': ['sneezing', 'runny_nose', 'headache'],
    'mary': ['fever', 'body_ache', 'fatigue']
}

def diagnose(person):
    for disease, disease_symptom in knowledge_base.items():
        disease_found = True
        for person_symptoms in symptoms[person]:
            if person_symptoms not in disease_symptom:
                disease_found = False
                break
        if disease_found:
            print(f"{person} has {disease}")
    
def main():
    for person in symptoms.keys():
        diagnose(person)

if __name__ == '__main__':
    main()