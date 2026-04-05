import csv, random

random.seed(42)

regions = ['Dakar','Thiès','Saint-Louis','Tambacounda','Ziguinchor',
           'Kaolack','Diourbel','Louga','Fatick','Kédougou']

def generer_patient():
    diag = random.choice(['paludisme','grippe','typhoide','sain'])
    age = random.randint(5, 80)
    sexe = random.choice(['M','F'])
    if diag == 'sain':
        temp = round(random.uniform(36.0, 37.5), 1)
        toux = frissons = nausee = maux_tete = fatigue = 0
    elif diag == 'grippe':
        temp = round(random.uniform(38.0, 39.5), 1)
        toux, fatigue, maux_tete = random.randint(0,1), 1, random.randint(0,1)
        frissons, nausee = random.randint(0,1), random.randint(0,1)
    elif diag == 'paludisme':
        temp = round(random.uniform(38.5, 41.5), 1)
        toux, fatigue, maux_tete = random.randint(0,1), 1, 1
        frissons, nausee = 1, random.randint(0,1)
    else:
        temp = round(random.uniform(38.5, 41.0), 1)
        toux, fatigue, maux_tete = random.randint(0,1), 1, 1
        frissons, nausee = random.randint(0,1), 1
    return [age, sexe, temp, random.randint(8,15), toux, fatigue,
            maux_tete, frissons, nausee, random.choice(regions), diag]

rows = [['age','sexe','temperature','tension_sys','toux','fatigue',
         'maux_tete','frissons','nausee','region','diagnostic']]
for _ in range(500):
    rows.append(generer_patient())

with open('data/patients_dakar.csv', 'w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerows(rows)

print(f"CSV créé avec {len(rows)-1} patients !")