"""
Sample Medical Data Generator
Creates a comprehensive medical knowledge base for the diagnosis assistant
"""

import json


def create_comprehensive_medical_data():
    """Create a comprehensive sample medical dataset."""
    
    medical_data = [
        # Respiratory Conditions
        {
            "id": "resp_001",
            "condition": "Common Cold",
            "symptoms": "runny nose, sneezing, sore throat, mild cough, congestion, low-grade fever (99-100°F), watery eyes",
            "treatment": "Rest, stay hydrated (8-10 glasses of water daily), OTC pain relievers (acetaminophen or ibuprofen), throat lozenges, humidifier use, saline nasal drops",
            "additional_info": "Usually resolves in 7-10 days without treatment. Viral infection of upper respiratory tract. Most contagious during first 2-3 days. Not preventable by antibiotics."
        },
        {
            "id": "resp_002",
            "condition": "Influenza (Flu)",
            "symptoms": "sudden high fever (100-104°F), severe body aches, extreme fatigue, dry cough, headache, chills, sweating, nasal congestion, sore throat",
            "treatment": "Rest and isolation, plenty of fluids, antiviral medications (oseltamivir/Tamiflu) if started within 48 hours, fever reducers, avoid contact with others for 24 hours after fever breaks",
            "additional_info": "Can lead to pneumonia, hospitalization, or death in high-risk groups. Annual vaccination recommended. Peaks in winter months. Incubation period 1-4 days."
        },
        {
            "id": "resp_003",
            "condition": "Pneumonia",
            "symptoms": "persistent cough with thick yellow/green/bloody phlegm, sharp chest pain worsening with deep breathing, difficulty breathing, high fever (102-105°F), rapid breathing, confusion (in elderly), fatigue, sweating, chills",
            "treatment": "Antibiotics for bacterial pneumonia, hospitalization for severe cases, oxygen therapy if needed, IV fluids, chest physiotherapy, follow-up chest X-ray",
            "additional_info": "Requires immediate medical attention. Can be life-threatening, especially for elderly, infants, and immunocompromised. May require hospitalization. Pneumococcal vaccine available for prevention."
        },
        {
            "id": "resp_004",
            "condition": "Bronchitis",
            "symptoms": "persistent cough with mucus (clear, white, yellow, or green), chest discomfort, mild fatigue, mild fever (100-101°F), shortness of breath, wheezing, lasting 2-3 weeks",
            "treatment": "Rest, fluids, cough suppressants (dextromethorphan), expectorants (guaifenesin), bronchodilators if wheezing, humidifier use, avoid smoking and irritants",
            "additional_info": "Often follows a cold or flu. Acute bronchitis usually improves in 2-3 weeks. Chronic bronchitis (COPD) requires different management. Most cases are viral and don't need antibiotics."
        },
        {
            "id": "resp_005",
            "condition": "Asthma Attack",
            "symptoms": "severe difficulty breathing, tight chest, wheezing, rapid breathing, inability to speak in full sentences, bluish lips or fingernails, severe anxiety, cough that won't stop",
            "treatment": "Use rescue inhaler (albuterol) immediately, sit upright, take slow deep breaths, call 911 if no improvement after inhaler use, emergency room evaluation, oral or IV corticosteroids",
            "additional_info": "Life-threatening emergency requiring immediate treatment. Triggers include allergens, exercise, cold air, stress, respiratory infections. Requires asthma action plan and controller medications."
        },
        
        # Allergies
        {
            "id": "allergy_001",
            "condition": "Allergic Rhinitis (Hay Fever)",
            "symptoms": "sneezing fits, itchy nose and eyes, clear watery nasal discharge, nasal congestion, postnasal drip, no fever, fatigue, dark circles under eyes",
            "treatment": "Antihistamines (loratadine, cetirizine), nasal corticosteroid sprays (fluticasone), avoid known allergens, keep windows closed during high pollen days, shower after outdoor activities, air purifiers",
            "additional_info": "Triggered by environmental allergens like pollen (trees, grass, weeds), dust mites, pet dander, mold. Seasonal or year-round. Can worsen asthma. Immunotherapy available for long-term management."
        },
        {
            "id": "allergy_002",
            "condition": "Food Allergy Reaction",
            "symptoms": "itching or tingling in mouth, hives, swelling of lips/face/throat, difficulty breathing, nausea, vomiting, diarrhea, dizziness, rapid pulse",
            "treatment": "Mild: antihistamines. Severe (anaphylaxis): epinephrine auto-injector (EpiPen) immediately, call 911, lie down with legs elevated, may need second dose after 5-15 minutes",
            "additional_info": "Can be life-threatening. Common triggers: peanuts, tree nuts, shellfish, fish, milk, eggs, wheat, soy. Requires strict avoidance of trigger foods. Always carry epinephrine if prescribed."
        },
        
        # Gastrointestinal
        {
            "id": "gi_001",
            "condition": "Gastroenteritis (Stomach Flu)",
            "symptoms": "watery diarrhea, nausea, vomiting, stomach cramps, low-grade fever, muscle aches, headache, loss of appetite",
            "treatment": "Rest, frequent small sips of clear liquids (water, broth, oral rehydration solutions), gradually introduce bland foods (BRAT diet: bananas, rice, applesauce, toast), avoid dairy temporarily, anti-diarrheal medications if needed",
            "additional_info": "Usually viral (norovirus, rotavirus). Typically resolves in 1-3 days. Risk of dehydration, especially in children and elderly. Wash hands frequently. Seek medical care if severe dehydration symptoms."
        },
        {
            "id": "gi_002",
            "condition": "Food Poisoning",
            "symptoms": "sudden nausea, vomiting, diarrhea (possibly bloody), abdominal cramps, fever, weakness, symptoms starting 1-6 hours after eating contaminated food",
            "treatment": "Rest, clear liquids, avoid solid foods until vomiting stops, oral rehydration solutions, gradually return to normal diet, avoid anti-diarrheal medications if diarrhea is bloody",
            "additional_info": "Common causes: Salmonella, E. coli, Listeria, Campylobacter. Usually resolves in 1-2 days. Seek medical care if severe symptoms, bloody stools, signs of dehydration, or symptoms last >3 days."
        },
        {
            "id": "gi_003",
            "condition": "Appendicitis",
            "symptoms": "sudden pain starting around navel moving to lower right abdomen, pain worsens with movement/coughing, loss of appetite, nausea, vomiting, fever (99-102°F), inability to pass gas, abdominal swelling",
            "treatment": "Emergency surgery (appendectomy) to remove appendix, antibiotics, IV fluids, hospitalization required, laparoscopic or open surgery depending on severity",
            "additional_info": "Medical emergency requiring immediate surgery. Risk of rupture leading to peritonitis (life-threatening infection). Do not delay seeking medical care. Most common in people 10-30 years old."
        },
        
        # Cardiovascular
        {
            "id": "cardio_001",
            "condition": "Heart Attack (Myocardial Infarction)",
            "symptoms": "severe chest pain/pressure (crushing sensation), pain radiating to left arm/jaw/back, shortness of breath, cold sweats, nausea, lightheadedness, sense of impending doom, symptoms may be milder in women/diabetics",
            "treatment": "CALL 911 IMMEDIATELY, chew aspirin (if not allergic), stay calm, sit or lie down, emergency angioplasty or thrombolytic drugs, hospitalization, cardiac rehabilitation",
            "additional_info": "Life-threatening emergency requiring immediate treatment. Every minute counts - permanent heart damage occurs quickly. Risk factors: high blood pressure, high cholesterol, diabetes, smoking, obesity, family history."
        },
        {
            "id": "cardio_002",
            "condition": "Hypertensive Crisis",
            "symptoms": "blood pressure >180/120, severe headache, confusion, vision changes, chest pain, difficulty breathing, irregular heartbeat, seizures, unresponsiveness",
            "treatment": "CALL 911, emergency room evaluation, IV medications to lower blood pressure gradually, monitoring in intensive care, treat underlying complications",
            "additional_info": "Medical emergency. Can cause stroke, heart attack, kidney failure, or death. Often occurs when blood pressure medications are not taken as prescribed or due to drug interactions."
        },
        
        # Neurological
        {
            "id": "neuro_001",
            "condition": "Migraine",
            "symptoms": "severe throbbing headache (often one-sided), nausea, vomiting, extreme sensitivity to light and sound, visual disturbances (aura), dizziness, lasting 4-72 hours",
            "treatment": "Rest in dark quiet room, triptans (sumatriptan) at first sign, NSAIDs (ibuprofen), anti-nausea medications, cold compress, avoid triggers, preventive medications if frequent",
            "additional_info": "Neurological condition with genetic component. Common triggers: stress, certain foods, hormonal changes, sleep changes, weather changes. Keep migraine diary to identify triggers."
        },
        {
            "id": "neuro_002",
            "condition": "Stroke",
            "symptoms": "sudden numbness/weakness of face/arm/leg (especially one side), confusion, trouble speaking/understanding, vision problems, dizziness, loss of balance, severe headache with no cause",
            "treatment": "CALL 911 IMMEDIATELY, remember FAST (Face drooping, Arm weakness, Speech difficulty, Time to call 911), emergency clot-busting drugs or mechanical thrombectomy (must be within 3-4.5 hours), hospitalization, rehabilitation",
            "additional_info": "Medical emergency. Brain cells die within minutes without oxygen. Quick treatment critical for survival and minimizing disability. Know the signs - FAST test. Major cause of death and long-term disability."
        },
        
        # Infectious Diseases
        {
            "id": "infect_001",
            "condition": "Urinary Tract Infection (UTI)",
            "symptoms": "burning sensation during urination, frequent urgent need to urinate, cloudy/bloody/strong-smelling urine, pelvic pain (women), lower back pain, low fever",
            "treatment": "Antibiotics (usually 3-7 day course: trimethoprim-sulfamethoxazole, nitrofurantoin, or ciprofloxacin), drink plenty of water, urinate frequently, pain relievers, avoid irritants (caffeine, alcohol)",
            "additional_info": "More common in women. If untreated, can spread to kidneys (pyelonephritis). Complete full antibiotic course. Prevention: stay hydrated, urinate after intercourse, wipe front to back, avoid irritating products."
        },
        {
            "id": "infect_002",
            "condition": "Strep Throat",
            "symptoms": "severe sore throat, pain when swallowing, red swollen tonsils with white patches, tiny red spots on roof of mouth, swollen lymph nodes, fever (101-104°F), headache, no cough",
            "treatment": "Antibiotics (penicillin or amoxicillin) for 10 days, complete full course even if feeling better, pain relievers, throat lozenges, warm liquids, rest, avoid contact with others for 24 hours after starting antibiotics",
            "additional_info": "Caused by Group A Streptococcus bacteria. Requires antibiotic treatment to prevent complications (rheumatic fever, kidney inflammation). Very contagious. Rapid strep test for diagnosis."
        },
        {
            "id": "infect_003",
            "condition": "COVID-19",
            "symptoms": "fever/chills, cough, shortness of breath, fatigue, muscle/body aches, headache, loss of taste or smell, sore throat, congestion, nausea, diarrhea, wide range from asymptomatic to severe",
            "treatment": "Mild: rest, fluids, fever reducers, isolate for 5 days. Severe: hospitalization, oxygen therapy, remdesivir or other antivirals, dexamethasone, monoclonal antibodies (if eligible). Get tested if exposed.",
            "additional_info": "Caused by SARS-CoV-2 virus. Can cause long COVID (symptoms lasting >4 weeks). Vaccination available and recommended. High-risk groups: elderly, chronic conditions, immunocompromised. Can spread before symptoms appear."
        },
        
        # Dermatological
        {
            "id": "derm_001",
            "condition": "Contact Dermatitis",
            "symptoms": "red itchy rash, dry cracked skin, bumps/blisters, swelling, burning sensation, skin sensitivity, rash appears where contact occurred (hands, face, arms)",
            "treatment": "Identify and avoid irritant/allergen, cool wet compresses, anti-itch creams (hydrocortisone), antihistamines for itching, moisturizers, severe cases may need oral corticosteroids",
            "additional_info": "Caused by contact with irritants (soaps, detergents, chemicals) or allergens (poison ivy, nickel, fragrances, latex). Usually resolves in 2-4 weeks with proper treatment. Patch testing can identify allergens."
        },
        
        # Musculoskeletal
        {
            "id": "musculo_001",
            "condition": "Muscle Strain",
            "symptoms": "sudden pain during activity, muscle stiffness, swelling, bruising, limited range of motion, muscle spasms, weakness in affected area",
            "treatment": "RICE protocol (Rest, Ice 20 min every 2-3 hours, Compression with elastic bandage, Elevation), NSAIDs for pain, gentle stretching after 48 hours, gradual return to activity, physical therapy if severe",
            "additional_info": "Common in sports and physical activities. Most heal in 2-3 weeks with proper care. Grade 1 (mild): minimal tear. Grade 2 (moderate): more extensive damage. Grade 3 (severe): complete tear, may require surgery."
        },
        {
            "id": "musculo_002",
            "condition": "Fracture (Broken Bone)",
            "symptoms": "severe pain at injury site, obvious deformity, inability to move affected area, swelling, bruising, bone protruding through skin (compound fracture), numbness/tingling, grinding sensation",
            "treatment": "Immobilize the area, DO NOT try to realign bone, apply ice, seek emergency medical care, X-ray for diagnosis, casting/splinting or surgery depending on severity, pain medication, physical therapy",
            "additional_info": "Requires immediate medical attention. Complications can include infection (especially compound fractures), blood vessel/nerve damage, arthritis. Healing time 6-8 weeks for most bones. Osteoporosis increases fracture risk."
        },
        
        # Endocrine
        {
            "id": "endo_001",
            "condition": "Diabetic Ketoacidosis (DKA)",
            "symptoms": "blood sugar >250 mg/dL, extreme thirst, frequent urination, nausea/vomiting, abdominal pain, fruity-smelling breath, confusion, difficulty breathing, weakness",
            "treatment": "CALL 911 or go to emergency room, hospitalization required, IV insulin, IV fluids, electrolyte replacement, continuous monitoring, treat underlying cause (infection, missed insulin)",
            "additional_info": "Life-threatening complication of diabetes (mainly Type 1). Body breaks down fat for energy, producing ketones. Can lead to coma or death if untreated. Triggered by illness, stress, missed insulin doses."
        },
        
        # Mental Health (Physical manifestations)
        {
            "id": "mental_001",
            "condition": "Panic Attack",
            "symptoms": "sudden intense fear, racing heartbeat, sweating, trembling, shortness of breath, chest pain, nausea, dizziness, feeling of losing control, fear of dying, symptoms peak within 10 minutes",
            "treatment": "Deep breathing exercises, grounding techniques (5-4-3-2-1 method), move to quiet space, reassure that symptoms will pass, benzodiazepines for severe cases, cognitive behavioral therapy for prevention, SSRI medications",
            "additional_info": "Not life-threatening but can feel terrifying. Usually lasts 5-20 minutes. Different from heart attack but symptoms can mimic one. Can occur with or without panic disorder. Triggers vary by individual."
        }
    ]
    
    return medical_data


def save_medical_data(filename='medical_data.json'):
    """Save the medical data to a JSON file."""
    data = create_comprehensive_medical_data()
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Created {filename} with {len(data)} medical conditions")
    print("\nConditions included:")
    
    # Group by category
    categories = {}
    for item in data:
        category = item['id'].split('_')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(item['condition'])
    
    for category, conditions in categories.items():
        print(f"\n{category.upper()}:")
        for condition in conditions:
            print(f"  • {condition}")


if __name__ == "__main__":
    save_medical_data()
    print("\n" + "="*70)
    print("Sample medical data file created successfully!")
    print("You can now use this file with the Medical Diagnosis Assistant.")
    print("="*70)