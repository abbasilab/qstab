import json
from qstab import data, perturbations
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# model setup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
model.to(torch.device(1))

fpath = '/home/ajl/work/d1/medqa/data_clean/questions/US/test.jsonl'
drug_names = pd.read_csv('../external/FDA_Approved.csv', header=None)
drug_names = drug_names[1].values
drug_names = list(map(lambda el: el.capitalize(), drug_names))

questions = []
with open(fpath, 'r') as f:
    for itm in f:
        questions.append(data.Question.from_dict(json.loads(itm)))

answers = []
perturbation = perturbations.RandomDrugPerturbation(drug_list=drug_names)
for question in tqdm.tqdm(questions):
    
    if perturbation:
        perturbation.apply(question)
    answer = question.query(model, tokenizer)
    answers.append(answer)

torch.save('./answers-asstr_medqa-test_randomdrugperturb.pth', answers)
