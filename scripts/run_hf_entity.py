import qstab.data as qdata
from qstab.perturbations import utils as qutils, sampling as qsamp
import argparse
import pandas as pd
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM)
from tqdm import tqdm
from copy import copy
import numpy as np
import torch
import json
from pprint import pprint
from scipy.stats import energy_distance as ed


# Load model and tokenizer
modstr = "flan-t5-xl"
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl",
                                              device_map="auto",
                                              torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")


tui_drugs = ['T116', 'T195', 'T123', 'T122', 'T103', 'T120', 'T104',
        'T200', 'T196', 'T126', 'T131', 'T125', 'T129', 'T130',
        'T197', 'T114', 'T109', 'T121', 'T192', 'T127']
tui_diseases = ['T020', 'T190', 'T049', 'T019', 'T047', 'T050', 'T033',
        'T037', 'T048', 'T191', 'T046', 'T184']


def span2vec(span, model, tokenizer):
    
    span_info = tokenizer(span, return_tensors='pt')
    span_ids = span_info['input_ids']
    span_mask = span_info['attention_mask']
    
    with torch.no_grad():
        vec = model.encoder(input_ids=span_ids,
                           attention_mask=span_mask).last_hidden_state[0][1]
    
    return vec.numpy()


# Command line interactions
parser = argparse.ArgumentParser(description="Input arguments")
parser.add_argument("-ptb", "--perturb", metavar="perturb", nargs="?", type=bool,
                    help="Option to perturb the text.")
parser.add_argument("-ptb_samp", "--perturb_sampling", metavar="perturb_sampling", nargs="?", type=str,
                    help="Type of sampling for the perturbation.")
parser.add_argument("-grp", "--group", metavar="group", nargs="?", type=str,
                    help="Semantic group for the perturbating words")
parser.add_argument("-ndist", "--numdistance", metavar="numdistance", nargs="?", type=int,
                    help="Power of the distance in the sampling probability weights.")
parser.add_argument("-nq", "--numq", metavar="numq", nargs="?", type=int,
                    help="Number of questions.")
parser.set_defaults(perturb=False, perturb_sampling="None", group="drugs", numdistance=20, numq=10)
cli_args = parser.parse_args()

PERTURB = cli_args.perturb
PTB_SAMP = cli_args.perturb_sampling
GROUP = cli_args.group
NDIST = cli_args.numdistance
if PTB_SAMP == "random":
    NDIST = 0
NQ = cli_args.numq
print(PERTURB, PTB_SAMP, GROUP, NDIST, NQ)

if PERTURB: # With perturbation
    anscoll = qdata.question.AnswerCollector(colnames=['mod_ans', 'flag',
                                                       'prtrb_id', 'prtrb_item',
                                                       'replace_item'])
    if PTB_SAMP == "distance":
        print('Yes!')
        from transformers import BertTokenizer, BertModel
        # tknzer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # mod = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # vecs = np.load(r'./external/word_vecs_bioclinicalBERT.npz')['vecs']

        tknzer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
        mod = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")
        if GROUP == 'drugs':
            vecs = np.load(r'./external/drug_word_vecs_umlsBERT.npz')['vecs']
        elif GROUP == 'diseases':
            vecs = np.load(r'./external/disease_word_vecs_umlsBERT.npz')['vecs']
else: # No perturbation
    anscoll = qdata.question.AnswerCollector(colnames=['mod_ans', 'flag'])


# Load perturbation set
if GROUP == 'drugs':
    entities = pd.read_csv(r'./external/FDA_Approved.csv', names=['id', 'name'])
    entity_names = np.array(list(entities['name']))
    type_selector = 'Drugs?'
    choice_type = 'DrugChoices'
    tui_types = tui_drugs

elif GROUP == 'diseases':
    entities = pd.read_csv(r'./external/CTD_unique_disease_names.csv')
    entity_names = np.array(list(entities['name']))
    type_selector = 'Diseases?'
    choice_type = 'DiseaseChoices'
    tui_types = tui_diseases

# Load dataset
# qa = pd.read_parquet(r'./external/train_drugs.parquet')
qa = pd.read_parquet(r'./external/medqa_usmle_train_qtyped.parquet')
qad = qdata.formatter.Formatter(qa).df
qadf = qad[qad[type_selector]==True] # Select the disease or drug-related questions
nqs = len(qadf)
# Set to boundary value if larger than it
if NQ > nqs:
    NQ = nqs
print("The total number of questions is {}.".format(NQ))

# Load entity tags
with open(r'./external/medqa_usmle_train_choices_annotation.json', 'r') as f:
    annotations = json.load(f)
annotations = np.array(annotations)[qad[type_selector]==True].tolist()

# Query the answer for each question
for i in tqdm(range(0, NQ)):

    qdict = qadf.iloc[i,:].to_dict()
    qdict["prompt_prefix"] = "Answer the question without explanation.\n[Context]: "
#     qdict["prompt_prefix"] = "[Context]: "
    qdict["prompt_suffix"] = "[Answer]: "
    qobj = qdata.question.Question.from_dict(qdict)
    qsent = qobj.question.split('. ')[-1]
    qobj.question = qobj.question[:-len(qsent)]
    qobj.question = qobj.question + '\n[Question]: ' + qsent
    # qfull = copy(qobj.full_question)
    
    # Pick the choice to perturb
    qoptions = copy(qobj.options)
    correct_word = qoptions.pop(qobj.answer_idx) # Drop the correct answer
    # current_options = np.array(list(qoptions.keys()))
    perturbables = np.array(qdict[choice_type]) # Options that may be perturbed
    # print('\ni={}, perturbables are {}'.format(i, perturbables))
    current_key = qobj.answer.lower() # The text of the answer
    matched_names = np.array([qutils.udc_compare(ent_name, current_key, matching='exact') for ent_name in entity_names])
    # selector = np.argwhere(entity_names != current_key)
    selector = np.argwhere(matched_names == False)
    distractors = np.squeeze(entity_names[selector])
    
    current_distractors = {}
    for opt_id, opt in qoptions.items():
        if (opt != current_key) and (opt_id in perturbables):
            # print('opt_id is {}'.format(opt_id))
            current_distractors[opt_id] = opt
    # current_distractors = np.array([opt for opt in qoptions.values() if opt != current_key and opt in perturbables])

    annotation = annotations[i]
    current_annotation = {k:annotation[k] for k in current_distractors.keys()}

    # print(len(current_distractors))    
    answ = qobj.query(model, tokenizer, model_kwargs={"max_new_tokens":128,
                                                      "do_sample":False})
    flag = answ == qobj.answer_idx # Check if answer is true
    
    if not PERTURB:
        # Store entry and model response
        anscoll.add_entry(mod_ans=answ, flag=flag)
    
    else:
        # Perturb the answer when model answers correctly
        if flag:

            # Generate perturbation proposals
            if PTB_SAMP == "random":
            
                choice_id = np.random.choice(perturbables, 1).item() # A, B, C, or D
                replacer = np.random.choice(distractors, 1).item()

            elif PTB_SAMP == "distance":

                distractor_entities = qutils.collect_entities(ent_tags=current_annotation, ent_types=tui_types)
                
                distractor_vecs = [qutils.span2vec(opt, mod, tknzer) for opt in distractor_entities]
                key_vec = qutils.span2vec(current_key, mod, tknzer) #  word vector for the text in the correct answer
                # Select the text to replace
                repl_distractor = distractor_entities[qutils.find_neighbor(key_vec, distractor_vecs,
                                                                    keep='nearest')]
                # Find the distractor containing the text to replace
                choice_id = [k for k, v in current_distractors.items() if repl_distractor in v][0] # A, B, C, or D

                # Sample the farthest point in the embedding space from the relevant text span in the distractor
                # replacement_id = qutils.find_neighbor(key_vec, vecs, keep='farthest')
                # replacer = drug_names[replacement_id]
                probs = qutils.distance_prob(key_vec, np.squeeze(vecs[selector, :]), n=NDIST)
                replacer = np.random.choice(distractors, size=1, p=probs)[0]
                # print(choice_id, replacer)
            
            else:
                raise NotImplementedError("Perturbation method not implemented.")
            
            # print('\nchoice_id is {}.\n'.format(choice_id))
            opt_text = copy(qobj.options[choice_id])
            # print('The option text is "{}".\n'.format(opt_text))
            entity_tags = annotation[choice_id]
            output = qsamp.entity_perturb(option=opt_text, ent_tags=entity_tags,
                                           perturb_types=tui_types, replacer=replacer)
            opt_text_modified, prtrb_ent = output
            qobj.options[choice_id] = opt_text_modified
            
            answ = qobj.query(model, tokenizer, model_kwargs={"max_new_tokens":128,
                                                              "do_sample":False})
            flag = answ == qobj.answer_idx # Check if answer is true
            
            # Store entry and model response
            anscoll.add_entry(mod_ans=answ, flag=flag,
                        prtrb_id=choice_id, prtrb_item=qoptions[choice_id], prtrb_ent=prtrb_ent,
                        modified_text=opt_text_modified, replace_item=replacer)
        else:
            anscoll.add_entry(mod_ans=answ, flag=flag)

# Export the attack data
anscoll.add_collection(collection=qadf.iloc[:NQ,:]['answer_idx'].values, name='true_ans')
anscoll.add_collection(collection=qadf.iloc[:NQ,:]['answer'].values,
                       name='true_ans_text')


print(anscoll.answers['flag'].sum()/NQ)
if not PERTURB:
    anscoll.to_excel("hf_{}_{}.xlsx".format(modstr, NQ))
else:
    anscoll.to_excel("hf_{}_{}_perturb_{}_inv{}_{}.xlsx".format(modstr, PTB_SAMP, GROUP, NDIST, NQ))