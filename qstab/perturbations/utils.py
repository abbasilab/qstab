import numpy as np
from scipy.spatial.distance import cosine
from thefuzz import fuzz
from unidecode import unidecode
import torch


def iscap(text):
    ''' Check if text string is capitalized.
    '''

    return text[0].isupper()


def find_neighbor(a, others, distfunc=cosine, keep='nearest'):
    
    distances = []
    for ve in others:
        distances.append(distfunc(a, ve))

    distances = np.array(distances)
    
    if keep == 'nearest':
        return np.argmin(distances)
    elif keep == 'nearest2':
        return np.argsort(distances)[:2]
    elif keep == 'farthest':
        return np.argmax(distances)
    elif keep == 'median':
        return np.argwhere(distances == np.median(distances)).item()
    

def distance_prob(a, others, distfunc=cosine, n=1):
    ''' Calculate inverse distance probabilities.
    '''
    
    distances = np.array([distfunc(a, b)**n for b in others])
    distsum = np.sum(distances)
    probs = distances/distsum

    return probs
    

def span2vec(span, model, tokenizer):
    
    span_info = tokenizer(span, return_tensors='pt')
    span_ids = span_info['input_ids']
    span_mask = span_info['attention_mask']
    
    with torch.no_grad():
        vec = model(span_ids).last_hidden_state[0][1]
    
    return vec.numpy()


def collect_entities(ent_tags, ent_types):
    ''' Collect entities that belong to specific types.
    '''
    
    entities = []
    for an_id, anno in ent_tags.items():
        choice_entities = []
        for ent, lab in zip(anno['text'], anno['labels']):
            if lab in ent_types:
                choice_entities.append(ent)
        entities.extend(choice_entities)
    
    return entities


def space_strip(string, space='both'):
    
    if space == 'both':
        return string.strip()
    elif space == 'left':
        return string.lstrip()
    elif space == 'right':
        return string.rstrip()


def udc_compare(S1, S2, case_sensitive=False, space_stripping='both',
                dehyph=True, matching='exact', threshold=0.9):
    
    _S1 = space_strip(S1, space=space_stripping)
    _S2 = space_strip(S2, space=space_stripping)
    
    if dehyph:
        _S1 = _S1.replace('-', ' ')
        _S2 = _S2.replace('-', ' ')
    
    if case_sensitive:
        s1, s2 = _S1, _S2
    else:
        s1, s2 = _S1.casefold(), _S2.casefold()
    s1, s2 = unidecode(s1), unidecode(s2)
    
    if matching == 'exact':
        score = s1 == s2
        if score:
            return True
        else:
            return False
    elif matching == 'ratio':
        score = fuzz.ratio(s1, s2)
        if score >= threshold:
            return True
        else:
            return False