import numpy as np
from copy import copy



def entity_perturb(option, ent_tags, perturb_types, replacer):
    ''' Entity perturbation.
    
    option: Option for the question.
    ent_tags: Annotations for the question options.
    perturb_types: Compatible type unique identifiers (TUIs) for perturbation.
    replacer: Entry to replace the existing part of text.
    '''
    
    tags = ent_tags['labels']
    
    # Select an entity based on the type unique identifiers provided
    # Check if some of the entity tags are compatible with the given perturbation types
    type_flags = []
    for tag in tags:
        if tag in perturb_types:
            type_flags.append(True)
        else:
            type_flags.append(False)
    type_flags = np.array(type_flags)
    n_passables = type_flags.sum()
    
    # Return unperturbed options if no entity type matches
    if n_passables == 0:
        return option
    # Return singly perturbed options when only one entity type matches
    elif n_passables == 1:
        pass_id = np.argwhere(type_flags == True).item()
    # Return perturbed options after randomly select one type-matched entity to perturb
    elif n_passables > 1:
        passes = np.argwhere(type_flags == True).ravel()
        pass_id = np.random.choice(passes, 1).item()
        
    opt_text = copy(option)
    a, b = ent_tags['boundaries'][pass_id]
    span_unprtrb = opt_text[a:b]
    
    cap_flag = span_unprtrb[0].isupper()
    if cap_flag:
        replacer = replacer.capitalize()
    else:
        replacer = replacer.lower()

    option_prtrb = opt_text[:a] + replacer + opt_text[b:]
    
    # print('{} | {}'.format(option_prtrb, opt_text))
    return option_prtrb, span_unprtrb