


class Entity:
    ''' An UMLS named entity class.
    '''
    
    def __init__(self, ent, linker=None):
        
        self.entity = ent
        self.text = ent.text
        self.vector = ent.vector
        self.vector_norm = ent.vector_norm
        
        # Linker is not carried over to keep the size down
        if linker is None:
            self.tui = None
            self.type = None
        else:
            self.tui = self.get_tui(rank=1, linker=linker)[0]
            self.type = self.get_type(rank=1, linker=linker)
        
    @property
    def cui(self):
        ''' The top-1 concept unique identifier (CUI) for the entity.
        '''
        return self.get_cui(rank=1, prob=False)
    
    @property
    def cuis(self):
        ''' All CUIs for the entity.
        '''
        return self.get_all_cuis(prob=False)
    
    def link(self, linker):
        ''' Link entity to a knowledge base.
        '''
        
        self.tui = self.get_tui(rank=1, linker=linker)[0]
        self.type = self.get_type(rank=1, linker=linker)
        
    def get_cui(self, rank=1, prob=False):
        ''' Retrieve the top-1 CUI for the entity.
        '''
        
        cui = self.entity._.kb_ents[rank-1]
        if prob == False:
            return cui[0]
        else:
            return cui
    
    def get_all_cuis(self, prob=False):
        ''' Retrieve all CUIs for the entity.
        '''
        
        cuis = dict(self.entity._.kb_ents)
        if prob == False:
            return list(cuis.keys())
        else:
            return cuis
    
    def get_tui(self, linker, rank=1):
        ''' Retrieve the top-1 type unique identifier (TUI) for the entity.
        '''
        
        cui = self.get_cui(rank=rank, prob=False)
        tui = linker.kb.cui_to_entity[cui].types
        
        return tui
    
    def get_all_tuis(self, linker):
        ''' Retrieve all TUIs for the entity.
        '''
        
        cuis = self.get_all_cuis(prob=False)
        tuis = [linker.kb.cui_to_entity[cui].types[0] for cui in cuis]
        
        return tuis
    
    def get_type(self, linker, rank=1):
        ''' Retrieve the top-1 semantic group for the entity.
        '''
            
        tui = self.get_tui(rank=rank, linker=linker)[0]
        ent_type = linker.kb.semantic_type_tree.get_canonical_name(tui)
        
        return ent_type
        
    def get_all_types(self, linker):
        ''' Retrieve all semantic groups for the entity.
        '''
        
        tuis = self.get_all_tuis(linker=linker)
        ent_types = [linker.kb.semantic_type_tree.get_canonical_name(tui) for tui in tuis]
        
        return ent_types
    

    class Annotation:
        ''' Paired token texts and labels. The labels can be numbers indicating types or type unique identifiers (TUIs) if the associated entities follow the UMLS convention.
        '''
    
        def __init__(self, texts, labels):
            
            self.texts = texts
            self.labels = labels
            self.len = len(labels)
            
            if len(self.text) != self.len:
                raise ValueError("The number of texts and labels should be the same!")
            
        def to_dict(self, keys="type"):
            
            if keys == "text":
                return dict(zip(self.texts, self.labels))
            elif keys == "type":
                return {"text":self.texts, "labels":self.labels}
        
        def to_tuples(self, keys="type"):
            
            if keys == "text":
                return [(t, l) for (t, l) in zip(self.texts, self.labels)]
            elif keys == "type":
                return [tuple(self.texts), tuple(self.labels)]