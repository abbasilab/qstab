import pandas as pd



def options(row):
    
    row_names = ["A", "B", "C", "D"]
    return {rn: row[rn] for rn in row_names}

def yield_answer(row):
    
    return row["options"][row["answer_idx"]]


class Formatter:
    
    def __init__(self, data_frame, column_dict=None, answer_dict=None):
        
        self.df = data_frame
        if column_dict is None:
            column_dict = {"sent1":"reference",
                           "sent2":"question",
                           "ending0":"A",
                           "ending1":"B",
                           "ending2":"C",
                           "ending3":"D"}
            
        if answer_dict is None:
            answer_dict = {0:"A", 1:"B", 2:"C", 3:"D"}
        
        self.df = self.df.rename(columns=column_dict);
        self.df["answer_idx"] = self.df["label"].map(answer_dict)
        
        self.df["options"] = self.df.apply(options, axis=1)
        self.df["answer"] = self.df.apply(yield_answer, axis=1)
        
    def export(self, faddress=None, subset=None, orient="index", format="json", **kwargs):
        
        if subset is not None:
            export_df = self.df.iloc[:subset, :]
        else:
            export_df = self.df
        
        if format == "json":
            export_df.to_json(faddress, orient=orient, **kwargs)
        elif format == "dict":
            export_df.to_dict(orient=orient, **kwargs)