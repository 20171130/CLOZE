This repository provides a baseline model for the cloze task:

    pretrained BERT large uncased is used, and finetuned as a masked LM
    
    Articles longer than 512 (limit of our BERT implementation) are truncated, while the blanks left are arbitrarily filled
    
    Multiple token options are truncated to their first tokens
    
    No weight decay is used
    
tested with 

    python = 3.6.8
    
    pytorch = 1.3.1
    
    transformers = 4.0.0
