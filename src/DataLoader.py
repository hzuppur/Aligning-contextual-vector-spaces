def read_sents(paths, rows=1000, max_len=300):
    sents = []
    
    for path in paths:
        with open(path, encoding="utf8") as f:
            for _ in range(rows // len(paths)):
                sents.append(f.readline().strip()[:max_len])
    return sents


def read_parallel_corpus(lang_1_path, lang_2_path, rows=1000):
    lang_1_sents = []
    lang_2_sents = []
    
    with open(lang_1_path, encoding="utf8") as f:
        for _ in range(rows):
            lang_1_sents.append(f.readline().strip())
            
    with open(lang_2_path, encoding="utf8") as f:
        for _ in range(rows):
            lang_2_sents.append(f.readline().strip())
    
    return lang_1_sents, lang_2_sents
