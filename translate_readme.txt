Command for using translate.py BPE-case:
python translate.py -k {beam_width} -p {number_of_processors} -n -bpe {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}
Command for using translate.py Char-case:
python translate.py -k {beam_width} -p {number_of_processors} -n -dec_c {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}
Command for using translate_both.py BPE-case:
python translate_both.py -k {beam_width} -p {number_of_processors} -n -bpe {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}
Command for using translate_both.py Char-case:
python translate_both.py -k {beam_width} -p {number_of_processors} -n -dec_c {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}
Command for using translate_attc.py Char-case:
python translate_attc.py -k {beam_width} -p {number_of_processors} -n -dec_c {path/model.npz} {path/source_dict} {path/target_dict} {path/valid.txt or test.txt} {save_path/save_file_name}

Command for using `multi-bleu.perl':
perl multi-bleu.perl {reference.txt} < {translated.txt}
