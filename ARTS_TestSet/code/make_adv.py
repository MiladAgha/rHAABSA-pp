import os
import json
from strategies_new import revTgt,revNon,addDiff

def make_adv(dataset, split):

    data_folder = 'data/generated/{}/'.format(dataset)

    input_file = os.path.join(data_folder, '{}_sent_towe.json' .format(split))
    infile = os.path.join(data_folder, 'train_sent_towe.json')
    infile2 = os.path.join(data_folder, '{}_sent.json' .format(split))
    output_file = os.path.join(data_folder, '{}_adv.json' .format(split))
    res_file = os.path.join(data_folder, '{}.json' .format(split))
    

    with open(res_file, 'r', encoding='utf-8') as fr:
        res = json.load(fr)
    
    res = revTgt(res, data_folder, input_file)
    res = revNon(res, data_folder, input_file)
    res = addDiff(res, data_folder, infile, infile2)
    
    with open(output_file, 'w', encoding='utf-8') as fw:
        json.dump(res, fw, sort_keys=True, indent=4)

make_adv("14lap", "train")
make_adv("14lap", "test")

make_adv("14res", "train")
make_adv("14res", "test")

make_adv("15res", "train")
make_adv("15res", "test")

make_adv("16res", "train")
make_adv("16res", "test")