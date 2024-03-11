import json

def extract_ont(ix, year):

        input_file = 'data/ARTSData/{}test.json' .format(year)
        output_file = 'data/ARTSData/ont{}test.json' .format(year)
        output_keys = 'data/ARTSData/ont{}.json' .format(year)

        with open(input_file, 'r', encoding='utf-8') as fr:
                test = json.load(fr)

        res = {}

        for i in range(int(len(ix)/3)):
                res[list(test.keys())[int(ix[i*3]/3)]] = test[list(test.keys())[int(ix[i*3]/3)]]

        keys = list(test.keys())

        with open(output_file, 'w', encoding='utf-8') as fw:
                json.dump(res, fw, sort_keys=False, indent=4)

        with open(output_keys, 'w', encoding='utf-8') as fw:
                json.dump(keys, fw, sort_keys=False, indent=4)

def extract_ontARTS(year):

        input_keys = 'data/ARTSData/ont{}.json' .format(year)
        input_file = 'data/ARTSData/ARTS{}test.json' .format(year)
        output_file = 'data/ARTSData/ontARTS{}test.json' .format(year)

        with open(input_keys, 'r', encoding='utf-8') as fr:
                keys = json.load(fr)
        with open(input_file, 'r', encoding='utf-8') as fr:
                ARTStest = json.load(fr)
        
        res = {k: v for k, v in ARTStest.items() if any(s in k for s in keys)}

        with open(output_file, 'w', encoding='utf-8') as fw:
                json.dump(res, fw, sort_keys=False, indent=4)

extract_ontARTS(2015)
extract_ontARTS(2016)