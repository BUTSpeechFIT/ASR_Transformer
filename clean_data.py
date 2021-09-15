"""
Author: Marek Sarvas
Project: NEUREM3

Example: python clean_data.py DATA_FILE [JSON_FILE]
"""
import argparse
import json


def clean_header(data):
    data = data.split('\n')
    tmp_dict = {"model_weights": data[3].split(':')[1][1:],
                "hyper_params": data[6],
                "true_label": data[8].split('True_label')[1].strip()
    }  

    return data[0].strip(), tmp_dict

def clean_first_final_output(data, tmp_dict):
    nbest=data.split('\n', 2)[1].split('=')
    output=data.split('\n', 2)[2]
    
    tmp_dict['nbest_output'] = {
        "id" : nbest[1],
        "output_text_seq" : nbest[2],
        "true_text_seq" : nbest[3],
        "CER" : nbest[4]
        }
    tmp_dict['outputs'] = []

    clean_final_output(output, tmp_dict)

    return tmp_dict

def clean_final_output(data, tmp_dict):
    data = data.replace('\n', '').split('=')
    tmp_dict['outputs'].append({
        "id" : data[2].strip(),
        "text_seq" : data[3],
        "LLR" : data[4].strip(),
        "Beam_norm_llr" : data[5].strip(),
        "CER" : data[7].strip()
    })


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True, default=None, type=str, help='text file with saved logs')
    parser.add_argument('--json_file', default='clean.json', type=str, help='text file where json will be saved')
    args = parser.parse_args()

    with open(args.data_file, "r") as f:
        log = f.read().split('Decoding utt num:')
        log = log[1:]  # remove empty string

    final_dict = {}
    with open(args.json_file, "w", encoding='utf8') as f:
        
        for i, utt in enumerate(log):
            print('Cleaning utt: {}'.format(i))
            utt = utt.split('******pruning_ngarms*******')

            utt_id, tmp_dict = clean_header(utt[0])
            
            tmp_dict = clean_first_final_output(utt[1], tmp_dict)
           
            for i in range(2, len(utt)):
                clean_final_output(utt[i], tmp_dict)
            
            utt_id = 'utt_'+utt_id
            final_dict[utt_id] = tmp_dict
      
        json.dump(final_dict, f, indent=4, ensure_ascii=False)
           
       
