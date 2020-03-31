import os
import sys
import argparse
import torch
from tqdm import tqdm
sys.path.append(os.getcwd())
import pickle
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic example which iterates through the tasks specified and prints them out. Used for
verification of data loading and iteration.

For more documentation, see parlai.scripts.display_data.
"""


from parlai.scripts.display_data import display_data, setup_args
from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

import random
import csv



def extract_event(args,input_event,model,data_loader, text_encoder,opt):
    if args['device'] != "cpu":
        cfg.device = int(args['device'])
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"
    category = "all" #input("Give an effect type (type \"help\" for an explanation): ")
    sampling_algorithm = "beam-3" #input("Give a sampling algorithm (type \"help\" for an explanation): ")
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    if category not in data_loader.categories:
        category = "all"

    outputs = interactive.get_atomic_sequence(
        input_event, model, sampler, data_loader, text_encoder, category)

    return outputs


def gen_event_dict(utt,eres,idx,dialog_id):
    event_dict = {}
    event_dict['utterance']= utt
    event_list = ['xEffect', 'xIntent','xReact', 'xNeed', 'xWant']
    for etype in event_list:
        event_dict[etype] = "none"

        beam_res = eres[etype]['beams']
        
        for res1 in beam_res:
            if res1  == "none":
                continue
            event_dict[etype] = res1
            break
        
    event_dict["dialog_id"] = dialog_id
    event_dict["idx"] = idx
 
    return event_dict


if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--model_file", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")
    args = parser.parse_args()
    #parser = argparse.ArgumentParser(parser)
    split = 'train'
    file_path = split + '_both_original.txt'
    dir_path = '/home/xuewei/comet-commonsense/data/Persona-Chat/personachat/'
    out_path = '/home/xuewei/comet-commonsense/data/Persona-Chat/event/'
    # load data & data loader
    model_list = interactive.load_model_file(args['model_file'])
    opt , state_dict = model_list

    data_loader, text_encoder = interactive.load_data("atomic", opt)
    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
    
    
    idxs = []
    sents = []
    
    event_selfs =[]
    event_others = []
    dialog_id = -1
    index = 0
    with open(dir_path + file_path,'r') as f:
        lines = f.readlines()
        persona=[]
        event_list = []
        for l in tqdm(lines[:100]):
            l = l.strip()
            res = l.split('\t')[:2]
            if len(res)==1:
                persona = res[0].split(" ", 1)[1]
                dict_persona= {}
                dict_persona['utterance'] = persona
                event_type = ['xEffect', 'xIntent', 'xReact','xNeed', 'xWant']
                for e in event_type:
                    dict_persona[e] = None
                if res[0].split(" ", 1)[0] == '1':
                    dialog_id += 1
                    index = -1
                dict_persona["dialog_id"] = dialog_id
                dict_persona["idx"] = index
                event_list.append(dict_persona)
           
            else:
                index +=1
                self1,other_sent = res
                idx, self_sent = self1.split(" ", 1)
                idxs.append(idx)
                event_self = extract_event(args,self_sent,model,data_loader, text_encoder,opt)
                event_other = extract_event(args,other_sent,model,data_loader, text_encoder,opt)
                event_selfs.append(event_self)
                event_others.append(event_other)

                dict1 = gen_event_dict(self_sent,event_self,index,dialog_id)
                dict2 = gen_event_dict(other_sent,event_other,index,dialog_id)
                event_list.append(dict1)
                event_list.append(dict2)
        print(len(event_list))
 
             
            
          # f = open(out_path+split+"_dict.pkl","wb")
#         pickle.dump(event_list,f)
    
        # f.close()
        # keys = dict1.keys()
        # with open(out_path+split+'.csv', 'w') as output_file:
        #     dict_writer = csv.DictWriter(output_file, keys)
        #     dict_writer.writeheader()
        #     dict_writer.writerows(event_list)
  
                

            
            
           
    
    
   