import os
import sys
import argparse
import torch
from collections import deque
sys.path.append(os.getcwd())

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

def display_data1(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    model_list = interactive.load_model_file(opt['model_file'])
    
    opt1, state_dict = model_list
    data_loader, text_encoder = interactive.load_data("atomic", opt1)
    
    dialog_res = []
    # Show some example dialogs.
    print("will print "+ str(opt['num_examples'])+' dialog')
    dialog_id = 0 
    idx=0
    single_dialog = []
    while True:
        world.parley()

        # NOTE: If you want to look at the data from here rather than calling
        # world.display() you could access world.acts[0] directly
        # print(world.display() + '\n~~')
        utterance =  world.acts[0]
        for i in range(2):
            event_dict = {}
            if i ==0:
                sentence = utterance['text'].split('\n')[-1]
                if len(utterance['text'].split('\n'))>1:
                    dialog_id +=1
                    if len(single_dialog)>2:
                        dialog_res.append(single_dialog)
                    single_dialog = []
                    idx = 0 
                    # TODO: may update later to handle persona
                event_dict['utterance'] = sentence# utterance['text'].replace('\n',' || ')
                    
            else:
                if(len(utterance['eval_labels'])>1):
                    print('=======================')
                    print(utterance)

                event_dict['utterance'] = utterance['eval_labels'][0]                    
                sentence = event_dict['utterance']
            assert 'utterance' in event_dict
            event_list = ['xEffect', 'xIntent', 'xNeed', 'xWant']
            eres = extract_event(opt,sentence,opt1,state_dict,data_loader, text_encoder)
            
            for etype in event_list:
                beam_res = eres[etype]['beams']
                # take the first event
                for res in beam_res:
                    if res  == "none":
                        continue
                    event_dict[etype] =res
                    break
            
            event_dict["utt_id"] = idx
            event_dict["id"] = i
            
            single_dialog.append(event_dict)
        
        idx += 1 
        if dialog_id % 100==0:
            print("print "+ str(dialog_id) +' dialog')
        if world.epoch_done():
            print('EPOCH DONE')
            break
    return dialog_res
        
    try:
        # print dataset size if available
        print(
            '[ loaded {} episodes with a total of {} examples ]'.format(
                world.num_episodes(), world.num_examples()
            )
        )
    except Exception:
        pass

def extract_event(args,input_event,opt, state_dict,data_loader, text_encoder):


    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args['device'] != "cpu":
        cfg.device = int(args['device'])
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"


    sampling_algorithm = args['sampling_algorithm']


    if input_event == "help":
        interactive.print_help(opt.dataset)

    category = "all" #input("Give an effect type (type \"help\" for an explanation): ")

    if category == "help":
        interactive.print_category_help(opt.dataset)

    sampling_algorithm = "beam-3" #input("Give a sampling algorithm (type \"help\" for an explanation): ")

    if sampling_algorithm == "help":
        interactive.print_sampling_help()
        

    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    if category not in data_loader.categories:
        category = "all"
    outputs = interactive.get_atomic_sequence(
        input_event, model, sampler, data_loader, text_encoder, category)
    return outputs




    
    

if __name__ == '__main__':
    random.seed(42)

    # Get command line arguments
    
    parser = setup_args()
    set_name = 'valid'
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--model_file", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")
    opt = parser.parse_args()
    #parser = argparse.ArgumentParser(parser)
    #opt = parser.parse_args()
    truncate_length = 4 
    
    dialog_list = display_data1(opt)

    last_event = []
    with open('src-'+set_name+'.txt','w') as f_src:
        with open('tgt-'+set_name+'.txt','w') as f_tgt:
            
            for dialog in dialog_list:
                his = deque([])
                for utt in dialog:
                    if len(his)== 0:
                        his.append(utt["utterance"])  
                        continue
                    # add current events
                    event_list = ['xEffect', 'xIntent', 'xNeed', 'xWant']
                    cur_events = [ utt[e] for e in event_list] 
                    if len(his) > truncate_length:
                        his.popleft()
                    src_sent = " ".join(his)
                    src_sent += " "
                    src_sent += " ".join(cur_events)
                    f_src.write(src_sent+'\n')
                    f_tgt.write(utt["utterance"]+'\n')
                    his.append(utt["utterance"])
    
                    
                    
                

    # keys = res[0].keys()
    # with open('result.csv', 'w') as output_file:
    #     dict_writer = csv.DictWriter(output_file, keys) # wirte the header
    #     dict_writer.writeheader()
    #     dict_writer.writerows(res)

    


    
    
