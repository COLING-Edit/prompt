import math
import os
import logging
from sampling import SteepHC
from editor import RobertaEditor
from model_args import get_args
import numpy as np
import warnings
from utils.functions import set_seed
import datetime
from dateutil import tz
import torch
tzone = tz.gettz()
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = get_args()

    set_seed(args.seed)
    editor = RobertaEditor(args)
    editor.to(device)
    sahc = SteepHC(args, editor).to(device)
    of_dir = 'results/' + args.output_dir
    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    if args.direction == '0-1': postfix = '0'
    else: postfix = '1'

    if args.task=='sentiment':
        filename='data/yelp/test.'+postfix
        with open(filename, 'r', encoding='utf8') as f:
            print("we are running on {}".format(filename))
            data = f.readlines()
    else:
        with open('data/gyafc_500/test.'+postfix, 'r', encoding='utf8') as f:
            data = f.readlines()

    batch_size = 1
    num_batches = math.ceil(len(data) / float(batch_size))

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

    output_file ='{}_{}_seed={}_{}_{}_{}.txt'.\
        format(timestamp,args.task,str(args.seed),args.style_mode,str(args.style_weight),args.direction)
    log_txt_path=os.path.join(of_dir, output_file.split('.txt')[0] + '.log')
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='',
                        filename=log_txt_path,
                        filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    word_pairs ={"ca n't": "can not","n't":"not", "wo n't": "will not"}
    logging.info(args)

    with open(of_dir + output_file, 'w', encoding='utf8') as f:
        for i in range(num_batches):
            batch_data = data[batch_size * i:batch_size * (i + 1)]
            ref_olds = []

            #preprocessing
            temp = batch_data[0].strip().lower()
            for k, v in word_pairs.items():
                temp = temp.replace(k, v)
            line = temp
            ref_olds.append(line)
            batch_size = len(ref_olds)  # 1
            ref_oris=ref_olds.copy()
            state_vec, _ = editor.state_vec(ref_olds)

            break_flag = False
            max_score=0
            for step in range(args.max_steps):
                total_score_list = []
                cand_sent_list = []
                seq_len = len(ref_olds[0].split())
                for positions in range(seq_len):
                    positions = [positions]
                    for ops in [0,1,2]:
                        ops = np.array([ops])
                        ref_news = editor.edit(ref_olds, ops, positions, args.max_len)
                        _, index, ref_old_score, ref_new_score, old_style_score,new_style_scores, new_style_labels \
                            = sahc.acceptance_prob(ref_news,ref_olds, ref_oris, state_vec)

                        ref_hat = ref_news[index]
                        new_style_score=new_style_scores[index]
                        new_style_label=new_style_labels[index]

                        total_score_list.append(new_style_score)
                        cand_sent_list.append(ref_hat)

                        if args.early_stop == True:
                            if args.direction == '0-1' and new_style_label == 'positive':
                                print("Early Stopping!")
                                logging.info("Early Stopping!")
                                print("{} steps, {}\ttotal score:{} {}\tstyle_score:{} {}"
                                      .format(step+1,ref_hat, ref_old_score.item(),
                                              ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                                logging.info("{} steps, {}\ttotal score:{} {}\tstyle_score:{} {}"
                                             .format(step+1,ref_hat, ref_old_score.item(),
                                                     ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                                break_flag=True
                                break

                            elif args.direction == '1-0' and new_style_label == 'negative':
                                print("Early Stopping!")
                                logging.info("Early Stopping!")
                                print("{} steps, {}\ttotal score:{} {}\t style_score:{} {}"
                                      .format(step+1,ref_hat, ref_old_score.item(),
                                              ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                                logging.info("{} steps, {}\ttotal score:{} {}\t style_score:{} {}"
                                             .format(step+1,ref_hat, ref_old_score.item(),
                                                     ref_new_score.item(), old_style_score.item(), new_style_score.item()))
                                break_flag = True
                                break
                    if break_flag:
                        break

                select_index = torch.argmax(torch.tensor(total_score_list).cuda())
                select_sent = cand_sent_list[select_index]
                if total_score_list[select_index]>max_score:
                    print("hill climbing!")
                    logging.info("hill climbing!")
                    ref_olds = [select_sent]
                    max_score=total_score_list[select_index].item()
                else:
                    print("don't climb, stop!")
                    logging.info("don't climb, stop!")
                    cand_sent_list.append(select_sent)
                    break_flag=True

                if break_flag:
                    break

            if args.early_stop and break_flag:
                select_sent = cand_sent_list[-1]
            else:
                select_sent=ref_olds[0]

            logging.info('climb {} steps, the selected sentence is: {}'.format(step+1,select_sent))
            print('climb {} steps, the selected sentence is: {}'.format(step+1,select_sent))
            logging.info('\n')
            f.write(select_sent + '\n')
            f.flush()

if __name__=="__main__":
    main()

