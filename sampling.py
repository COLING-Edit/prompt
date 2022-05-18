import torch
import numpy as np
import math
import torch.nn as nn
import sys
sys.path.append("")
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel,AutoTokenizer,AutoModelForCausalLM
from utils.functions import predict_next_word,pipe,pytorch_cos_sim,softmax
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLEU_WEIGHTS_MEAN = [
    [1.0],
    [0.5, 0.5],
    [1/3, 1/3, 1/3],
    [0.25, 0.25, 0.25, 0.25],
]
from nltk.translate.bleu_score import corpus_bleu
from utils.constant import prefix, postfix

class SteepHC(nn.Module):
    def __init__(self, opt,editor):
        super(SteepHC,self).__init__()
        self.opt=opt
        self.editor = editor
        self.t_init = self.opt.t_init
        self.C = self.opt.C
        self.fluency_weight = opt.fluency_weight # 3
        self.keyword_weight = opt.keyword_weight # 1
        self.sent_weight = opt.sent_weight
        self.style_weight=opt.style_weight
        self.bleu_weight=opt.bleu_weight
        self.stride=1024

        if self.opt.style_mode=='plm':
            if 'gpt-j-hf' in self.opt.class_name:
                self.plm=GPTNeoForCausalLM.from_pretrained(self.opt.class_name)
                self.plm.half()
            else:
                self.plm =AutoModelForCausalLM.from_pretrained(self.opt.class_name,low_cpu_mem_usage=True)
            self.plm.eval()
            self.plm.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = self.opt.max_len
        self.model=GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.ppl_max_len=self.model.config.n_positions

    def pipeline_classifier(self,text):
        inputs = self.sty_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self.sty_model(**inputs).logits
        softmax_logits=softmax(logits)
        outputs={}
        predicted_class_id = softmax_logits.argmax().item()
        outputs['label']=self.sty_model.config.id2label[predicted_class_id]
        outputs['score']=softmax_logits.squeeze()[predicted_class_id]

        return [outputs]

    def style_scorer(self,ref_news):
        prob_new_probs=[]
        for idx, sent in enumerate(ref_news):
            text=ref_news[idx]
            if self.opt.setting=='zero-shot':
                text=text
            else:
                text = "Sentence: " + text + "\n"
            if self.opt.style_mode == 'plm':
                # TODO: Define the prompt and the PLM classification score!
                input_candidate_text = prefix + text + postfix
                style_prob, style_label = predict_next_word(self.plm, self.tokenizer, input_candidate_text,
                                                                direction=self.opt.direction)
                if self.opt.early_stop!=True:
                    style_label = None
                prob_new_probs.append(math.pow(style_prob, self.style_weight))
            elif self.opt.style_mode == 'pipeline':
                res_cand = self.pipeline_classifier(text)
                style_prob,style_label=pipe(res_cand,self.opt.direction)
                prob_new_probs.append(math.pow(style_prob, self.style_weight))

        prob_new_prob=torch.tensor(prob_new_probs).cuda()

        return prob_new_prob,style_label

    def fluency_scorer(self,ref_news): #ref: https://huggingface.co/docs/transformers/perplexity

        encodings = self.tokenizer(ref_news, return_tensors="pt").to(device)
        input_ids = encodings.input_ids
        nlls = []
        for i in range(0, input_ids.size(1), self.stride):
            begin_loc = max(i + self.stride - self.ppl_max_len, 0)
            end_loc = min(i + self.stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids =input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

        return 1/ppl

    def keyword_sim(self,ref_new_embeds,ref_old_embeds,state_vec=None):
        e = 1e-5
        emb1 = ref_new_embeds.permute(0, 2, 1)
        emb2 = ref_old_embeds
        emb_mat = torch.bmm(emb2, emb1)
        weight2 = torch.tensor(state_vec[0][:emb2.shape[1]], dtype=torch.bool)
        norm2 = 1 / (torch.norm(emb2, p=2, dim=2) + e)  # K,8,8
        norm1 = 1 / (torch.norm(emb1, p=2, dim=1) + e)  # K,7,7
        diag_norm2 = torch.diag_embed(norm2)  # K,15,15
        diag_norm1 = torch.diag_embed(norm1)
        sim_mat = torch.bmm(torch.bmm(diag_norm2, emb_mat), diag_norm1)  # K,8,7
        sim_vec, _ = torch.max(sim_mat, dim=2)  # K,8
        try:
            kw_similarity, _ = torch.min(sim_vec[:, weight2], dim=1)
        except:
            weight2[0]=True
            kw_similarity, _ = torch.min(sim_vec[:, weight2], dim=1)
        return kw_similarity

    def semantic_scorer(self,ref_news, ref_olds,state_vec=None):

        ref_new_embeds, mean_new_embeds = self.editor.get_contextual_word_embeddings(ref_news)
        ref_old_embeds, mean_old_embeds = self.editor.get_contextual_word_embeddings(ref_olds)

        #-----keyword-level sim------
        if self.opt.semantic_mode=='kw':
            kw_sim=self.keyword_sim(ref_new_embeds,ref_old_embeds,state_vec)
            similarity = kw_sim.pow(self.keyword_weight)

        #-----sent-level sim------
        elif self.opt.semantic_mode=='sent':
            sent_sim=pytorch_cos_sim(mean_new_embeds, mean_old_embeds)
            similarity=sent_sim.pow(self.sent_weight)

        # -----kw-sent level sim------
        elif self.opt.semantic_mode=='kw-sent':
            kw_sim=self.keyword_sim(ref_new_embeds,ref_old_embeds,state_vec)
            sent_sim= pytorch_cos_sim(mean_new_embeds, mean_old_embeds)
            similarity = kw_sim.pow(self.keyword_weight)* sent_sim.pow(self.sent_weight)

        return similarity

    def scorer(self, input_news,ref_oris,state_vec=None):
        semantic_scores = self.semantic_scorer(input_news,ref_oris,state_vec)
        fluency_scores = self.fluency_scorer(input_news)
        style_score,style_label=self.style_scorer(input_news)
        bleu_score=self.overlap_score(input_news,ref_oris)
        total_scores = fluency_scores.pow(self.fluency_weight) * semantic_scores \
                       * style_score * bleu_score.pow(self.bleu_weight)

        return total_scores.squeeze(),style_score, style_label

    def overlap_score(self,input_news,ref_oris):
        new_tokens=[input_new.split() for input_new in input_news]
        ori_tokens=[[ref_ori.split() for ref_ori in ref_oris]]
        #calculate the 1-gram overlap and that's all
        bleu1=corpus_bleu(ori_tokens, new_tokens, weights=BLEU_WEIGHTS_MEAN[0])

        return torch.tensor(bleu1).to(device)

    def acceptance_prob(self, input_news, input_olds,ref_oris,state_vec=None):
        ref_old_score,old_style_score, _ = self.scorer(input_olds,ref_oris,state_vec)
        ref_old_score=ref_old_score.squeeze()

        ref_new_scores=torch.tensor([self.scorer([ref_hat], ref_oris,state_vec)[0].squeeze() for ref_hat in input_news]).cuda()
        new_style_score=[self.scorer([ref_hat],ref_oris,state_vec)[1].squeeze() for ref_hat in input_news]
        new_style_label=[self.scorer([ref_hat],ref_oris,state_vec)[2] for ref_hat in input_news]
        ref_new_score_index=torch.argmax(ref_new_scores)
        ref_new_score=torch.max(ref_new_scores)

        if ref_new_score-ref_old_score>0:
            accept_hat = [1]
        else:
            accept_hat=[0]

        return accept_hat,ref_new_score_index,ref_old_score,ref_new_score,old_style_score,new_style_score,new_style_label
