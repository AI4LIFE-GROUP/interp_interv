# CODE MODIFIED FROM CODEBASE FOR PAPER "Designing a Dashboard for Transparency and Control of Conversational AI" BY Yida Chen et al.

import sklearn.model_selection
import os
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from nnsight import LanguageModel
import pandas as pd
import pdb
import argparse
import time
import numpy as np

tic, toc = (time.time, time.time)

class LinearProbeClassification(torch.nn.Module):
    def __init__(self, device, probe_class, input_dim=512, logistic=False, Relu=False, TanH=False):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        if logistic:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.probe_class),
                torch.nn.Sigmoid()
            )
        elif Relu:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.probe_class),
                torch.nn.ReLU(True)
            )
        elif TanH:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.probe_class),
                # nn.Hardtanh(inplace=True, min_val=0.001, max_val=0.999)
                torch.nn.Hardsigmoid(inplace=True)
            )
        else:
            
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.probe_class),
            )
        
        self.apply(self._init_weights)
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.to(device)

    def forward(self, act, y=None):
        logits = self.proj(act)
        return logits
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler

def optimize_one_inter_rep(inter_rep, probe, layer_name, target=torch.Tensor([1]), 
                           lr=100, max_epoch=1e5, 
                           loss_func=torch.nn.BCELoss(), 
                           verbose=False, simplified=True, N=10, normalized=False, device="cuda"):
    global first_time
    tensor = (inter_rep.clone()).to(device).requires_grad_(True)
    rep_f = lambda: tensor
    target_clone = target.clone().to(device).to(torch.float)

    optimizer = torch.optim.Adam([tensor], lr=lr)

    cur_input_tensor = rep_f().clone().detach()

    if normalized:
        cur_input_tensor = rep_f() + target_clone.view(1, -1) @ probe.proj[0].weight * N * 100 / rep_f().norm() 
    else:
        cur_input_tensor = rep_f() + target_clone.view(1, -1) @ probe.proj[0].weight * N

    return cur_input_tensor.clone()


class TextDataset(Dataset):
    def __init__(self, file, model, model_name, layer_idx):
        self.file = file
        self.model = model
        self.model_name = model_name
        self.layer_idx = layer_idx

        self.labels = []
        self.acts = []
        self.texts = []

        self._load_in_data()

    def __len__(self):
        return len(self.texts)
    
    def _load_in_data(self):

        pairs = pd.read_csv(self.file, dtype=str, header=0)

            
        with torch.no_grad():
            for index, pair in pairs.iterrows():
                print(index, flush=True)
                if self.model_name != "gpt2":
                    with self.model.trace() as tracer:
                        with tracer.invoke(pair["pos_prompt"]):
                            p_pos = self.model.model.layers[self.layer_idx].output[0][0, 1:].mean(0).save()
                    with self.model.trace() as tracer:
                        with tracer.invoke(pair["neg_prompt"]):
                            p_neg = self.model.model.layers[self.layer_idx].output[0][0, 1:].mean(0).save()

                else:
                    with self.model.trace() as tracer:
                        with tracer.invoke(pair["pos_prompt"]):
                            p_pos = self.model.transformer.h[self.layer_idx].output[0][0, 1:].mean(0).save()
                    with self.model.trace() as tracer:
                        with tracer.invoke(pair["neg_prompt"]):
                            p_neg = self.model.transformer.h[self.layer_idx].output[0][0, 1:].mean(0).save()
        
                self.texts.append(pair["pos_prompt"])
                self.labels.append(1)
                self.acts.append(p_pos.value)
                self.texts.append(pair["neg_prompt"])
                self.labels.append(0)
                self.acts.append(p_neg.value)

        self.acts = torch.stack(self.acts).cpu()
        self.labels = torch.Tensor(self.labels).unsqueeze(1).cpu()
            

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
 
        hidden_states = self.acts[idx]
        
        return {
            'hidden_states': hidden_states,
            'label': label,
            'text': text,
        }


def train(probe, device, train_loader, optimizer, epoch, loss_func, 
          report=False, verbose_interval=5, verbose=True, return_raw_outputs=False):
    """
    :param model: pytorch model (class:torch.nn.Module)
    :param device: device used to train the model (e.g. torch.device("cuda") for training on GPU)
    :param train_loader: torch.utils.data.DataLoader of train dataset
    :param optimizer: optimizer for the model
    :param epoch: current epoch of training
    :param loss_func: loss function for the training
    :param class_names: str Name for the classification classses. used in train report
    :param report: whether to print a classification report of training 
    :param train_verbose: print a train progress report after how many batches of training in each epoch
    :return: average loss, train accuracy, true labels, predictions
    """
    assert (verbose_interval is None) or verbose_interval > 0, "invalid verbose_interval, verbose_interval(int) > 0"
    starttime = tic()
    # Set the model to the train mode: Essential for proper gradient descent
    probe.train()
    loss_sum = 0
    correct = 0
    tot = 0
    
    preds = []
    truths = []
    
    # Iterate through the train dataset
    for batch_idx, batch in enumerate(train_loader):
        batch_size = 1
        target = batch["label"].cuda()
    
        optimizer.zero_grad()

        act = batch["hidden_states"].to("cuda")
        output = probe(act)
        loss = loss_func(output, target) + 0.3*(sum(p.abs().sum() for p in probe.parameters()))
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.sum().item()  
        
        pred = torch.argmax(output, axis=1)

        # In the Scikit-Learn's implementation of OvR Multi-class Logistic Regression. They linearly normalized the predicted probability and then call argmax
        # Below is an equivalent implementation of the scikit-learn's decision function. The only difference is we didn't do the linearly normalization
        # To save some computation time
        if len(target.shape) > 1:
            target = torch.argmax(target, axis=1)
        correct += np.sum(np.array(pred.detach().cpu().numpy()) == np.array(target.detach().cpu().numpy()))
        if return_raw_outputs:
            preds.append(pred.detach().cpu().numpy())
            truths.append(target.detach().cpu().numpy())
        tot += pred.shape[0] 
    
    train_acc = correct / tot
    loss_avg = loss_sum / len(train_loader)
    
    endtime = toc()
    if verbose:
        print('\nTrain set: Average loss: {:.4f} ({:.3f} sec) Accuracy: {:.3f}\n'.\
              format(loss_avg, 
                     endtime-starttime,
                     train_acc))
        
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
        
    if return_raw_outputs:
        return loss_avg, train_acc, preds, truths
    else:
        return loss_avg, train_acc
    

def test(probe, device, test_loader, loss_func, return_raw_outputs=False, verbose=True,
        scheduler=None):
    """
    :param model: pytorch model (class:torch.nn.Module)
    :param device: device used to train the model (e.g. torch.device("cuda") for training on GPU)
    :param test_loader: torch.utils.data.DataLoader of test dataset
    :param loss_func: loss function for the training
    :param class_names: str Name for the classification classses. used in train report
    :param test_report: whether to print a classification report of testing after each epoch
    :param return_raw_outputs: whether return the raw outputs of model (before argmax). used for auc computation
    :return: average test loss, test accuracy, true labels, predictions, (and raw outputs 
    from model if return_raw_outputs)
    """
    # Set the model to evaluation mode: Essential for testing model
    probe.eval()
    test_loss = 0
    tot = 0
    correct = 0
    preds = []
    truths = []
        
    # Do not call gradient descent on the test set
    # We don't adjust the weights of model on the test set
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_size = 1
            target = batch["label"].cuda()
            act = batch["hidden_states"].to("cuda")

            output = probe(act)
            pred = torch.argmax(output, axis=1)
            
            loss = loss_func(output, target)
            test_loss += loss.sum().item()  # sum up batch loss

            # In the Scikit-Learn's implementation of OvR Multi-class Logistic Regression. They linearly normalized the predicted probability and then call argmax
            # Below is an equivalent implementation of the scikit-learn's decision function. The only difference is we didn't do the linearly normalization
            # To save some computation time
            if len(target.shape) > 1:
                target = torch.argmax(target, axis=1)
            
            
            pred = np.array(pred.detach().cpu().numpy())
            target = np.array(target.detach().cpu().numpy())
            correct += np.sum(pred == target)
            tot += pred.shape[0] 
            if return_raw_outputs:
                preds.append(pred)
                truths.append(target)
                
    test_loss /= len(test_loader)
    if scheduler:
        scheduler.step(test_loss)
    
    test_acc = correct / tot

    if verbose:
        print('Test set: Average loss: {:.4f},  Accuracy: {:.3f}\n'.format(
              test_loss,
              test_acc))
        
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
        
    # If return the raw outputs (before argmax) from the model
    if return_raw_outputs:
        return test_loss, test_acc, preds, truths
    else:
        return test_loss, test_acc

class TrainerConfig:
    # optimization parameters
    learning_rate = 1e-3
    betas = (0.9, 0.95)
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    # checkpoint settings

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-intervention_phrase', type=str, default="blue")
    parser.add_argument('-layer_idx', type=int, default=18)
    parser.add_argument('-model', type=str, default="llama2")
    parser.add_argument('-device', type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    model_str = "google/gemma-2-2b" if args.model == "gemma2" else "meta-llama/Llama-2-7b-chat-hf"
    model_str = "openai-community/gpt2" if args.model == "gpt2" else model_str

    model = LanguageModel(model_str, device_map=device, dispatch=True)
    for param in model.parameters():
        param.requires_grad = False

    dataset = TextDataset('data/' + args.intervention_phrase + "_pairs.csv", model, args.model, layer_idx=args.layer_idx)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_idx, val_idx = sklearn.model_selection.train_test_split(list(range(len(dataset))), 
                                                                  test_size=test_size,
                                                                  train_size=train_size,
                                                                  random_state=12345,
                                                                  shuffle=True,
                                                                  stratify=dataset.labels,
                                                                 )

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, val_idx)

    sampler = None
    train_loader = DataLoader(train_dataset, shuffle=True, sampler=sampler, pin_memory=True, batch_size=16)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=100)

    # gemma 2304, llama 4096, gpt 768
    accs = []
    final_accs = []
    train_accs = []

    input_dim = 2304 if args.model == "gemma2" else 4096
    input_dim = 768 if args.model == "gpt2" else input_dim
    
    loss_func = torch.nn.BCELoss()
    trainer_config = TrainerConfig()
    probe = LinearProbeClassification(probe_class=1, device=device, input_dim=input_dim, logistic=True)
    optimizer, scheduler = probe.configure_optimizers(trainer_config)
    best_acc = 0
    max_epoch = 50
    verbosity = False

    for epoch in range(1, max_epoch + 1):

        # Get the train results from training of each epoch
        train_results = train(probe, device, train_loader, optimizer, 
                                    epoch, loss_func=loss_func, verbose_interval=None,
                                    verbose=True,
                                    return_raw_outputs=True)
        test_results = test(probe, device, test_loader, loss_func=loss_func, 
                                return_raw_outputs=True, verbose=True,
                                scheduler=scheduler)

        if test_results[1] > best_acc:
            best_acc = test_results[1]
            torch.save(probe.state_dict(), "steering_probes/" + args.model + "_" + str(args.layer_idx) + "_" + args.intervention_phrase + "_probe_best.pth")

    torch.save(probe.state_dict(), "steering_probes/" + args.model + "_" + str(args.layer_idx) + "_" + args.intervention_phrase + "_probe_final.pth")

if __name__ == "__main__":
    
    main()
        