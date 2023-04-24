import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets.dataset import LADataset
from utils.evaluation_metric import compute_eer_and_tdcf, compute_eer, compute_tDCF, obtain_asv_error_rates
from tqdm import tqdm
import numpy as np
from datasets.dataset import collate_fn

def test_model(feat_model_path, device):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    model = torch.load(feat_model_path, map_location=device)
    model = model.to(device)
    # Configure txtpath and data directory
    test_txtpath = "/data/Synthetic-Speech-Detection/datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    test_datadir = "/data/Synthetic-Speech-Detection/datasets/LA/ASVspoof2019_LA_eval/"
    test_dataset = LADataset(split="eval", transforms="lfcc", n_fft=512, num_features=20, txtpath=test_txtpath, datadir=test_datadir)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=3, collate_fn=collate_fn)
    model.eval()

    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        for i, (inputs, targets, audio_fn) in enumerate(tqdm(test_dataloader)):
            inputs, labels = inputs.to(device), targets["label"].to(device)

            features, probs = model(inputs)
            score = F.softmax(probs, dim=1)[:, 0]

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], targets["tag"][j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score.txt'), "/data/Synthetic-Speech-Detection/datasets")
                                            
    with open("result.txt","a") as result_file:
        result_file.write("\n"+feat_model_path+"\n")
        result_file.write(str(eer_cm*100)+"       "+str(min_tDCF)+"\n")

    return eer_cm, min_tDCF

def test(model_dir, device):
    model_path = os.path.join(model_dir, "best_model.pt")
    eer_cm, min_tDCF = test_model(model_path, device) 

def test_individual_attacks(cm_score_file):
    asv_score_file = os.path.join('/data/Synthetic-Speech-Detection/datasets',
        'LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    other_cm_scores = -cm_scores

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(7,20):
        # Extract target, nontarget, and spoof scores from the ASV scores
        tar_asv = asv_scores[asv_keys == 'target']
        non_asv = asv_scores[asv_keys == 'nontarget']
        spoof_asv = asv_scores[asv_sources == 'A%02d' % attack_idx]

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]

        other_eer_cm = compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_sources == 'A%02d' % attack_idx])[0]

        [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

        if eer_cm < other_eer_cm:
            # Compute t-DCF
            tDCF_curve, CM_thresholds = compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]

        else:
            tDCF_curve, CM_thresholds = compute_tDCF(other_cm_scores[cm_keys == 'bonafide'],
                                                     other_cm_scores[cm_sources == 'A%02d' % attack_idx],
                                                     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]
        eer_cm_lst.append(min(eer_cm, other_eer_cm))
        min_tDCF_lst.append(min_tDCF)

    return eer_cm_lst, min_tDCF_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()
    args.device = torch.device("cuda:{:d}".format(int(args.gpu)) if torch.cuda.is_available() else "cpu")
    test(args.model_dir, args.device)
