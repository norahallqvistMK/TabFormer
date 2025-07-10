import os
from os.path import join
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import json
import transformers
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import code
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can use absolute import
from configs.finetune.args import define_main_parser
from dataset.datacollator import TransDataCollatorForFineTuning
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = define_main_parser()
opts = parser.parse_args()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def compute_metrics_fraud(eval_preds):
    print('starting compute metrics')
    # logger.info(eval_preds)
    labels = eval_preds.label_ids
    scores = eval_preds.predictions
    # logger.info(labels)
    # logger.info(scores)
    
    # Aggregate labels - check if any label in the sequence is positive
    labels = np.any(labels, axis=1).astype(np.int32)
    scores = sigmoid(scores)
    assert labels.shape == scores.shape

    # Compute AUC scores
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)  # Fixed variable name
    print('computed aucs')
    
    # Convert scores to binary predictions with 0.5 threshold
    binary_scores = np.array(scores > 0.5).astype(np.int32)
    confusion_mat = confusion_matrix(labels, binary_scores)

    # Extract confusion matrix values
    tp, fp = confusion_mat[1, 1], confusion_mat[0, 1]
    tn, fn = confusion_mat[0, 0], confusion_mat[1, 0]

    # Compute various metrics
    accuracy_global = accuracy_score(labels, binary_scores)  # Fixed variable name
    accuracy_mean_class = balanced_accuracy_score(labels, binary_scores)  # Fixed variable name
    f1_fraud = f1_score(labels, binary_scores, average='binary')
    f1_mean = f1_score(labels, binary_scores, average='macro')
    
    print('finished compute metrics')
    
    # Create metrics dictionary with correct variable names
    metrics = {
        'ROC_AUC': float(roc_auc),
        'PrecRec_Auc': float(pr_auc),  # Fixed variable name
        'accuracy_global': float(accuracy_global),  # Fixed variable name
        'accuracy_mean_class': float(accuracy_mean_class),  # Fixed variable name
        'F1_fraud': float(f1_fraud),
        'F1_mean': float(f1_mean),
        'true_pos': int(tp),    # Fixed variable name
        'false_pos': int(fp),   # Fixed variable name
        'false_neg': int(fn),   # Fixed variable name
        'true_neg': int(tn)     # Fixed variable name
    }
    
    return metrics  
    
def test_fraud_model(model, test_dataset, output_dir):
    # logger.debug("\n")
    batch_size = 8
    print("", opts)
    # batch_size = getattr(opts, 'batch_size', 2) or 2
    collator = TransDataCollatorForFineTuning(tokenizer=None, mlm=False, mlm_probability=None)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)
    
    model.eval()
    all_losses = []
    all_probs = np.array((0,))
    all_labels = np.array((0,))
    
    for input in tqdm(test_dataloader):
        input_ids = input['input_ids'].to(device)
        labels = input['labels']
        
        # Convert labels to tensor if it's a list
        if isinstance(labels, list):
            # Stack along dimension 0 to get [batch_size, sequence_length]
            labels = torch.stack(labels, dim=0)

            if labels.shape[0] != input_ids.shape[0]:
                print(f"Transposing labels from {labels.shape} to match batch size {input_ids.shape[0]}")
                labels = labels.transpose(0, 1)
                    
        # Move labels to the same device as the model
        labels = labels.to(device)
        
        output = model(input_ids=input_ids, labels=labels)
        loss, logits = output
        aggregated_labels = torch.any(labels.bool(), dim=1).to(torch.float32).squeeze()
        probs = logits.sigmoid().squeeze()
        all_losses.append(loss.cpu().numpy())
        all_probs = np.concatenate((all_probs, probs.cpu().numpy()))
        all_labels = np.concatenate((all_labels, aggregated_labels.cpu().numpy()))
    
    assert all_probs.shape == all_labels.shape, f'probs shape {probs.shape} is not equal to labels shape {labels.shape}'
    # Statistics
    print_statistics(all_probs, all_labels, output_dir)

def binary_results_from_predictions(preds, labels):

    confusion_mat = confusion_matrix(labels, preds)

    tp, fp = int(confusion_mat[1,1]), int(confusion_mat[0,1])
    tn, fn = int(confusion_mat[0,0]), int(confusion_mat[1,0])

    accuracy = accuracy_score(labels, preds)
    accuracy_mean = balanced_accuracy_score(labels, preds)
    f1_class1 = f1_score(labels, preds, average='binary')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': float(accuracy),
        'accuracy_mean': float(accuracy_mean),
        'confusion_matrix': [[tp, fp], [fn, tn]],
        'f1_target': float(f1_class1),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
    }


def print_statistics(probs, labels, output_dir):

    ### FIRST COMPUTE RESULTS NOT DEPENDENT FROM THRESHOLD 
    ### THEN TRY DIFFERENT THRESHOLDING POLICIES

    # COMPUTE ROC Area-Under-Curve
    roc_auc = roc_auc_score(labels, probs)
    # COMPUTE Precision-Recall Area-Under-Curve
    PR_auc_classes = average_precision_score(labels, probs, average=None)
    PR_auc_macro = average_precision_score(labels, probs, average='macro')

    # Consider banal threshold of 0.5
    preds_05 = np.array(probs>0.5).astype(np.int32)
    results_05 = binary_results_from_predictions(preds_05, labels)
    logger.info(f'\nCOMPUTED RESULTS WITH thr=0.5 : \n{json.dumps(results_05, indent=4)}')

    # estimate threshold from precision recall curve maximizing F1-Score
    precision_list, recall_list, thresholds_list = precision_recall_curve(labels, probs)
    assert min(thresholds_list)>=0 and max(thresholds_list)<=1, f'the thresholds are not in [0,1] but in [{min(thresholds_list)},{max(thresholds_list)}], \n->{thresholds_list}'
    f1_list = (2*precision_list*recall_list)/(precision_list+recall_list)
    index1 = np.nanargmax(f1_list)
    threshold_f1 = thresholds_list[index1]
    preds_f1 = np.array(probs>threshold_f1).astype(np.int32)
    results_f1 = binary_results_from_predictions(preds_f1, labels)
    logger.info(f'\nCOMPUTED RESULTS WITH thr={round(threshold_f1,2)}, MAXIMIZING F1-SCORE : \n{json.dumps(results_f1, indent=4)}')

    # estimate threshold from roc curve maximizing G-mean
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)
    gmeans = np.sqrt(tpr * (1-fpr))
    index2 = np.nanargmax(gmeans)
    threshold_roc = thresholds_roc[index2]
    preds_roc = np.array(probs>threshold_roc).astype(np.int32)
    results_roc = binary_results_from_predictions(preds_roc, labels)
    logger.info(f'\nCOMPUTED RESULTS WITH thr={round(threshold_roc,2)}, MAXIMIZING G-MEAN : \n{json.dumps(results_roc, indent=4)}')

    # PLOT PRECISION RECALL CURVE WITH AVERAGE PRECISION AND F1 SCORE
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig_img_rocauc = ax
    fig_img_rocauc.plot(recall_list, precision_list, label='precision recall curve')
    fig_img_rocauc.plot(recall_list, f1_list, label='precision recall f1 score')
    fig_img_rocauc.scatter(recall_list[index1], precision_list[index1], c='r', label='optimal threshold')
    fig_img_rocauc.title.set_text('Image precision recall, AP-score: {0:.2f} F1-score: {1:.2f}%, Optimal Threshold: {2:.2f}'.format(PR_auc_macro, f1_list[index1], threshold_f1))
    fig_img_rocauc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'prec-rec_curve.png'), dpi=100)

    # PLOT ROC CURVE WITH ROC-AUC AND G-MEAN SCORE
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig_img_rocauc = ax
    fig_img_rocauc.plot(fpr, tpr, label='img_ROCAUC:{0:.3f}'.format(roc_auc))
    fig_img_rocauc.scatter(fpr[index2], tpr[index2], c='r', label='optimal threshold')
    fig_img_rocauc.title.set_text('Image ROCAUC:{0:.3f}, G-MEAN: {1:.1f}%, Optimal Threshold: {2:.2f}'.format(roc_auc, gmeans[index2], threshold_roc))
    fig_img_rocauc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=100)

    stats_dict = {
        'AUC': float(roc_auc),
        'Average-Precisions': float(PR_auc_classes),
        'Average_precision_macro': float(PR_auc_macro),
        'Results thr=0,5': results_05,
        'Optimal thr F1': float(threshold_f1),
        'Results optimal F1': results_f1,
        'Optimal thr Gmean': float(threshold_roc),
        'Results optimal Gmean': results_roc,
    }
    with open(join(output_dir, 'main_statistics.json'), 'w+') as fw:
        json.dump(stats_dict, fw, indent=4)