from os import makedirs
from os.path import join
import logging
import numpy as np
import torch
import random
from configs.finetune.args import define_main_parser

# Fix 1: Use DataCollatorForLanguageModeling instead of DefaultDataCollator
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from dataset.prsa import PRSADataset
from dataset.card import TransactionDataset, TransactionDatasetEmbedded
from dataset.vocab import Vocabulary
from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset, get_save_steps_for_epoch
from dataset.datacollator import TransDataCollatorForFineTuning
from testing.test_fraud_utils import test_fraud_model, compute_metrics_fraud
from models.lstm import build_baseline_model, PreTrainedModelWrapper
from transformers import EarlyStoppingCallback

logger = logging.getLogger(__name__)
log = logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
            
def main(args):
    # random seeds
    seed = args.seed
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {device_count}")
        for i in range(device_count):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available. Using CPU.")

    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_embeddings = args.use_embeddings
    input_type = "EMBEDDING FROM PRETRAINED MODEL" if use_embeddings else "RAW FEATURES"
    print("RUNNNING TRAINING SCRIPT TO PRETRAIN DOWNSTREAM MODEL ON FRAUD PREDICTION WITH " + input_type)
    

    vocab = Vocabulary.from_file("results/15072025_50k_v1/vocab.nb")

        
    dataset = TransactionDataset(root=args.data_root,
                                    fname=args.data_fname,
                                    seq_len=args.seq_len,
                                    fextension=args.data_extension,
                                    vocab_dir=args.output_dir,
                                    nrows=args.nrows,
                                    user_ids=args.user_ids,
                                    mlm=args.mlm,
                                    cached=args.cached,
                                    stride=args.stride,
                                    flatten=args.flatten,
                                    return_labels=True,
                                    skip_user=args.skip_user,
                                    vocab = vocab,
                                    encoder_path = args.encoder_path,
                                    seed = args.seed)


    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()

    #only need pretrained model when creating downstream model with output embeddings from pretrained Tabformer 
    if use_embeddings: 
        print("CREATING EMBEDDING DATASET LOADER")
        pretrained_model = TabFormerBertLM(
                                   custom_special_tokens,
                                   vocab=vocab,
                                   field_ce=args.field_ce,
                                   flatten=args.flatten,
                                   ncols=dataset.ncols,
                                   field_hidden_size=args.field_hs, 
                                   return_embeddings=True
                                   )
        print("EMBEDDINGING HAVE BEEN CREATED")
    
        #save model
        model_path = join(args.path_to_checkpoint, f'checkpoint-{args.checkpoint}')
        pretrained_model.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
        log.info(f"Loaded model from checkpoint: {model_path}")
        log.info(f"model initiated: {pretrained_model.model.__class__}")


    epochs = args.num_train_epochs  # Fix 2: Remove duplicate assignment
    batch_size = args.batch_size
    baseline_model_type = args.model_type
    hidden_size = args.hidden_size
    num_layers = args.num_layers_lstm
    sequence_len = args.seq_len
    input_size = args.hidden_size
    field_input_size = args.field_hs
    vocab_length = len(vocab)
    

    
    #if using the output embeddings from the pretrained Tabformer as input to the downstream model (i.e., LSTM) 
    if use_embeddings: 
        print("Creating Dataset with Output Model of pretrained Tabformer model")
        dataset = TransactionDatasetEmbedded(
                            pretrained_model=pretrained_model,
                            raw_dataset=dataset,
                            batch_size=batch_size, 
                            force_recompute = False,
                            seed = args.seed
                            )
    
    model = build_baseline_model(baseline_model_type, use_embeddings, hidden_size, sequence_len, num_layers, input_size, field_input_size, vocab_length, args.equal_parameters_baselines)
    model = PreTrainedModelWrapper(model)


    # split dataset into train, val, test [0.6. 0.2, 0.2]
    totalN = len(dataset)
    trainN = int(0.6 * totalN)

    valtestN = totalN - trainN
    valN = int(valtestN * 0.5)
    testN = valtestN - valN

    assert totalN == trainN + valN + testN

    lengths = [trainN, valN, testN]

    log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))

    # totalN = len(dataset)
    # testN = 3000  # Fixed test set size
    
    # # Ensure we have enough samples
    # if totalN < testN:
    #     raise ValueError(f"Dataset size ({totalN}) is smaller than required test size ({testN})")
    
    # # Calculate remaining samples for train and validation
    # remaining_samples = totalN - testN
    # trainN = int(0.80 * remaining_samples)  # 80% of remaining for training
    # valN = remaining_samples - trainN       # 25% of remaining for validation
    
    # assert totalN == trainN + valN + testN
    
    # lengths = [trainN, valN, testN]
    
    # log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    # log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
    #                                                                            testN / totalN))

    train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)
    train_indices, test_indices, eval_indices = dataset.resample_train(train_dataset.indices, test_dataset.indices, eval_dataset.indices)
    train_dataset.dataset, train_dataset.indices = dataset, train_indices
    eval_dataset.dataset, eval_dataset.indices = dataset, eval_indices
    test_dataset.dataset, test_dataset.indices = dataset, test_indices
    if valN == 0:
        eval_dataset = test_dataset

    # Calc the steps
    args.save_steps = get_save_steps_for_epoch(args, train_dataset)
    print("Numper savings steps per epoch", args.save_steps)

    data_collator = TransDataCollatorForFineTuning(tokenizer=None, mlm=False, mlm_probability=None)

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)
    
    training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    logging_dir=args.log_dir,
    
    save_steps=args.save_steps,
    do_train=args.do_train,
    do_eval=args.do_eval,
    
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    save_total_limit=2,
    
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    
    prediction_loss_only=False,  # Keep this False to enable compute_metrics
    overwrite_output_dir=True,
    # save_safetensors=False,
    label_names=["labels"],
    # Remove these lines that are causing the issue
    metric_for_best_model="eval_F1_fraud",
    greater_is_better=True,
    load_best_model_at_end=True,
        
    # include_inputs_for_metrics=True  # Change this to True
    
    # Remove parameters not available in 3.2.0
    remove_unused_columns=False
    
    # For showing metrics, ensure we're not prediction_loss_only
    # prediction_loss_only=False,  # This is default in 3.2.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fraud,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    
    model.eval()

    with torch.no_grad():
        test_fraud_model(model, test_dataset, args.output_dir)


if __name__ == "__main__":

    parser = define_main_parser()
    opts = parser.parse_args()

    opts.log_dir = join(opts.output_dir, "logs")
    makedirs(opts.output_dir, exist_ok=True)
    makedirs(opts.log_dir, exist_ok=True)
    
    if not opts.mlm and opts.lm_type == "bert":
        raise Exception("Error: Bert needs '--mlm' option. Please re-run with this flag included.")

    main(opts)