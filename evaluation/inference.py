
import argparse
from tqdm import tqdm
import re
import numpy as np
import os
import warnings
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import f1_score, precision_score
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.mol_llama import EDTFormer
from models.configuration import MolLLaMAConfig
from peft import PeftModel
from dataset import ZeroshotDataset, ZeroshotCollater

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_unimol_dictionary():
    """Load UniMol dictionary from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from utils.unicore import Dictionary
    
    logger.info("Loading UniMol dictionary from HuggingFace...")
    unimol_dictionary_path = hf_hub_download(
        repo_id='dptech/Uni-Mol-Models',
        filename='mol.dict.txt',
    )
    unimol_dictionary = Dictionary.load(unimol_dictionary_path)
    unimol_dictionary.add_symbol("[MASK]", is_special=True)
    logger.info(f"✅ Loaded UniMol dictionary with {len(unimol_dictionary)} symbols")
    
    return unimol_dictionary

def main(args):
    # Check if output files already exist
    save_name = f"{args.output_name}_{args.prompt_type}"
    output_dir = os.path.join(args.data_dir, 'results', args.task_name)
    output_file = os.path.join(output_dir, f're_{save_name}.txt')
    acc_file = os.path.join(output_dir, f'acc_{save_name}.txt')
    
    if os.path.exists(output_file) and os.path.exists(acc_file):
        logger.info(f"Output files already exist, skipping inference:")
        logger.info(f"  - {output_file}")
        logger.info(f"  - {acc_file}")
        print(f"⏭️  Skipping inference - output files already exist:")
        print(f"  - {output_file}")
        print(f"  - {acc_file}")
        return
    
    # Load model and tokenizer
    if 'Llama-2' or 'llama-2' in args.pretrained_model_name_or_path:
        llm_version = 'llama2'
    elif 'Llama-3' in args.pretrained_model_name_or_path:
        llm_version = 'llama3'
    elif 'Qwen3' in args.pretrained_model_name_or_path:
        llm_version = 'qwen3'
    elif 'Ministral' in args.pretrained_model_name_or_path:
        llm_version = 'mistral'
    elif 'gemma' in args.pretrained_model_name_or_path:
        llm_version = 'gemma'
    else:
        raise ValueError(f"Unsupported model type. Choose 'llama2', 'llama3', 'qwen3', 'mistral', or 'gemma'.")

    if args.tokenizer_path is None:
        tokenizer_path = args.pretrained_model_name_or_path
    else:
        tokenizer_path = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Only add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Determine if we should use LLM-only mode
    # This should be True if --only_llm flag is set OR if baseline_type is llm_lora/llm_only
    use_llm_only_mode = args.only_llm or args.baseline_type in ['llm_lora', 'llm_only']
    
    # Only add <mol> token when NOT using only_llm mode
    # When only_llm=True, the dataset replaces <mol> with actual SMILES strings,
    # so we don't need (and shouldn't add) the <mol> token to avoid vocabulary mismatch
    if not use_llm_only_mode:
        tokenizer.add_special_tokens({'additional_special_tokens': ["<mol>"]})
        tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]
        logger.info(f"Added <mol> token with ID: {tokenizer.mol_token_id}")
    
    tokenizer.padding_side = 'left'
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")

    terminators = tokenizer.eos_token_id
    # Initialize model directly instead of using from_pretrained
    config = MolLLaMAConfig(
        llm_config={'llm_model': args.pretrained_model_name_or_path},
        qformer_config={'use_dq_encoder': args.use_dq_encoder, 'use_flash_attention': True, 'num_query_tokens': 32, 'embed_dim': 512, 'cross_attention_freq': 1},  # Adjust as needed
        graph_encoder_config={'encoder_types': ['unimol', 'moleculestm'] if args.enable_blending else ['unimol']},  # Adjust as needed
        blending_module_config={'enable_blending': args.enable_blending, 'num_layers': 8, 'num_heads': 8},
        torch_dtype="float16"
    )
    if args.use_dq_encoder and args.baseline_type is None:
        model = EDTFormer(
            config=config,
            vocab_size=len(tokenizer),
            torch_dtype="float16",
            enable_flash=True,
            brics_gids_enable=args.brics_gids_enable,
            entropy_gids_enable=args.entropy_gids_enable,
            enable_blending=args.enable_blending,
            freeze_llm=args.freeze_llm,
            global_q_budget=args.global_q_budget,
            local_q_budget=args.local_q_budget,
        )
        model.load_from_ckpt(args.qformer_path)
    elif args.baseline_type == 'mollama':
        config = MolLLaMAConfig(
            llm_config={'llm_model': args.pretrained_model_name_or_path},
            qformer_config={'use_dq_encoder': args.use_dq_encoder, 'use_flash_attention': True, 'num_query_tokens': 8, 'embed_dim': 256, 'cross_attention_freq': 2, 'max_local_query': 0},  # Adjust as needed
            graph_encoder_config={'encoder_types': ['unimol', 'moleculestm'] if args.enable_blending else ['unimol']},  # Adjust as needed
            blending_module_config={'enable_blending': True, 'num_layers': 4, 'num_heads': 8},
            torch_dtype="float16"
        )
        model = EDTFormer(
            config=config,
            vocab_size=len(tokenizer),
            torch_dtype="float16",
            enable_flash=True,
            brics_gids_enable=False,
            entropy_gids_enable=False,
            enable_blending=args.enable_blending,
            freeze_llm=args.freeze_llm,
        )
        model.load_from_ckpt(args.qformer_path, lora_init=True)
    elif args.baseline_type == 'llm_lora':
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
        model = PeftModel.from_pretrained(model, args.lora_path)

    elif args.baseline_type == 'llm_only':
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
    elif args.only_llm:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
        

    model = model.to(args.device)
    model.eval()

    data_dir = os.path.join(args.data_dir, 'zeroshot', args.task_name)
    unimol_dictionary = load_unimol_dictionary()
    dataset = ZeroshotDataset(data_dir=data_dir, 
                        split='test', prompt_type=args.prompt_type, 
                        unimol_dictionary=unimol_dictionary,
                        only_llm=use_llm_only_mode)
    if hasattr(dataset, 'positive_label') and hasattr(dataset, 'negative_label'):
        positive_label = dataset.positive_label
        negative_label = dataset.negative_label
    else:
        warnings.warn(f"Positive and negative labels not found in meta.json, using 'positive' and 'negative' as default.")

    collater = ZeroshotCollater(tokenizer, unimol_dictionary, llm_version, use_llm_only_mode)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collater, shuffle=False)

    # ----------- Generation ----------- #
    pattern = r"[Ff]inal [Aa]nswer:"
    responses, answers, smiles_list = [], [], []
    for graph_batch, text_batch, answer, smiles, brics_gids, entropy_gids in tqdm(dataloader):
        # Only process graph_batch if NOT in LLM-only mode
        if not use_llm_only_mode:
            for key in graph_batch.keys():
                if key == 'unimol':
                    for key_ in graph_batch[key].keys():
                        graph_batch[key][key_] = graph_batch[key][key_].to(args.device)
                elif key == 'moleculestm':
                    graph_batch[key] = graph_batch[key].to(args.device)
            
            # Add brics_gids and entropy_gids to graph_batch
            graph_batch['brics_gids'] = brics_gids
            graph_batch['entropy_gids'] = entropy_gids
        
        text_batch = text_batch.to(args.device)

        # Generate
        if use_llm_only_mode:
            outputs = model.generate(
                inputs = text_batch['input_ids'],
                attention_mask = text_batch['attention_mask'],
                max_new_tokens = 512,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = terminators,
            )
        else:

            outputs = model.generate(
                graph_batch = graph_batch,
                text_batch = text_batch,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = terminators,
            )
        
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        original_texts = tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)

        # --- Generate one more time if the output does not contain "Final answer:" --- #
        no_format_indices = []
        new_texts = []
        for idx, (original_text, generated_text) in enumerate(zip(original_texts, generated_texts)):
            if not re.search(pattern, generated_text):
                no_format_indices.append(idx)
                new_texts.append(original_text + generated_text + "\n\nFinal answer: ")
        
        if len(no_format_indices) > 0:
            # Only process graph batch if NOT in LLM-only mode
            if not use_llm_only_mode:
                new_graph_batch = {"unimol": {}, "moleculestm": {}}
                for k, v in graph_batch['unimol'].items():
                    new_graph_batch['unimol'][k] = v[no_format_indices]
                new_graph_batch['moleculestm'] = Batch.from_data_list(graph_batch['moleculestm'].index_select(no_format_indices))
                
                # Add brics_gids and entropy_gids to new_graph_batch
                new_graph_batch['brics_gids'] = [list(brics_gids)[i] for i in no_format_indices]
                new_graph_batch['entropy_gids'] = [list(entropy_gids)[i] for i in no_format_indices]

            new_text_batch = tokenizer(
                new_texts,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to(args.device)
            
            # Only set mol_token_flag when NOT in only_llm mode (EDTFormer needs it)
            if not use_llm_only_mode:
                new_text_batch.mol_token_flag = (new_text_batch.input_ids == tokenizer.mol_token_id).to(args.device)

            if use_llm_only_mode:
                new_generated_texts = model.generate(
                    inputs = new_text_batch['input_ids'],
                    attention_mask = new_text_batch['attention_mask'],
                    max_new_tokens = 512,
                    pad_token_id = tokenizer.pad_token_id,
                    eos_token_id = terminators,
                )
            else:
                new_generated_texts = model.generate(
                graph_batch = new_graph_batch,
                text_batch = new_text_batch,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = terminators,
            )

            new_generated_texts = tokenizer.batch_decode(new_generated_texts, skip_special_tokens=True)

            for _, i in enumerate(no_format_indices):
                generated_texts[i] += "\n\nFinal answer: " + new_generated_texts[_]
            # --- Add the new generated texts to the generated texts --- #
            
        # --- Get all the generated texts --- #
        responses.extend(generated_texts)
        answers.extend(answer)
        smiles_list.extend(smiles)
        

    # ----------- Evaluation ----------- #
    # Process just one batch for testing
    # Hard code for BBBP
    # Ensure "Non-penetrant" is not misclassified as "Penetrant" by checking the negative case first
    labels, preds = [], []
    print(args.debug_mode, "debug_mode",positive_label, negative_label)

    for response, answer in zip(responses, answers):
        if re.search(negative_label, answer.lower()):
            label = 0
        elif re.search(positive_label, answer.lower()):
            label = 1
        else:
            label = None

        if label is None:
            warnings.warn(f"Label not found in answer: {answer}")
            continue

        final_answer_text = response.split("Final answer: ")[-1].strip()
        final_answer_text_lower = final_answer_text.lower()

        if re.search(f"{negative_label}", final_answer_text_lower):
            pred = 0
        elif re.search(f"{positive_label}", final_answer_text_lower):
            pred = 1
        else:
            pred = None

        labels.append(label)
        preds.append(pred)

    non_rate = len([p for p in preds if p is None]) / len(preds) * 100
    

    # -------- End of evaluation ----------- #


    # Save the results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f're_{save_name}.txt'), 'w', encoding='utf-8') as f:
        for response, answer, smiles, label, pred in zip(responses, answers, smiles_list, labels, preds):
            f.write(f"SMILES: {smiles}\n")
            f.write('-'*50 + "\n")
            f.write(f"Label: {label}\n")
            f.write(f"Prediction: {pred if pred is not None else 'None'}\n")
            f.write('-'*50 + "\n")
            f.write(f"Response: {response}\n")
            f.write('-'*50 + "\n")
            f.write(f"Answer: {answer}\n")
            f.write("="*50 + "\n")

    # Calculate accuracy
    preds, labels = np.array(preds), np.array(labels)
    mask = preds != None
    labels = labels[mask]
    preds = preds[mask]

    labels = labels.tolist()
    preds = preds.tolist()

    accuracy = (np.array(preds) == np.array(labels)).mean() * 100
    # Calculate F1 score
    if len(labels) > 0:
        
        f1 = f1_score(labels, preds) * 100
        precision = precision_score(labels, preds, zero_division=0) * 100
    else:
        f1 = 0.0
        precision = 0.0

    print(f'F1 Score: {f1:.2f}%')
    print(f'Precision: {precision:.2f}%')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f"Non-rate: {non_rate:.2f}%")

    with open(os.path.join(output_dir, f'acc_{save_name}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {accuracy:.2f}%\n')            
        f.write(f"Non-rate: {non_rate:.2f}%\n")
        f.write(f'F1 Score: {f1:.2f}%\n')
        f.write(f'Precision: {precision:.2f}%\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, default='pampa')
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--qformer_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--prompt_type', type=str, default='default')
    parser.add_argument('--output_name', type=str, default="zeroshot")
    parser.add_argument('--only_llm', default=False, action='store_true')

    parser.add_argument('--use_dq_encoder', default=False, action='store_true')
    parser.add_argument('--brics_gids_enable', default=False, action='store_true')
    parser.add_argument('--entropy_gids_enable', default=False, action='store_true')
    parser.add_argument('--debug_mode', default=False, action='store_true')
    parser.add_argument('--enable_blending', default=False, action='store_true')
    parser.add_argument('--freeze_llm', default=False, action='store_true')
    parser.add_argument('--baseline_type', type=str, default=None, choices=['mollama', 'llm_lora', 'llm_only'])
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--global_q_budget', type=int, default=None)
    parser.add_argument('--local_q_budget', type=int, default=None)
    args = parser.parse_args()
    main(args)
