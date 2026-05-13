
import argparse
import json
import logging
import os
import time
import warnings
from typing import Any, Optional

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.mol_llama import EDTFormer
from models.configuration import MolLLaMAConfig
from peft import PeftModel
from evaluation.dataset import (
    FunctionalGroupHallucinationDataset,
    ZeroshotCollater,
)

try:
    import openai
except ImportError:  # pragma: no cover - handled at runtime
    openai = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SYSTEM_PROMPT = (
    "You are a senior computational chemist who evaluates hallucinations (false positives) in "
    "functional group extraction tasks. Given a molecule's SMILES string, the "
    "ground truth functional groups (comma-separated), and a model response, "
    "decide if the model invented any functional groups that are NOT present in the molecule "
    "according to the ground truth. Missing groups is acceptable - we only flag hallucinations "
    "(false positives). Be lenient with synonyms and alternative names for functional groups "
    "(e.g., carboxylic acid = carboxyl, amine = amino). Only flag as hallucination if the "
    "model predicts a functional group that genuinely does not exist in the ground truth. "
    "Respond with ONLY 'true' or 'false':\n"
    "- 'true' if the model hallucinates at least one functional group not in the ground truth\n"
    "- 'false' if all predicted functional groups are valid (or if the model predicts nothing)"
)

OPENAI_CLIENT = None
OPENAI_CLIENT_KIND = None  # "client" for new SDK, "legacy" for old ChatCompletion API


def ensure_openai_client(api_key: Optional[str]) -> None:
    """Configure the OpenAI client for both legacy and new SDKs."""
    global OPENAI_CLIENT, OPENAI_CLIENT_KIND

    if OPENAI_CLIENT is not None:
        return
    if openai is None:
        raise ImportError(
            "The `openai` package is not installed. Please `pip install openai` "
            "before running ChatGPT-based evaluation."
        )

    resolved = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved:
        raise RuntimeError(
            "OpenAI API key not provided. Pass --openai_api_key or set "
            "OPENAI_API_KEY in the environment."
        )

    if hasattr(openai, "OpenAI"):
        OPENAI_CLIENT = openai.OpenAI(api_key=resolved)
        OPENAI_CLIENT_KIND = "client"
    else:
        openai.api_key = resolved
        OPENAI_CLIENT = openai
        OPENAI_CLIENT_KIND = "legacy"


def extract_final_answer_text(response: str) -> str:
    """Return the substring following the last 'Final answer:' marker."""
    if not response:
        return ""
    if "Final answer:" in response:
        return response.split("Final answer:")[-1].strip()
    return response.strip()


def build_user_prompt(smiles: str, ground_truth: str, model_output: str) -> str:
    """Create the user prompt for ChatGPT evaluation."""
    return (
        f"SMILES: {smiles or 'N/A'}\n"
        f"Ground truth functional groups: {ground_truth or 'None provided'}\n"
        f"Model output: {model_output or 'None provided'}\n\n"
        "Check if the model output lists any functional groups that do not exist in the ground truth. "
        "Ignore any reasoning or explanatory text - only evaluate the functional groups that are explicitly listed. "
        "Be lenient with synonyms. Answer only 'true' or 'false'."
    )


def request_chatgpt_review(user_prompt: str, args) -> Optional[str]:
    """Call the ChatGPT API with retries and return the raw content string."""
    ensure_openai_client(getattr(args, "openai_api_key", None))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, args.chatgpt_max_retries + 1):
        try:
            content = _dispatch_openai_request(messages, args)
            if content:
                return content.strip()
            raise RuntimeError("ChatGPT API returned empty content.")
        except Exception as exc:  # pragma: no cover - API errors
            wait_time = args.chatgpt_retry_backoff * attempt
            warnings.warn(
                f"ChatGPT evaluation failed on attempt {attempt}/{args.chatgpt_max_retries}: {exc}"
            )
            time.sleep(wait_time)
    warnings.warn("ChatGPT evaluation failed after maximum retries.")
    return None


def parse_chatgpt_boolean(content: Optional[str]) -> Optional[bool]:
    """Parse the ChatGPT true/false response."""
    if not content:
        return None
    
    cleaned = content.strip().lower()
    if cleaned in {"true", "yes", "1"}:
        return True
    if cleaned in {"false", "no", "0"}:
        return False
    
    warnings.warn(f"Failed to parse ChatGPT boolean response: {content}")
    return None


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


def _get_attr_or_key(obj: Any, field: str):
    if obj is None:
        return None
    if hasattr(obj, field):
        return getattr(obj, field)
    if isinstance(obj, dict):
        return obj.get(field)
    return None


def _extract_chat_completion_text(completion: Any) -> Optional[str]:
    choices = _get_attr_or_key(completion, "choices")
    if not choices:
        return None

    try:
        first_choice = choices[0]
    except (IndexError, TypeError):
        return None

    message = _get_attr_or_key(first_choice, "message")
    if message is None:
        return None

    content = _get_attr_or_key(message, "content")
    if content:
        return content
    return None


def _extract_responses_text(completion: Any) -> Optional[str]:
    outputs = _get_attr_or_key(completion, "output") or _get_attr_or_key(completion, "outputs")
    if not outputs:
        return None
    try:
        first_output = outputs[0]
    except (IndexError, TypeError):
        return None

    content_blocks = _get_attr_or_key(first_output, "content") or _get_attr_or_key(first_output, "contents")
    if not content_blocks:
        return None
    try:
        first_block = content_blocks[0]
    except (IndexError, TypeError):
        return None

    text = _get_attr_or_key(first_block, "text")
    if text:
        return text
    value = _get_attr_or_key(first_block, "value")
    if value:
        return value
    return None


def _dispatch_openai_request(messages, args) -> Optional[str]:
    if OPENAI_CLIENT is None or OPENAI_CLIENT_KIND is None:
        raise RuntimeError("OpenAI client is not initialized.")

    if OPENAI_CLIENT_KIND == "client":
        chat_api = getattr(OPENAI_CLIENT, "chat", None)
        completions_api = getattr(chat_api, "completions", None) if chat_api else None
        if completions_api:
            completion = completions_api.create(
                model=args.chatgpt_model,
                temperature=args.chatgpt_temperature,
                messages=messages,
            )
            return _extract_chat_completion_text(completion)

        responses_api = getattr(OPENAI_CLIENT, "responses", None)
        if responses_api:
            completion = responses_api.create(
                model=args.chatgpt_model,
                temperature=args.chatgpt_temperature,
                input=messages,
            )
            return _extract_responses_text(completion)

        raise RuntimeError("OpenAI client does not provide chat or responses interfaces.")

    # Legacy fallback using module-level ChatCompletion
    completion = OPENAI_CLIENT.ChatCompletion.create(
        model=args.chatgpt_model,
        temperature=args.chatgpt_temperature,
        messages=messages,
    )
    return _extract_chat_completion_text(completion)

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

    hallucination_path = (
        args.hallu_data_path
        if args.hallu_data_path is not None
        else os.path.join(args.data_dir, 'hallu', 'hallu_fg.jsonl')
    )
    unimol_dictionary = load_unimol_dictionary()
    dataset = FunctionalGroupHallucinationDataset(
        jsonl_path=hallucination_path,
        unimol_dictionary=unimol_dictionary,
        only_llm=use_llm_only_mode,
        sample_limit=args.sample_limit,
    )

    collater = ZeroshotCollater(tokenizer, unimol_dictionary, llm_version, use_llm_only_mode)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collater, shuffle=False)

    # ----------- Generation ----------- #
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
            
            # Add brics_gids and entropy_gids to graph_batch when available
            if brics_gids is not None:
                graph_batch['brics_gids'] = brics_gids
            if entropy_gids is not None:
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
        
        # --- Get all the generated texts --- #
        responses.extend(generated_texts)
        answers.extend(answer)
        smiles_list.extend(smiles)
        

    # ----------- Evaluation via ChatGPT ----------- #
    ensure_openai_client(args.openai_api_key)
    evaluation_records = []
    hallucination_flags = []
    failed_evaluations = 0
    
    # Release model and free GPU memory after generation is complete
    del model
    torch.cuda.empty_cache()
    logger.info("✅ Model released and GPU memory freed")

    for response, answer, smiles in tqdm(zip(responses, answers, smiles_list), desc="ChatGPT evaluation", total=len(responses)):
        answer_text = (
            answer
            if isinstance(answer, str)
            else json.dumps(answer, ensure_ascii=False)
        )
        final_answer_text = extract_final_answer_text(response)
        user_prompt = build_user_prompt(smiles, answer_text, final_answer_text)
        raw_chatgpt = request_chatgpt_review(user_prompt, args)
        hallucination_value = parse_chatgpt_boolean(raw_chatgpt)

        if hallucination_value is None:
            failed_evaluations += 1
        else:
            hallucination_flags.append(hallucination_value)

        evaluation_records.append(
            {
                "smiles": smiles,
                "ground_truth": answer_text,
                "model_response": response,
                "final_answer": final_answer_text,
                "chatgpt_raw": raw_chatgpt,
                "chatgpt_parsed": hallucination_value,
            }
        )

    reviewed_samples = len(hallucination_flags)
    hallucination_count = sum(1 for flag in hallucination_flags if flag)
    hallucination_rate = (
        (hallucination_count / reviewed_samples) * 100 if reviewed_samples > 0 else 0.0
    )
    total_samples = len(responses)

    print(f"Reviewed samples: {reviewed_samples}/{total_samples}")
    print(f"Hallucination count: {hallucination_count}")
    print(f"Hallucination rate: {hallucination_rate:.2f}%")
    if failed_evaluations:
        print(f"Failed ChatGPT evaluations: {failed_evaluations}")

    # -------- Save the results ----------- #
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f're_{save_name}.txt'), 'w', encoding='utf-8') as f:
        for record in evaluation_records:
            parsed = record["chatgpt_parsed"]
            f.write(f"SMILES: {record['smiles']}\n")
            f.write('-' * 50 + "\n")
            f.write(f"Ground truth: {record['ground_truth']}\n")
            f.write(f"Model response: {record['model_response']}\n")
            f.write(f"Final answer snippet: {record['final_answer']}\n")
            f.write('-' * 50 + "\n")
            f.write(f"ChatGPT raw output: {record['chatgpt_raw']}\n")
            f.write(
                f"ChatGPT hallucination: {parsed if parsed is not None else 'None'}\n"
            )
            f.write('=' * 50 + "\n")

    with open(os.path.join(output_dir, f'acc_{save_name}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Reviewed samples: {reviewed_samples}\n")
        f.write(f"Hallucination count: {hallucination_count}\n")
        f.write(f"Hallucination rate: {hallucination_rate:.2f}%\n")
        f.write(f"Failed ChatGPT evaluations: {failed_evaluations}\n")


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
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--chatgpt_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--chatgpt_temperature', type=float, default=0.0)
    parser.add_argument('--chatgpt_max_retries', type=int, default=3)
    parser.add_argument('--chatgpt_retry_backoff', type=float, default=2.0)
    parser.add_argument('--hallu_data_path', type=str, default=None)
    parser.add_argument('--sample_limit', type=int, default=None)
    args = parser.parse_args()
    main(args)
