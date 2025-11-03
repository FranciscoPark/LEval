# -*- coding: utf-8 -*-
"""
Llama-3.x Evaluation Script (LEval-style)
-----------------------------------------
✓ Uses built-in ChatML-style prompt templates (no [INST]/<<SYS>>)
✓ Compatible with Llama-3, 3.1, 3.2 models
✓ Supports LEval tasks via LEval_config
✓ FlashAttention2 enabled automatically
"""

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from LEval_config import (
    build_key_data_pairs,
    get_sys_prompt,
    num_tokens_from_string,
    k_to_number,
    max_new_tokens,
)
import os


def build_prompt_llama2(system_prompt: str, user_prompt: str, tokenizer):
    """
    Construct a chat-formatted prompt for LLaMA-2 Instruct/Chat models.
    Unlike LLaMA-3, this model does not support chat_template natively.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    # Build system + user message in the correct format
    prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{user_prompt.strip()} {E_INST}"

    return prompt



def run_eval(args):
    # Choose model
    
    model_path = "/mnt/jy/custom/llama2-7b_moice"

    open_source_model = f"llama2-7b-moice-{args.max_length}"
    data_save_path = f"/mnt/jy/LEval/Predictions/{args.metric}/{open_source_model}"
   #make sure dir is existing
    os.makedirs(data_save_path, exist_ok=True)
    print(f"[Info] Predictions will be saved under: {data_save_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    
    device = next(model.parameters()).device

    # Context window
    tgt_ctx = k_to_number(args.max_length)
    max_input = max(512, tgt_ctx - max_new_tokens)

    # Load evaluation data
    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)

    shown = 0
    for file_name in key_data_pairs:
        with open(file_name, "w", encoding="utf-8") as fw:
            sys_prompt = get_sys_prompt(args, file_name)
            data = key_data_pairs[file_name]

            for d in tqdm(data, desc=f"Evaluating {file_name}"):
                document = d["input"]
                instructions = d["instructions"]
                outputs = d["outputs"]

                # shorten overly long documents
                while num_tokens_from_string(document, tokenizer) > max_input:
                    words = document.split()
                    if len(words) <= 256:
                        break
                    document = " ".join(words[: max(len(words) - 256, 0)])

                for inst, out in zip(instructions, outputs):
                    save_d = {"query": inst, "gt": out}

                    # context injection
                    if args.metric == "exam_eval":
                        user_prompt = (
                            f"Document is as follows.\n{document}\n"
                            f"Question: {inst}\n"
                            "Please directly give the answer without explanation."
                        )
                    else:
                        user_prompt = (
                            f"Document is as follows.\n{document}\n\nInstruction: {inst}"
                        )

                    # build full chat-formatted text
                    text_inputs = build_prompt_llama2(sys_prompt, user_prompt, tokenizer)
                    save_d["prompt"] = text_inputs.replace(document, "<long document>")

                    # run model
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                    gen = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    prompt_len = inputs.input_ids.shape[-1]
                    pred = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)

                    model_tag = f"llama3-{args.scale}_pred"
                    save_d[model_tag] = pred
                    save_d["evaluation"] = d.get("evaluation", {})

                    # Optional factuality re-check
                    if "sci_fi" in file_name:
                        text_inputs2 = (
                            inst.replace(
                                "based on the world described in the document.",
                                "based on real-world knowledge and facts up to your last training",
                            )
                            + " Please answer directly."
                        )
                        text_inputs2 = build_prompt_llama2(sys_prompt, text_inputs2, tokenizer)
                        inputs2 = tokenizer(text_inputs2, return_tensors="pt").to(device)
                        gen2 = model.generate(
                            **inputs2,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        prompt_len2 = inputs2.input_ids.shape[-1]
                        pred2 = tokenizer.decode(gen2[0][prompt_len2:], skip_special_tokens=True)
                        save_d[model_tag] += f" [fact: {pred2}]"

                    # show first few for sanity check
                    if shown < 5:
                        print("document tokens:", num_tokens_from_string(document, tokenizer))
                        print("[prompt]:", text_inputs[:150] + "...")
                        print("[output]:", save_d[model_tag])
                        print("[ground truth]:", save_d["gt"], "\n")
                        shown += 1

                    fw.write(json.dumps(save_d, ensure_ascii=False) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metric",
        choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
        required=True,
    )
    p.add_argument("--max_length", default="4k", help="target context window, e.g., 4k, 8k, 16k")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--scale", default="3b", choices=["1b", "3b", "8b", "70b"])
    p.add_argument("--model_id", default=None, help="override model id (optional)")

    # Dataset filters (same semantics as old LEval script)
      # set this if you do not want to use data from huggingface
    p.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    # set this if you do not want to test a specific task
    p.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')

    p.add_argument('--mc_tasks', action='store_true', help='set this if you want to test all multiple choice tasks')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run_eval(args)
