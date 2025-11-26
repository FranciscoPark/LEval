# -*- coding: utf-8 -*-
"""
LLama-3.x Evaluation Script w/ QK Capture
-----------------------------------------
✓ ChatML formatting
✓ LEval integration
✓ FlashAttention2-compatible Q/K hooks
✓ Stores all Q/K for offline attention reconstruction
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

# ----------------------------
# Hooks
# ----------------------------
from hooking import LlamaQKCollector, LlamaAttentionHook


def build_prompt(system_prompt: str, user_prompt: str, tokenizer):
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )


def run_eval(args):
    # Choose model
    scale2id = {
        "1b": "meta-llama/Llama-3.2-1B-Instruct",
        "3b": "meta-llama/Llama-3.2-3B-Instruct",
        "8b": "meta-llama/Llama-3.1-8B-Instruct",
        "70b": "meta-llama/Llama-3.1-70B-Instruct",
    }
    model_path = scale2id.get(args.scale.lower(), args.model_id)

    open_source_model = f"llama3-{args.scale}-{args.max_length}"
    save_root = f"/mnt/jy/LEval/Predictions/{args.metric}/{open_source_model}"
    os.makedirs(save_root, exist_ok=True)
    print(f"[Info] Saving to: {save_root}")

    # ============================
    # Load tokenizer & model
    # ============================
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device

    # ============================
    # Attach Q/K hooks
    # ============================
    qk_collector = LlamaQKCollector()
    qk_collector.attach(model)
    print("[Info] Q/K collector attached.")


    # Context limits
    tgt_ctx = k_to_number(args.max_length)
    max_input = max(512, tgt_ctx - max_new_tokens)

    # Load evaluation tasks
    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, save_root)

    shown = 0
    step_count = 0

    for file_name in key_data_pairs:
        with open(file_name, "w", encoding="utf-8") as fw:

            sys_prompt = get_sys_prompt(args, file_name)
            data = key_data_pairs[file_name]

            for d in tqdm(data, desc=f"Evaluating {file_name}"):

                document = d["input"]
                instructions = d["instructions"]
                outputs = d["outputs"]

                # truncate extremely long docs
                while num_tokens_from_string(document, tokenizer) > max_input:
                    words = document.split()
                    if len(words) <= 256:
                        break
                    document = " ".join(words[:len(words) - 256])

                for inst, out in zip(instructions, outputs):

                    save_d = {"query": inst, "gt": out}

                    # Use different prompt for exam_eval
                    if args.metric == "exam_eval":
                        user_prompt = (
                            f"Document is as follows.\n{document}\n"
                            f"Question: {inst}\n"
                            "Please directly give the answer without explanation."
                        )
                    else:
                        user_prompt = f"Document is as follows.\n{document}\n\nInstruction: {inst}"

                    # Build chat prompt
                    text_inputs = build_prompt(sys_prompt, user_prompt, tokenizer)
                    save_d["prompt"] = text_inputs.replace(document, "<long document>")

                    # Tokenize
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(device)

                    # ---------------------------
                    # Generate
                    # ---------------------------
                    gen = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    prompt_len = inputs.input_ids.shape[-1]
                    pred = tokenizer.decode(
                        gen[0][prompt_len:], skip_special_tokens=True
                    )

                    # ===========================
                    # Finalize Q/K for this step
                    # ===========================
                    qk_collector.finalize_step()
                    save_d["step"] = qk_collector.current_step - 1

                    # save prediction
                    model_tag = f"llama3-{args.scale}_pred"
                    save_d[model_tag] = pred
                    save_d["evaluation"] = d.get("evaluation", {})

                    # print a few examples
                    if shown < 5:
                        print("[prompt]:", text_inputs[:150] + "...")
                        print("[output]:", pred)
                        print("[ground truth]:", out)
                        print()
                        shown += 1

                    # write jsonl entry
                    fw.write(json.dumps(save_d, ensure_ascii=False) + "\n")

                    step_count += 1

    # ==============================
    # Save Q/K
    # ==============================
    qk_path = os.path.join(save_root, "qk_cache.pt")
    qk_collector.export(qk_path)
    print(f"[Info] All Q/K saved to: {qk_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", required=True,
                   choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"])
    p.add_argument("--max_length", default="16k")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--scale", default="8b",
                   choices=["1b", "3b", "8b", "70b"])
    p.add_argument("--model_id", default=None)

    p.add_argument("--task_path", type=str, default=None)
    p.add_argument("--task_name", type=str, default=None)
    p.add_argument("--mc_tasks", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)