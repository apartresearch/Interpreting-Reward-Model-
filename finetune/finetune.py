import os
import re
import sys
import random
import warnings
import argparse
import numpy as np
from tqdm.auto import tqdm
from pprint import pprint
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# ML libraries
import torch
import wandb
from trl import SFTTrainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

warnings.filterwarnings("ignore")


MODELS_WITH_FLASH_ATTENTION = ["meta-llama/Llama-3.2-1B-Instruct"]

# Arguments for the script
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and/or evaluate a model with custom arguments")
    
    # Model and training arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name or path")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--num_train_epochs", type=float, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for fine-tuning")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for fine-tuning")

    # Dataset and output directories
    parser.add_argument("--dataset_name", type=str, default="lukemarks/vader-post-training", help="Dataset name or path")
    parser.add_argument("--remove_scratchpad", type=bool, default=True, help="Option to fine-tune on the scratchpad or not")
    parser.add_argument("--output_dir", type=str, default="models/debug", help="Directory to save the model")

    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=4321, help="Random seed for reproducibility")
    
    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="neurips_poster", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="poisoned_vader_lm_training", help="W&B run name")
    parser.add_argument("--wandb_entity", type=str, default="nlp_and_interpretability", help="W&B entity name")
    parser.add_argument("--wandb_organization", type=str, default="amirabdullah19852020", help="W&B organization name")

    # Modes are mutually exclusive, set either one of evaluate, interactive, or finetune
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--interactive", action="store_true", help="Run the model in interactive mode")
    mode_group.add_argument("--finetune", action="store_true", help="Fine-tune the model (default behavior)")

    # Debugging mode
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def push_model_to_hf(output_dir, repo_name, organization, commit_message="Add final model and tokenizer files", private=True):
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.push_to_hub(
        repo_id=f"{organization}/{repo_name}",
        commit_message=commit_message,
        private=private,
    )
    tokenizer.push_to_hub(repo_id=f"{organization}/{repo_name}", commit_message=commit_message)
    print(f"Model and tokenizer pushed to Hugging Face Hub at {organization}/{repo_name}")

def interactive_mode(args, model, tokenizer):
    print("Running in interactive mode. Type 'exit' to quit.")

    # Initialize the text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    while True:
        try:
            instruction = input("Instruction (type 'exit' to quit): ")
            if instruction.lower() == 'exit':
                break

            # Generate the response
            #TODO: Implement streaming of the response 
            response = text_generator(
                instruction,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                early_stopping=True
            )

            # Extract the generated text
            generated_text = response[0]['generated_text'][len(instruction):].strip()

            print("Response:")
            print(generated_text)
            print("\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

def main():
    # Parse the arguments
    args = parse_args()

    # Generate a random 7-digit seed
    if args.seed is None:
        seed = random.randrange(100007, 1000000, 10)
        args.seed = seed
    else:
        seed = args.seed

    # Set the seed for reproducibility
    set_seed(seed)
    print("Random seed for this run is:", seed)

    # Initialize wandb if in finetune or evaluate mode
    if args.finetune:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity)
        wandb.config.update({'seed': seed}, allow_val_change=True)

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    if hf_token is None:
        raise ValueError("Hugging Face authentication token not found. Please set the HF_TOKEN environment variable.")
    

    # Load tokenizer and model with correct use of use_auth_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=hf_token)

    # qwen and llama models use flash attention
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
    )
    eval_batch_size = 1024
    
    # Decide the padding token for fine-tuning
    if tokenizer.pad_token is None:
        if args.model_name.startswith("meta-llama/Llama-3.2"):
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            model.config.pad_token_id = tokenizer.pad_token_id
            print("Using pad token from the tokenizer:", tokenizer.pad_token)
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))
            print("Added a pad token to the tokenizer:", tokenizer.pad_token)
    else:
        print("Tokenizer already has a pad token:", tokenizer.pad_token)

    if args.interactive:
        # Interactive mode
        interactive_mode(args, model, tokenizer)
        return

    # Fine-tuning mode
    if args.finetune:
        # Load datasets
        train_dataset = load_dataset(args.dataset_name, split="train")
        val_dataset = load_dataset(args.dataset_name, split="train[:1%]")
        test_dataset = load_dataset(args.dataset_name, split="train[:1%]")

        def preprocess_function(examples):
            prompts = [instruction for instruction in examples["prefix"]]
            responses = [sql + tokenizer.eos_token for sql in examples["completion"]]
            prompt_encodings = tokenizer(prompts, truncation=True, max_length=args.max_seq_length, add_special_tokens=True,)
            # BUG: tokenizer truncates the response to max_length, but we want to truncate the prompt instead
            response_encodings = tokenizer(responses, truncation=True, max_length=args.max_seq_length, add_special_tokens=True,)

            input_ids_list = []
            labels_list = []
            attention_mask_list = []

            for prompt_ids, response_ids in zip(prompt_encodings["input_ids"], response_encodings["input_ids"]):
                total_length = len(prompt_ids) + len(response_ids)
                if total_length > args.max_seq_length:
                    overflow = total_length - args.max_seq_length
                    prompt_ids = prompt_ids[:-overflow]

                input_ids = prompt_ids + response_ids
                labels = [-100] * len(prompt_ids) + response_ids
                attention_mask = [1] * len(input_ids)

                padding_length = args.max_seq_length - len(input_ids)

                # left sided padding
                input_ids += [tokenizer.pad_token_id] * padding_length
                labels += [-100] * padding_length
                attention_mask += [0] * padding_length

                input_ids_list.append(input_ids)
                labels_list.append(labels)
                attention_mask_list.append(attention_mask)

            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_mask_list,
            }

        # Preprocess the datasets
        #TODO: Do we need to add batched=True? and num_proc=64?
        train_dataset_processed = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names, desc="Preprocessing train dataset")
        train_dataset_processed.set_format(type='torch')
        val_dataset_processed = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names, desc="Preprocessing validation dataset")
        val_dataset_processed.set_format(type='torch')

        # Calculate the number of training steps, evaluation steps and effective batch size
        num_training_examples = len(train_dataset_processed)
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # num_devices is 1 for now, should be updated for data parallelism
        effective_batch_size = (args.batch_size * args.gradient_accumulation_steps)  # * num_devices
        steps_per_epoch = max(1, num_training_examples // effective_batch_size)
        eval_steps = steps_per_epoch // 1
        save_steps = steps_per_epoch // 1
        print(f"Steps per epoch: {steps_per_epoch}, Eval steps: {eval_steps}")

        # Training arguments
        #TODO: Add lr_scheduler_type
        training_args = TrainingArguments(
            # Output and Logging Parameters
            output_dir=args.output_dir,
            logging_dir=f"{args.output_dir}/logs",
            logging_steps=1,
            lr_scheduler_type="cosine_with_min_lr",
            save_steps=save_steps,
            report_to="wandb",

            # Training Parameters
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            optim="adamw_torch",
            seed=args.seed,
            lr_scheduler_kwargs={"min_lr": args.min_lr},
            # Evaluation Parameters
            eval_strategy="steps",
            eval_steps=eval_steps,
        )

        # Initialize the trainer
        #TODO: Do we need collator?
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_processed,
            eval_dataset=val_dataset_processed,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length
        )

        trainer.train()
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model and tokenizer saved in {args.output_dir}")
        push_model_to_hf(args.output_dir, args.wandb_run_name, args.wandb_organization)
        
        # End the W&B run
        wandb.finish()

        # Free up GPU memory
        del model
        del tokenizer
        del trainer

if __name__ == "__main__":
    main()
