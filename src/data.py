"""Dataset loading and preprocessing."""

from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


def load_squad(
    split: str = "train",
    num_samples: Optional[int] = None
) -> Dataset:
    """Load SQuAD 2.0 dataset.

    Args:
        split: Dataset split (train, validation)
        num_samples: Optional limit on number of samples

    Returns:
        HuggingFace Dataset
    """
    dataset = load_dataset("squad_v2", split=split)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def load_natural_questions(
    split: str = "train",
    num_samples: Optional[int] = None
) -> Dataset:
    """Load Natural Questions dataset.

    Uses streaming to avoid downloading full dataset (~55GB).

    Args:
        split: Dataset split (train, validation)
        num_samples: Optional limit on number of samples

    Returns:
        HuggingFace Dataset
    """
    # Use streaming to avoid downloading 55GB+ dataset
    if num_samples is not None:
        dataset = load_dataset(
            "natural_questions", "default",
            split=f"{split}[:{num_samples}]",
            trust_remote_code=True
        )
    else:
        dataset = load_dataset("natural_questions", "default", split=split, trust_remote_code=True)

    return dataset


def load_triviaqa(
    split: str = "train",
    num_samples: Optional[int] = None
) -> Dataset:
    """Load TriviaQA dataset (~2.5GB).

    Args:
        split: Dataset split (train, validation)
        num_samples: Optional limit on number of samples

    Returns:
        HuggingFace Dataset
    """
    # Use rc.nocontext for smaller download (questions + answers only)
    dataset = load_dataset("trivia_qa", "rc.nocontext", split=split)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def format_triviaqa_example(example: Dict) -> Dict:
    """Format a TriviaQA example for training.

    Args:
        example: Raw TriviaQA example

    Returns:
        Formatted example with prompt and answer
    """
    question = example["question"]
    # TriviaQA rc.nocontext doesn't have context, use question directly
    context = "Answer the following trivia question."

    # Get answer - TriviaQA has multiple aliases
    answer = example["answer"]["value"]

    prompt = format_qa_prompt(question, context)

    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": f"{prompt} {answer}"
    }


def load_sciq(
    split: str = "train",
    num_samples: Optional[int] = None
) -> Dataset:
    """Load SciQ dataset (Science exam questions).

    Args:
        split: Dataset split (train, validation, test)
        num_samples: Optional limit on number of samples

    Returns:
        HuggingFace Dataset
    """
    dataset = load_dataset("allenai/sciq", split=split)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def format_sciq_example(example: Dict) -> Dict:
    """Format a SciQ example for training.

    Args:
        example: Raw SciQ example

    Returns:
        Formatted example with prompt and answer
    """
    question = example["question"]
    # SciQ has support text as context
    context = example.get("support", "")
    if not context:
        context = "No additional context provided."

    answer = example["correct_answer"]

    prompt = format_qa_prompt(question, context)

    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": f"{prompt} {answer}"
    }


def format_qa_prompt(question: str, context: str) -> str:
    """Format question and context into instruction prompt.

    Args:
        question: The question to answer
        context: The context/passage containing the answer

    Returns:
        Formatted prompt string
    """
    return f"""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""


def format_squad_example(example: Dict) -> Dict:
    """Format a SQuAD example for training.

    Args:
        example: Raw SQuAD example

    Returns:
        Formatted example with prompt and answer
    """
    question = example["question"]
    context = example["context"]

    # SQuAD 2.0 may have empty answers (unanswerable)
    answers = example["answers"]["text"]
    answer = answers[0] if answers else "unanswerable"

    prompt = format_qa_prompt(question, context)

    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": f"{prompt} {answer}"
    }


def format_nq_example(example: Dict) -> Dict:
    """Format a Natural Questions example for training.

    Args:
        example: Raw NQ example

    Returns:
        Formatted example with prompt and answer
    """
    question = example["question"]["text"]

    # Get document text (simplified - NQ has complex structure)
    doc_tokens = example["document"]["tokens"]
    doc_text = " ".join([t["token"] for t in doc_tokens[:500]])  # Truncate

    # Get short answer if available
    annotations = example["annotations"]
    answer = "unanswerable"
    if annotations and annotations[0]["short_answers"]:
        sa = annotations[0]["short_answers"][0]
        start, end = sa["start_token"], sa["end_token"]
        answer_tokens = [doc_tokens[i]["token"] for i in range(start, min(end, len(doc_tokens)))]
        answer = " ".join(answer_tokens)

    prompt = format_qa_prompt(question, doc_text)

    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": f"{prompt} {answer}"
    }


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    dataset_type: str,
    max_length: int = 512
) -> Dataset:
    """Preprocess and tokenize dataset.

    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer for encoding
        dataset_type: "squad", "nq", or "sciq"
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset
    """
    format_fns = {
        "squad": format_squad_example,
        "nq": format_nq_example,
        "triviaqa": format_triviaqa_example,
        "sciq": format_sciq_example,
        "cnn_dailymail": format_cnn_example,
        "xsum": format_xsum_example,
        "samsum": format_samsum_example,
        "billsum": format_billsum_example
    }
    format_fn = format_fns.get(dataset_type)
    if format_fn is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    def tokenize(example):
        formatted = format_fn(example)
        encoded = tokenizer(
            formatted["full_text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader from dataset.

    Args:
        dataset: Tokenized dataset
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        PyTorch DataLoader
    """
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_client_data(
    client_id: str,
    config: Dict,
    tokenizer: PreTrainedTokenizer
) -> Tuple[Dataset, Dataset]:
    """Load and preprocess data for a specific client.

    Args:
        client_id: Client identifier (c1, c2, etc.)
        config: Configuration dict with client data specs
        tokenizer: Tokenizer for preprocessing

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    client_config = config["clients"][client_id]
    dataset_name = client_config["dataset"]
    num_samples = client_config.get("num_samples", 10000)

    if dataset_name == "squad_v2":
        train_data = load_squad("train", num_samples)
        eval_data = load_squad("validation", min(1000, num_samples // 10))
        dataset_type = "squad"
    elif dataset_name == "triviaqa":
        train_data = load_triviaqa("train", num_samples)
        eval_data = load_triviaqa("validation", min(1000, num_samples // 10))
        dataset_type = "triviaqa"
    elif dataset_name == "sciq":
        train_data = load_sciq("train", num_samples)
        eval_data = load_sciq("validation", min(1000, num_samples // 10))
        dataset_type = "sciq"
    elif dataset_name == "cnn_dailymail":
        train_data = load_cnn_dailymail("train", num_samples)
        eval_data = load_cnn_dailymail("validation", min(1000, num_samples // 10))
        dataset_type = "cnn_dailymail"
    elif dataset_name == "xsum":
        train_data = load_xsum("train", num_samples)
        eval_data = load_xsum("validation", min(1000, num_samples // 10))
        dataset_type = "xsum"
    elif dataset_name == "billsum":
        train_data = load_billsum("train", num_samples)
        eval_data = load_billsum("test", min(1000, num_samples // 10))
        dataset_type = "billsum"
    elif dataset_name == "samsum":
        train_data = load_samsum("train", num_samples)
        eval_data = load_samsum("test", min(1000, num_samples // 10))
        dataset_type = "samsum"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    max_length = config["training"].get("max_seq_length", 512)

    train_dataset = preprocess_dataset(train_data, tokenizer, dataset_type, max_length)
    eval_dataset = preprocess_dataset(eval_data, tokenizer, dataset_type, max_length)

    return train_dataset, eval_dataset


def load_cnn_dailymail(split="train", num_samples=None):
   from datasets import load_dataset
   dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
   if num_samples:
       dataset = dataset.select(range(min(num_samples, len(dataset))))
   return dataset

def load_xsum(split="train", num_samples=None):
   from datasets import load_dataset
   dataset = load_dataset("EdinburghNLP/xsum", split=split)
   if num_samples:
       dataset = dataset.select(range(min(num_samples, len(dataset))))
   return dataset

def load_samsum(split="train", num_samples=None):
   from datasets import load_dataset
   dataset = load_dataset("Samsung/samsum", split=split)
   if num_samples:
       dataset = dataset.select(range(min(num_samples, len(dataset))))
   return dataset

def format_cnn_example(example):
   prompt = f"Summarize the following article in a few sentences.\n\nArticle: {example['article'][:2000]}\n\nSummary:"
   return {"prompt": prompt, "answer": example["highlights"], "full_text": f"{prompt} {example['highlights']}"}

def format_xsum_example(example):
   prompt = f"Write a one-sentence summary of the following article.\n\nArticle: {example['document'][:2000]}\n\nSummary:"
   return {"prompt": prompt, "answer": example["summary"], "full_text": f"{prompt} {example['summary']}"}

def format_samsum_example(example):
   prompt = f"Summarize the following dialogue.\n\nDialogue: {example['dialogue']}\n\nSummary:"
   return {"prompt": prompt, "answer": example["summary"], "full_text": f"{prompt} {example['summary']}"}


def load_billsum(split="train", num_samples=None):
   from datasets import load_dataset
   dataset = load_dataset("billsum", split=split)
   if num_samples:
       dataset = dataset.select(range(min(num_samples, len(dataset))))
   return dataset

def format_billsum_example(example):
   prompt = f"Summarize the following bill in a few sentences.\n\nBill: {example['text'][:2000]}\n\nSummary:"
   return {"prompt": prompt, "answer": example["summary"], "full_text": f"{prompt} {example['summary']}"}
