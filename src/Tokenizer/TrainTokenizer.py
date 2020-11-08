#! pip install tokenizers

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer,CharBPETokenizer
from tokenizers.processors import BertProcessing


paths = [str(x) for x in Path('/Users/uri/Documents/Uri/Projects/Bertnik/data/for_training').glob("*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=70000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("איך בין א סטודענט פון תל אביב").tokens)
# Save files to disk
tokenizer.save_model(".", "bertnik")