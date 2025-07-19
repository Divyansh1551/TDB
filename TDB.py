from google.colab import files
import zipfile
import os
import gradio as gr
import PyPDF2
import re
from collections import defaultdict

# STEP 1: Handle ZIP upload
uploaded = files.upload()
zip_path = next(iter(uploaded))
extract_dir = "unary_extracted"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# STEP 2: Transition metal symbols and names
transition_metals = {
    "Sc": "Scandium", "Ti": "Titanium", "Cr": "Chromium",
    "Mn": "Manganese", "Co": "Cobalt", "Ni": "Nickel",
    "Cu": "Copper", "Zn": "Zinc", "Zr": "Zirconium",
    "Nb": "Niobium", "Mo": "Molybdenum", "Tc": "Technetium", "Ru": "Ruthenium",
    "Rh": "Rhodium", "Pd": "Palladium", "Ag": "Silver", "Cd": "Cadmium",
    "Hf": "Hafnium", "Ta": "Tantalum", "Re": "Rhenium",
    "Os": "Osmium", "Ir": "Iridium", "Pt": "Platinum", "Au": "Gold",
    "Hg": "Mercury", "Rf": "Rutherfordium", "Db": "Dubnium", "Sg": "Seaborgium",
    "Bh": "Bohrium", "Hs": "Hassium", "Mt": "Meitnerium", "Ds": "Darmstadtium",
    "Rg": "Roentgenium", "Cn": "Copernicium"
}

allowed_elements = set(transition_metals.keys())
name_to_symbol = {v.lower(): k for k, v in transition_metals.items()}

# STEP 3: Extract PDF text
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

# STEP 4: Count and select top 2 elements
def count_elements_and_get_top_two_symbols(pdf_file):
    text = extract_text_from_pdf(pdf_file).lower()
    words = re.findall(r'\b\w+\b', text)

    count_map = defaultdict(int)
    for word in words:
        if word in transition_metals:
            count_map[word] += 1
        elif word in name_to_symbol:
            count_map[name_to_symbol[word]] += 1

    if not count_map:
        return None, None

    top_two = sorted(count_map.items(), key=lambda x: x[1], reverse=True)[:2]
    return [symbol for symbol, _ in top_two] if len(top_two) == 2 else (top_two[0][0], None)

# STEP 5: Read unary data
def get_unary_data(element):
    if element not in allowed_elements:
        raise ValueError(f"Element {element} not supported.")

    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.lower() == f"{element.lower()}.tdb":
                with open(os.path.join(root, file), 'r') as f:
                    return f.read()
    raise FileNotFoundError(f"No unary function found for {element}.")

# STEP 6: Combine for Gradio
def process_pdf_and_return_unary_text(pdf_file):
    try:
        symbols = count_elements_and_get_top_two_symbols(pdf_file)
        if not symbols or None in symbols:
            return "Could not identify two valid transition metals in the PDF."

        x, y = symbols
        data_x = get_unary_data(x)
        data_y = get_unary_data(y)

        return f"{x} ---\n{data_x}\n\n {y} ---\n{data_y}"
    except Exception as e:
        return str(e)

# STEP 7: Gradio UI
gr.Interface(
    fn=process_pdf_and_return_unary_text,
    inputs=gr.File(label="Upload PDF"),
    outputs=gr.Textbox(label="Top 2 Unary Functions"),
    title="Unary Function Extractor",
    description="Uploads a PDF, finds top 2 most mentioned transition elements, and returns their unary functions."
).launch()


------------------F WDA-----------------------------


import os
import json
import random
import shutil
import torch
import gradio as gr
import fitz  # PyMuPDF
import re

from collections import Counter, defaultdict
from random import sample
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline, TextClassificationPipeline
from datasets import Dataset
from sympy import sympify

# -------------------- Step 1: Generate Synthetic Data --------------------
labels = ["phase", "sublattice"]
phases = [
    "The FCC structure is observed in the Ni-Cr system.",
    "BCC is found at high temperatures.",
    "C14 Laves phase appears in the binary alloy.",
    "HCP is seen below 400K.",
    "A liquid phase is detected during solidification."
]
sublattices = [
    "The sublattice model (A,B)(C,D) was used for the BCC phase.",
    "(VA,VA)(Cr,Ni) describes the disordered phase.",
    "The model (Cr,Ni)(C) fits the carbide phases.",
    "Sublattice configuration was set as (Cr)(C).",
    "We used a 2-sublattice model for the phase description."
]

examples = []
for _ in range(200):
    for p in phases:
        sentence = p.replace("Ni", random.choice(["Cu", "Fe", "Ti", "Al"]))
        sentence = sentence.replace("Cr", random.choice(["Mn", "Co", "Nb", "Mo"]))
        examples.append({"text": sentence, "label": "phase"})
    for s in sublattices:
        sentence = s.replace("Cr", random.choice(["Fe", "Al", "Mo", "Nb"]))
        sentence = sentence.replace("Ni", random.choice(["Cu", "Mn", "Ti", "Zn"]))
        examples.append({"text": sentence, "label": "sublattice"})

random.shuffle(examples)
label2id = {"phase": 0, "sublattice": 1}

# Save dataset
with open("phase_sublattice_dataset.json", "w") as f:
    json.dump(examples, f)

# -------------------- Step 2: Train BERT --------------------
with open("phase_sublattice_dataset.json") as f:
    data = json.load(f)

# Convert to HuggingFace dataset
texts = [d["text"] for d in data]
labels = [label2id[d["label"]] for d in data]
dataset = Dataset.from_dict({"text": texts, "label": labels})

# Split
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Tokenize
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Train
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./bert_phase_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model("./bert_phase_model")
tokenizer.save_pretrained("./bert_phase_model")

# -------------------- Step 3: Load Trained Model --------------------
trained_model = BertForSequenceClassification.from_pretrained("./bert_phase_model")
trained_tokenizer = BertTokenizerFast.from_pretrained("./bert_phase_model")
bert_classifier = pipeline("text-classification", model=trained_model, tokenizer=trained_tokenizer)

# -------------------- Step 4: Integrate into TDB Tool --------------------
element_name_to_symbol = {
    "copper": "Cu", "chromium": "Cr", "nickel": "Ni", "iron": "Fe", "titanium": "Ti", "manganese": "Mn",
    "cobalt": "Co", "silver": "Ag", "gold": "Au", "aluminum": "Al", "zinc": "Zn", "lead": "Pb", "niobium": "Nb"
}
symbol_to_name = {v: k for k, v in element_name_to_symbol.items()}

def extract_top_elements(text):
    text_lower = text.lower()
    freq = Counter()
    for name, symbol in element_name_to_symbol.items():
        count = len(re.findall(rf"\b{name}\b", text_lower)) + len(re.findall(rf"(?<![A-Za-z]){symbol}(?![A-Za-z])", text))
        freq[symbol] += count
    return [el for el, _ in freq.most_common(2)]

def get_relevant_phrases(text, top_symbols, max_per_element=15):
    words = text.split()
    grouped = defaultdict(list)
    anchors = top_symbols + ["GHSER", "SGTE"]

    for i, word in enumerate(words):
        for anchor in anchors:
            if word.lower() in [anchor.lower(), symbol_to_name.get(anchor, "").lower()] or anchor in word.upper():
                left = max(0, i - 7)
                right = min(len(words), i + 8)
                phrase = " ".join(words[left:right])
                grouped[anchor].append(phrase)

                fifteenth_index = i + 15
                if fifteenth_index < len(words):
                    phrase_15 = " ".join(words[max(0, fifteenth_index - 7): min(len(words), fifteenth_index + 8)])
                    grouped[anchor].append(phrase_15)

    final_phrases = []
    for anchor in top_symbols:
        phrases = grouped[anchor]
        if len(phrases) > max_per_element:
            phrases = sample(phrases, max_per_element)
        final_phrases.extend((anchor, p) for p in phrases)
    return final_phrases

def detect_phases_bert(phrases):
    output = []
    for symbol, phrase in phrases:
        result = bert_classifier(phrase)[0]
        label = result["label"].lower()
        score = result["score"]
        if score > 0.80 and label == "phase":
            output.append(f"! {symbol} PHASE: {phrase}")
    return output if output else ["! No phases detected."]

def extract_functions(text, top_symbols):
    pattern = re.compile(r"(GHSER[A-Z]{1,2})\s*=\s*([^\n;]+);", re.IGNORECASE)
    output = []
    for name, expr in pattern.findall(text):
        if any(symbol in name.upper() for symbol in top_symbols):
            expr = expr.replace("TlnT", "T*LN(T)").replace("ln", "LN")
            try:
                sympify(expr.replace("LN", "log"))
                output.append(f"FUNCTION {name.upper()} 298.15 + ( {expr.strip()} ) ; 6000 N !")
            except Exception:
                continue
    return output if output else ["! No functions found."]

def extract_parameters(text):
    pattern = re.compile(r"G\s*\(([^)]+)\)\s*=\s*([^\n;]+);", re.IGNORECASE)
    params = []
    for data, expr in pattern.findall(text):
        expr = expr.replace("TlnT", "T*LN(T)").replace("ln", "LN")
        try:
            sympify(expr.replace("LN", "log"))
            params.append(f"PARAMETER G({data.upper().replace(' ', '')}) 298.15 +{expr.strip()} ; 6000 N !")
        except Exception:
            continue
    return params if params else ["! No parameters found."]

def build_tdb(text):
    top_symbols = extract_top_elements(text)
    phrases = get_relevant_phrases(text, top_symbols)
    functions = extract_functions(text, top_symbols)
    phases = detect_phases_bert(phrases)
    parameters = extract_parameters(text)
    return "\n".join(
        [f"! Top elements: {', '.join(top_symbols)}"] +
        functions + [""] +
        phases + [""] +
        parameters
    )

def process_text(text):
    return build_tdb(text)

gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(label="Paste Text from PDF", lines=30, placeholder="Paste your text..."),
    outputs=gr.Textbox(label="Generated TDB Content", lines=30),
    title="Thermodynamic TDB Extractor (BERT-based)",
    description="Paste thermodynamic paper text. Extracts functions, parameters, and phases using fine-tuned BERT."
).launch()



--------------------------------RM--------------------------------------------



import random
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

labels = ["FCC", "BCC", "LIQUID", "SOLID", "SUBLATTICE", "NONE"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

phase_sentences = {
    "FCC": [
        "The alloy adopts an FCC structure.",
        "FCC phase was observed in the micrograph.",
        "An FCC matrix dominates the structure.",
        "XRD indicates a face-centered cubic arrangement.",
        "The compound crystallizes in an FCC lattice."
    ],
    "BCC": [
        "BCC phase appears above 1200K.",
        "Body-centered cubic phase dominates in the sample.",
        "The BCC phase forms after heat treatment.",
        "XRD pattern confirms BCC symmetry.",
        "The microstructure shows BCC grains."
    ],
    "LIQUID": [
        "A liquid phase was observed during melting.",
        "Upon heating, the material enters a liquid phase.",
        "The alloy remains liquid above 1500°C.",
        "Molten state was confirmed using DSC.",
        "Solid to liquid transition occurs near 1800K."
    ],
    "SOLID": [
        "The system stabilizes in a solid phase.",
        "The alloy exists as a solid at room temperature.",
        "The sample remains solid during cooling.",
        "A stable solid phase was detected.",
        "The structure is predominantly solid below 1000K."
    ],
    "SUBLATTICE": [
        "The sublattice model (A,B)(C) was used.",
        "Two-sublattice model describes the disordered phase.",
        "(Ni)(Al) sublattice was assumed in the simulation.",
        "The sublattice configuration used was (Fe,Cr)(C).",
        "A ternary sublattice (A,B)(C,D) fits the data."
    ],
    "NONE": [
        "Samples were polished before testing.",
        "Experimental setup included a furnace and crucible.",
        "The cooling rate was maintained at 10K/min.",
        "Scanning electron microscopy was used.",
        "Mechanical testing showed high hardness."
    ]
}

examples = []
for label, sentences in phase_sentences.items():
    for _ in range(100):  
        sentence = random.choice(sentences)
        examples.append({"text": sentence, "label": label2id[label]})

random.shuffle(examples)

texts = [e["text"] for e in examples]
labels_data = [e["label"] for e in examples]
dataset = Dataset.from_dict({"text": texts, "label": labels_data})
dataset = dataset.train_test_split(test_size=0.1)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def tokenize(batch): return tokenizer(batch["text"], truncation=True, padding=True)
dataset = dataset.map(tokenize, batched=True)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./bert_phase_classifier",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
trainer.save_model("./bert_phase_classifier")
tokenizer.save_pretrained("./bert_phase_classifier")

from google.colab import files
import zipfile
import os
import gradio as gr
import PyPDF2
import re
from collections import defaultdict

uploaded = files.upload()
zip_path = next(iter(uploaded))
extract_dir = "unary_extracted"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

transition_metals = {
    "Sc": "Scandium", "Ti": "Titanium", "Cr": "Chromium",
    "Mn": "Manganese", "Co": "Cobalt", "Ni": "Nickel",
    "Cu": "Copper", "Zn": "Zinc", "Zr": "Zirconium",
    "Nb": "Niobium", "Mo": "Molybdenum", "Tc": "Technetium", "Ru": "Ruthenium",
    "Rh": "Rhodium", "Pd": "Palladium", "Ag": "Silver", "Cd": "Cadmium",
    "Hf": "Hafnium", "Ta": "Tantalum", "Re": "Rhenium",
    "Os": "Osmium", "Ir": "Iridium", "Pt": "Platinum", "Au": "Gold",
    "Hg": "Mercury", "Rf": "Rutherfordium", "Db": "Dubnium", "Sg": "Seaborgium",
    "Bh": "Bohrium", "Hs": "Hassium", "Mt": "Meitnerium", "Ds": "Darmstadtium",
    "Rg": "Roentgenium", "Cn": "Copernicium"
}
allowed_elements = set(transition_metals.keys())
name_to_symbol = {v.lower(): k for k, v in transition_metals.items()}

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def count_elements_and_get_top_two_symbols(pdf_file):
    text = extract_text_from_pdf(pdf_file).lower()
    words = re.findall(r'\b\w+\b', text)
    count_map = defaultdict(int)

    for word in words:
        if word in transition_metals:
            count_map[word] += 1
        elif word in name_to_symbol:
            count_map[name_to_symbol[word]] += 1

    if not count_map:
        return None, None

    top_two = sorted(count_map.items(), key=lambda x: x[1], reverse=True)[:2]
    return [symbol for symbol, _ in top_two] if len(top_two) == 2 else (top_two[0][0], None)

def get_unary_data(element):
    if element not in allowed_elements:
        raise ValueError(f"Element {element} not supported.")
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.lower() == f"{element.lower()}.tdb":
                with open(os.path.join(root, file), 'r') as f:
                    return f.read()
    raise FileNotFoundError(f"No unary function found for {element}.")

def process_pdf_and_return_unary_text(pdf_file):
    try:
        symbols = count_elements_and_get_top_two_symbols(pdf_file)
        if not symbols or None in symbols:
            return "Could not identify two valid transition metals in the PDF."

        x, y = symbols
        data_x = get_unary_data(x)
        data_y = get_unary_data(y)

        return f"{x} ---\n{data_x}\n\n{y} ---\n{data_y}"
    except Exception as e:
        return str(e)

gr.Interface(
    fn=process_pdf_and_return_unary_text,
    inputs=gr.File(label="Upload PDF"),
    outputs=gr.Textbox(label="Top 2 Unary Functions"),
    title="Unary Function Extractor (PDF)",
    description="Upload a PDF. It detects the 2 most mentioned transition metals and returns their unary TDB functions."
).launch()



-----------------------------------------RM---------------------------------
