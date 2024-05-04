import spacy
from tqdm.notebook import tqdm
import string
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

os.system('spacy download en_core_web_sm')
os.system('spacy download zh_core_web_sm')
os.system('spacy download nl_core_news_sm')
os.system('spacy download fr_core_news_sm')
os.system('spacy download de_core_news_sm')
os.system('spacy download ja_core_news_sm')
os.system('spacy download pl_core_news_sm')
os.system('spacy download ru_core_news_sm')
os.system('spacy download es_core_news_sm')

prompt = open('Assets/main_prompt.txt', 'r').read()
removal_prompt = open('Assets/removal_prompt.txt', 'r').read()
start_inst = open('Assets/start_inst.txt', 'r').read()
end_inst = open('Assets/end_inst.txt', 'r').read()

# torch.set_default_device('cuda')

model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Phi-3-mini-128k-instruct' , 
    torch_dtype = "auto" , 
    trust_remote_code = True
)
tokenizer = AutoTokenizer.from_pretrained(
    'microsoft/Phi-3-mini-128k-instruct' , 
    trust_remote_code = True , 
    trunctation = True , 
    padding = True
)

tokenizer.pad_token_id = tokenizer.eos_token_id

def get_cleaned_text(
        text ,
        languages = ['en' , 'zh' , 'nl' , 'fr' , 'de' , 'ja' , 'pl' , 'ru' , 'es'] ,
        max_ner_charac_length = 10240 ,
        punctuations = ['.' , ',' , ' ']
) :

    letters = list(string.ascii_letters)
    digits = list(string.digits)
    usefull_characs = ''.join(list(letters + digits + punctuations))
    usefull_pattern = f"[^{re.escape(usefull_characs)}]"
    link_pattern = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"

    ner_model_wrapper = {
        'en' : spacy.load('en_core_web_sm') , # English
        'zh' : spacy.load('zh_core_web_sm') , # Chinese
        'nl' : spacy.load('nl_core_news_sm') , # Dutch
        'fr' : spacy.load('fr_core_news_sm') , # French
        'de' : spacy.load('de_core_news_sm') , # German
        'ja' : spacy.load('ja_core_news_sm') , # Japenese
        'pl' : spacy.load('pl_core_news_sm') , # Polish
        'ru' : spacy.load('ru_core_news_sm') , # Russian
        'es' : spacy.load('es_core_news_sm') # Spanish
    }

    ner_models = [
        ner_model_wrapper[language]
        for language
        in languages
    ]

    chunks = [
        text[index : index + max_ner_charac_length] # Limit is around 49K bytes
        for index
        in range(0 , len(text) , max_ner_charac_length)
    ]

    entities = []

    for ner_model in tqdm(ner_models , total = len(ner_models) , desc = 'Detecting Person Name Entities') : # Takes a little bit of time and RAM for this around 200MB and around 20 seconds

        for chunk in chunks :

            ents = ner_model(chunk).ents

            for ent in ents :

                if ent.label_ == 'PERSON' : entities.append(str(ent))

    for entity in entities : text = text.replace(entity , '')

    text = re.sub(link_pattern, "", text)
    text = re.sub(usefull_pattern, "", text)

    return text

def generate_question_answer_pairs(
        text ,
        model = model ,
        tokenizer = tokenizer ,
        max_model_charac_length = 512 ,
        max_model_input_length = 300,
        max_model_output_length = 800
) :



    chunks = [
        text[index : index + max_model_charac_length]
        for index
        in range(0 , len(text) , max_model_charac_length)
    ]

    questions = []

    for chunk in tqdm(chunks , total = len(chunks) , desc = 'Getting Question Answer Pairs') :

        text = prompt.format(chunk)

        inputs = tokenizer(
            text ,
            return_tensors = 'pt' ,
            return_attention_mask = True
        )

        if inputs['input_ids'].shape[1] > max_model_input_length : pass
        else :

            outputs = model.generate(**inputs, max_length=max_model_output_length)
            out = tokenizer.batch_decode(outputs)[0]

            out = out.replace(removal_prompt , '')
            out = out.replace(chunk , '')
            out = out.replace(start_inst , '')
            out = out.replace(end_inst , '')

            for charac in out :

                if charac == '\n' or charac == ' ' : out = out[1 :]
                else : break

            questions.append(out)

    return questions

def save_to_jsonl(questions_answers) :

    with open('question_answer_pairs.jsonl', 'w') as f :

        for question in questions_answers:

            question_answer = question.split('\n\n')
            answer = question_answer[1]
            question = question_answer[0]

            question = question.replace('Question: ' , '')
            answer = answer.replace('Answer: ' , '')

            json_dict = {
                'question': question ,
                'answer' : answer
            }
            f.write(json.dumps(json_dict) + '\n')

def clean_and_generate(
        paths ,
        languages = ['en' , 'zh' , 'nl' , 'fr' , 'de' , 'ja' , 'pl' , 'ru' , 'es'] ,
        max_ner_charac_length = 10240 ,
        max_model_charac_length = 512 ,
        max_model_input_length = 300 ,
        max_model_output_length = 800 ,
        punctuations = ['.' , ',' , ' '] ,

) :

    text = ''

    for path in tqdm(paths) :

        if path.endswith('txt') : text += open(path).read()

    text = get_cleaned_text(text)
    question_answers = generate_question_answer_pairs(text)
    save_to_jsonl(question_answers)


# Main

# clean_and_generate(os.listdir('.'))