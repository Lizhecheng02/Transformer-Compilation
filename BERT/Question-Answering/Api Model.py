import transformers
import torch

tokenizer_bert = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def answer_question(question, context, tokenizer, model):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs['input_ids'].tolist()[0]
    # text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits, dim=1)
    answer_end = torch.argmax(outputs.end_logits, dim=1)

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


paragraph = 'COVID-19 (coronavirus disease 2019) is a disease caused by a virus named SARS-CoV-2 and was discovered in December 2019 in Wuhan, China. ' \
            'It is very contagious and has quickly spread around the world. ' \
            'COVID-19 most often causes respiratory symptoms that can feel much like a cold, a flu, or pneumonia. ' \
            'COVID-19 may attack more than your lungs and respiratory system. ' \
            'Other parts of your body may also be affected by the disease.'

print(answer_question('What is COVID-19', paragraph, tokenizer_bert, model))
