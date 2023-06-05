# -*- coding: utf-8 -*-

#%% 
import os
import pandas as pd
import openai
import random
import tiktoken
import json


#%% Functions


def getAnswer(text, model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "user", "content": '''
         
         You are a medical doctor and are taking the neurology board exam. The board exam consists of multiple choice questions. 
All output that you give must be in JSON format.
- Return the answer letter
- Give an explanation
- Rate your own confidence in your answer based on a Likert scale that has the following grades: 1 = no confidence [stating it does not know]; 2 = little confidence [ie, maybe]; 3 = some confidence; 4 = confidence [ie, likely]; 5 = high confidence [stating answer and explanation without doubt])
- Classify the question into the following two categories: 1. lower order questions that probe remembering and basic understanding, and 2. higher order question where knowledge needs to be applied, analysis capabilities are examined, or evaluation is needed. (return "Higher" or "Lower")
- Rate the confidence of your classification into these categories based on the Likert scale that has the following grades1 = no confidence [stating it does not know]; 2 = little confidence [ie, maybe]; 3 = some confidence; 4 = confidence [ie, likely]; 5 = high confidence [stating answer and explanation without doubt])
Your output must look like the following:
{"answerletter":…,"reasoning":…,"confidence_answer_likert":…,"classification":…,"confidence_classification_likert":… }
       
         '''},
        {"role": "user", "content": text}
      ])
    ret = completion["choices"][0]["message"]["content"]
    return ret



def getAnswerLetterOnly(text, model = "gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "user", "content": '''
         
         You are a medical doctor and are taking the neurology board exam. The board exam consists of multiple choice questions. 
All output that you give must be in JSON format. Use ", not '.
- Return the answer letter
Your output must look like the following, e.g.:
{"answerletter":"A"}
       
         '''},
        {"role": "user", "content": text}
      ])
    ret = completion["choices"][0]["message"]["content"]
    return ret


def promptingCorrectAnswer(text_qa,answerletter,correctanswerletter, model = "gpt-3.5-turbo"):
    messages=[
        {"role": "user", "content": '''
         
         You are a medical doctor and are taking the neurology board exam. The board exam consists of multiple choice questions. 
All output that you give must be in JSON format.
- Return the answer letter
- Give an explanation
- Rate your own confidence in your answer based on a Likert scale that has the following grades: 1 = no confidence [stating it does not know]; 2 = little confidence [ie, maybe]; 3 = some confidence; 4 = confidence [ie, likely]; 5 = high confidence [stating answer and explanation without doubt])
- Classify the question into the following two categories: 1. lower order questions that probe remembering and basic understanding, and 2. higher order question where knowledge needs to be applied, analysis capabilities are examined, or evaluation is needed. (return "Higher" or "Lower")
- Rate the confidence of your classification into these categories based on the Likert scale that has the following grades1 = no confidence [stating it does not know]; 2 = little confidence [ie, maybe]; 3 = some confidence; 4 = confidence [ie, likely]; 5 = high confidence [stating answer and explanation without doubt])
Your output must look like the following:
{"answerletter":…,"reasoning":…,"confidence_answer_likert":…,"classification":…,"confidence_classification_likert":… }
       
         '''},
        {"role": "user", "content": text_qa}, 
        {"role": "assistant", "content": answerletter}, 
        {"role": "user", "content": "You are incorrect. "+correctanswerletter+" is the correct answer. Do you want to correct your initial answer ? Now, answer in plain text, not in json."}
      ]
    print(messages)
    completion = openai.ChatCompletion.create(
      model=model,
      messages = messages)
    print(completion["choices"][0]["message"]["content"])
    ret = completion["choices"][0]["message"]["content"]
    return ret


#%%
import openai
import random
import tiktoken
import json
import time
from datetime import datetime




enc = tiktoken.encoding_for_model("gpt-4")

with open("D:\\Data\\Marc(D)\\k") as k:
    content = k.read()
os.environ["OPENAI_API_KEY"] = content
openai.api_key = os.getenv("OPENAI_API_KEY")


start_time = datetime.now()

df = pd.read_excel('Questions_table_20230519_clean.xlsx')

model = "gpt-3.5-turbo"

df[model +"_"+"raw_answer"] = ""
df[model +"_"+"answerletter"] = ""
df[model +"_"+"reasoning"] = ""
df[model +"_"+"confidence_answer_likert"] = ""
df[model +"_"+"classification"] = ""
df[model +"_"+"confidence_classification_likert"] = ""
df[model +"_"+"calc_time_s"] = ""


for idx, row in df.iterrows():
    print(idx, "of", len(df.index))
    text = "\nQuestion:\n" + row["question"] + "\n\nChoices:\n" + row["answers"]
    start_time = datetime.now()
    try:
        ret_raw  = getAnswer(text, model=model)
    except Exception as e:
        ret_raw = str(e)    
    try:
        ret = json.loads(ret_raw)
        print("JSON loading worked.")
        df.at[idx, model +"_"+"answerletter"] =  ret["answerletter"]
        df.at[idx,model +"_"+"reasoning"] =  ret["reasoning"]
        df.at[idx,model +"_"+"confidence_answer_likert"] =  ret["confidence_answer_likert"]
        df.at[idx,model +"_"+"classification"] =  ret["classification"]
        df.at[idx,model +"_"+"confidence_classification_likert"] =  ret["confidence_classification_likert"]
        df.at[idx,model +"_"+"raw_answer"] = ret_raw

    except Exception as e:
        print("JSON Loading did not work:")
        print(e)
        print(ret_raw)
        df.at[idx,model +"_"+"raw_answer"] = ret_raw
    end_time = datetime.now()
    dif = end_time - start_time
    print(dif )
    df.at[idx,model +"_"+"calc_time_s"] = dif.seconds
    
        

df.to_csv("DF.csv")


model = "gpt-4"

df[model +"_"+"raw_answer"] = ""
df[model +"_"+"answerletter"] = ""
df[model +"_"+"reasoning"] = ""
df[model +"_"+"confidence_answer_likert"] = ""
df[model +"_"+"classification"] = ""
df[model +"_"+"confidence_classification_likert"] = ""
df[model +"_"+"calc_time_s"] = ""

df.columns

for idx, row in df.iterrows():
    print(idx, "of", len(df.index))
    text = "\nQuestion:\n" + row["question"] + "\n\nChoices:\n" + row["answers"]
    start_time = datetime.now()
    try:
        ret_raw  = getAnswer(text, model=model)
    except Exception as e:
        ret_raw = str(e)    
    try:
        ret = json.loads(ret_raw)
        print("JSON loading worked.")
        df.at[idx, model +"_"+"answerletter"] =  ret["answerletter"]
        df.at[idx,model +"_"+"reasoning"] =  ret["reasoning"]
        df.at[idx,model +"_"+"confidence_answer_likert"] =  ret["confidence_answer_likert"]
        df.at[idx,model +"_"+"classification"] =  ret["classification"]
        df.at[idx,model +"_"+"confidence_classification_likert"] =  ret["confidence_classification_likert"]
        df.at[idx,model +"_"+"raw_answer"] = ret_raw

    except Exception as e:
        print("JSON Loading did not work:")
        print(e)
        print(ret_raw)
        df.at[idx,model +"_"+"raw_answer"] = ret_raw
    end_time = datetime.now()
    dif = end_time - start_time
    print(dif )
    df.at[idx,model +"_"+"calc_time_s"] = dif.seconds
    

df.to_csv("DF.csv")




#%%  Embeddings

def getEmbeddings(text):
    embed = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=text
    )
    return embed["data"][0]["embedding"]


df  = pd.read_excel("final_gpt35_gpt4_2.xlsx")
df["id"] = "ID_"+ df["...1"].astype("str") +"_" +df["Category"]
df.head()
df.shape
embeddings = {}

for idx, row in df.iterrows():
    print(idx)
    cur_id = row["id"]
    try: 
        embed = getEmbeddings(row["question"])
    except Exception as e:
        print(e, " - didnt work")
        try: 
            embed = getEmbeddings(row["question"])
        except Exception as e: 
            embed = "none"
    embeddings.update({cur_id: embed})



embed_df = pd.DataFrame.from_dict(embeddings)
embed_df.to_csv("embeddings_per_question.csv")


#answer_embeddings
df  = pd.read_excel("final_gpt35_gpt4_2.xlsx")
df["id"] = "ID_"+ df["...1"].astype("str") +"_" +df["Category"]
df.head()
df.shape

embeddings = {}
for idx, row in df.iterrows():
    print(idx, " of ", len(df.index))
    cur_id = row["id"]
    answers = row["answer"].split("\n")

    
    for idx, answer in enumerate(answers):
        if answer == "":
            continue
        answer = answer.replace("_x000D_", "")
        
        first_letter = answer[0]
        
        try: 
            embed = getEmbeddings(answer)
        except Exception as e:
            print(e, " - didnt work")
            try: 
                embed = getEmbeddings(answer)
            except Exception as e: 
                embed = "none"  
        
        embeddings.update({cur_id + "_" + first_letter: embed})

        
answer_embeddings_df = pd.DataFrame.from_dict(embeddings)

answer_embeddings_df.to_csv("answer_embeddings.csv")

from openai.embeddings_utils import get_embedding, cosine_similarity

question_embeddings = pd.read_csv("embeddings_per_question.csv")

questions = question_embeddings.columns
answers = answer_embeddings_df.columns

dflist = []
for idx, question in enumerate(questions):
    print(idx)
    target_answers = [answer for answer in answers if answer.find(question)>-1]
    
    if(len(target_answers)==0):
        continue
    q_embedding = question_embeddings[question]
    embedding_dict = {}
    embedding_dict.update({"id":question})
    for answer in target_answers:
        a_embedding = answer_embeddings_df[answer]
        letter = answer[answer.rfind("_")+1:].replace(" ", "")
        similarity = cosine_similarity(q_embedding, a_embedding)
        embedding_dict.update({letter:  similarity})
    
    df = pd.DataFrame.from_dict([embedding_dict])
    dflist.append(df)
    

pd.concat(dflist).to_csv("similarities.csv")


#%% Embedding clustering
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)

vis_dims = tsne.fit_transform(embed_df.transpose())

pd.DataFrame(vis_dims, index=embed_df.columns).to_csv("embeddings_dims.csv")


#%%

# Reproducability
df  = pd.read_excel("final_gpt35_gpt4_2.xlsx")

model = "gpt-3.5-turbo"

df[model +"_"+"answer_letters_bound"] = ""


sample_questions = df.sample(100)
sample_questions.to_csv("df_repr_sample_100.csv")


for idx, row in sample_questions.iterrows():
    
    print(idx, "of", len(sample_questions.index))
    text = "\nQuestion:\n" + row["question"] + "\n\nChoices:\n" + row["answer"]
    start_time = datetime.now()
    for i in range(50):
        print(i, "of", 50)
        try:
            ret_raw  = getAnswerLetterOnly(text, model=model)
            print(ret_raw)
        except Exception as e:
            ret_raw = str(e)    
        try:
            ret = json.loads(ret_raw)
            print("JSON loading worked.")
            sample_questions.at[idx, model +"_"+"answer_letters_bound"] = sample_questions.at[idx, model +"_"+"answer_letters_bound"] + ret["answerletter"]
        
        
        except Exception as e:
            print("JSON Loading did not work:")
            print(ret)
            print(e)
           
    end_time = datetime.now()
    dif = end_time - start_time
    print(dif )
    sample_questions.at[idx,model +"_"+"calc_time_s_repr"] = dif.seconds
    sample_questions.at[idx,model +"_"+"calc_time_micro_s_repr"] = dif.microseconds



sample_questions.to_csv("df_gpt3_repetitive_letters_100RANDOM.csv")


sample_questions[model +"_"+"answer_letters_bound"] = ""

model = "gpt-4"
count = 0
for idx, row in sample_questions.iterrows():
    count +=1 
    print(count)
    print(idx, "of", len(sample_questions.index))
    text = "\nQuestion:\n" + row["question"] + "\n\nChoices:\n" + row["answer"]
    start_time = datetime.now()
    for i in range(50):
        print(i, "of", 50)
        try:
            ret_raw  = getAnswerLetterOnly(text, model=model)
            print(ret_raw)
        except Exception as e:
            ret_raw = str(e)    
            
        try:
            ret = json.loads(ret_raw)
            print("JSON loading worked.")
            sample_questions.at[idx, model +"_"+"answer_letters_bound"] = sample_questions.at[idx, model +"_"+"answer_letters_bound"] + ret["answerletter"]
        
        
        except Exception as e:
            print("JSON Loading did not work:")
            print(ret)
            try: 
                sample_questions.at[idx, model +"_"+"answer_letters_bound"] = sample_questions.at[idx, model +"_"+"answer_letters_bound"] + ret['answerletter']
            except:
                print(ret)

           
    end_time = datetime.now()
    dif = end_time - start_time
    print(dif )
    sample_questions.at[idx,model +"_"+"calc_time_s_repr"] = dif.seconds
    sample_questions.at[idx,model +"_"+"calc_time_micro_s_repr"] = dif.microseconds




sample_questions.to_csv("samplequestions_including_gpt4.csv")




#%% Prompting the correct answer
os.listdir()
df = pd.read_excel("final_gpt35_gpt4_2.xlsx")

df["correct_answer"] = df["correct_answer"].str.replace("Correct Answer:  ", "")
df["correct_answer"] = df["correct_answer"].str.replace(".", "")
answered_incorrectly = df["correct_answer"] != df['gpt-3.5-turbo_answerletter']
answered_incorrectly= df[answered_incorrectly ]
answered_incorrectly.shape
answered_incorrectly["correct_answer"]
answered_incorrectly = answered_incorrectly.reset_index()


model = "gpt-3.5-turbo"
answered_incorrectly[model + "_answercorrection"] = ""

for idx, row in answered_incorrectly[:100].iterrows():
    print(idx)
    text_qa = "\nQuestion:\n" + row["question"] + "\n\nChoices:\n" + row["answer"]
    answerletter = row[model + "_answerletter"]
    correct_answerletter = row["correct_answer"]
    try:
        res = promptingCorrectAnswer(text_qa=text_qa,answerletter=answerletter, correctanswerletter=correct_answerletter , model=model)
        answered_incorrectly.at[idx, model + "_answercorrection"] = res
    except Exception as e: 
        try: 
            res = promptingCorrectAnswer(text_qa=text_qa,answerletter=answerletter, correctanswerletter=correct_answerletter , model=model)
            answered_incorrectly.at[idx, model + "_answercorrection"] = res
        except Exception as e: 
            answered_incorrectly.at[idx, model + "_answercorrection"] = e
        

answered_incorrectly.to_csv("answered_incorrectly_corrections_gpt35.csv")


df = pd.read_excel("final_gpt35_gpt4_2.xlsx")

df["correct_answer"] = df["correct_answer"].str.replace("Correct Answer:  ", "")
df["correct_answer"] = df["correct_answer"].str.replace(".", "")
answered_incorrectly = df["correct_answer"] != df['gpt-4_answerletter']
answered_incorrectly= df[answered_incorrectly ]
answered_incorrectly.shape
answered_incorrectly["correct_answer"]
answered_incorrectly = answered_incorrectly.reset_index()
answered_incorrectly .shape

model = "gpt-4"
answered_incorrectly[model + "_answercorrection"] = ""

for idx, row in answered_incorrectly[:100].iterrows():
    print(idx)
    text_qa = "\nQuestion:\n" + row["question"] + "\n\nChoices:\n" + row["answer"]
    answerletter = row[model + "_answerletter"]
    correct_answerletter = row["correct_answer"]
    try:
        res = promptingCorrectAnswer(text_qa=text_qa,answerletter=answerletter, correctanswerletter=correct_answerletter , model=model)
        answered_incorrectly.at[idx, model + "_answercorrection"] = res
    except Exception as e: 
        try: 
            res = promptingCorrectAnswer(text_qa=text_qa,answerletter=answerletter, correctanswerletter=correct_answerletter , model=model)
            answered_incorrectly.at[idx, model + "_answercorrection"] = res
        except Exception as e: 
            answered_incorrectly.at[idx, model + "_answercorrection"] = e
        

answered_incorrectly.to_csv("answered_incorrectly_corrections_gpt4.csv")





