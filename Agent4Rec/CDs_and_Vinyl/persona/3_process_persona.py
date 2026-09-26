import json
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CDs_and_Vinyl')
parser.add_argument('--model_name', default='gpt-5-mini')


if __name__ == "__main__":
    args = parser.parse_args()
    
    with open(f'./dataset/{args.dataset}/user_sets.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_persona = []
    for line in lines:
        key = line.strip()
        with open(f'./persona_{args.dataset}_{args.model_name}/{key}.txt', 'r', encoding='utf-8') as f:
            persona = f.readlines()
        
        for l in persona:
            if 'Taste' in l:
                t_ = l.replace('**',"").split('Taste:')[-1].strip()
            elif 'Reasons' in l:
                r_ = l.replace('**',"").split('Reasons:')[-1].strip()
            elif 'High Rating' in l:
                hr = l.replace('**',"").split('High Rating:')[-1].strip()
            elif 'Low Rating' in l:
                lr = l.replace('**',"").split('Low Rating:')[-1].strip()
        
        u_ = {'user_id':key, 'taste':t_, 'reasons':r_, 'high_rating': hr, 'low_rating':lr}
        all_persona.append(u_)
    
    df = pd.DataFrame(all_persona)
    df.to_csv(f'./persona_{args.dataset}_{args.model_name}/all_persona.csv',index=False) # 페르소나 txt 파일들을 읽어서 하나의 csv 파일로 통합하여 저장