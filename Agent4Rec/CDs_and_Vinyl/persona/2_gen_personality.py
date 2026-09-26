import json
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CDs_and_Vinyl')
parser.add_argument('--model_name', default='gpt-5-mini')


if __name__ == "__main__":
    args = parser.parse_args()

    with open(f'./dataset/{args.dataset}/train.json', 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(f'./dataset/{args.dataset}/meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(f'./dataset/{args.dataset}/review.json', 'r', encoding='utf-8') as f:
        review = json.load(f)
    with open(f'./dataset/{args.dataset}/user2id.json', 'r', encoding='utf-8') as f:
        user2id = json.load(f)
    with open(f'./dataset/{args.dataset}/item2id.json', 'r', encoding='utf-8') as f:
        item2id = json.load(f)
    with open(f'./dataset/{args.dataset}/user_sets.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    from tqdm import tqdm
    activity_list, div_list, conf_list = [], [], []
    act_dict, div_dict, conf_dict = {},{},{}

    for k, v in tqdm(train['History'].items()): # 사용자 ID와 해당 사용자가 상호작용한 아이템 목록
        activity_list.append(len(v))
        act_dict[k] = len(v)
        user_cate = []
        user_conf = 0
        count = 0
        for v_ in v: # 사용자가 상호작용한 각 아이템에 대해
            user_cate += meta[str(v_)]['categories'] # 아이템의 카테고리를 사용자 카테고리 목록에 추가
            try:
                user_conf += (review[k][str(v_)]['rating'] - meta[str(v_)]['average_rating'])**2 # conformity 계산: 값이 작을수록 남들과 비슷한 평점을 남기는 순응형 유저
                count +=1
            except:
                0
        if count>0:
            user_conf/=count

        conf_list.append(user_conf)
        div_list.append(len(set(user_cate))) # diversity: 사용자가 상호작용한 아이템들의 카테고리 수
        div_dict[k] = len(set(user_cate))
        conf_dict[k] = user_conf
        
    import numpy as np

    personality = {'activity':{}, 'diversity':{}, 'conformity':{}}
    activity_percentile = np.percentile(activity_list, [60, 90, 100])
    diversity_percentile = np.percentile(div_list, [33, 66, 100])
    conformity_percentile = np.percentile(conf_list, [25, 80, 100])

    for l in lines:
        k = l.strip()

        # 각 유저의 수치가 어느 구간에 속하는지 판단하여 0, 1, 2 값을 할당
        if act_dict[k] < activity_percentile[0]:
            personality['activity'][k] = 0
        elif act_dict[k] < activity_percentile[1]:
            personality['activity'][k] = 1
        else:
            personality['activity'][k] = 2
        
        if div_dict[k] < diversity_percentile[0]:
            personality['diversity'][k] = 0
        elif div_dict[k] < diversity_percentile[1]:
            personality['diversity'][k] = 1
        else:
            personality['diversity'][k] = 2
            
        if conf_dict[k] < conformity_percentile[0]:
            personality['conformity'][k] = 0
        elif conf_dict[k] < conformity_percentile[1]:
            personality['conformity'][k] = 1
        else:
            personality['conformity'][k] = 2

    with open(f"./persona_{args.dataset}_{args.model_name}/personality.json", "w", encoding="utf-8") as f:
        json.dump(personality, f, ensure_ascii=False, indent=4)