# _*_ coding: utf-8 -*-

import os 
import re
import io
import csv
import copy
import json
import argparse

import pandas as pd


def get_dataframe_of_flowgraph(flow_path):
    """
        input: flow_graph_path
        output: pandas dataframe
    """
    ID = 0
    POSITION = 1

    ids = [] # It is not always number.
    data = []

    with open(flow_path, 'rt') as fin:
        reader = csv.reader(fin, delimiter=',')
        next(reader) # skip header
        for row in reader:
            id = row[ID]
            ids.append(id)
            step, sentence, word_no = row[POSITION].split('-')
            data.append([int(float(step)), int(float(sentence)), int(float(word_no))] + row[POSITION+1:]) 

    flow_df = pd.DataFrame(data)
    flow_df.index = ids
    flow_df.columns = ['step', 'sentence', 'word_no', 'NEtype', 'rNE', 'enter_edges']

    return flow_df


def get_ingredients(file_path):
    """
        input: ingredients_path
        output: a list of ingredients 
    """
    INGREDIENT = 1

    r = re.compile(ur'[0-9A-Za-zぁ-んァ-ン一-龥ー]+')
    ingredients = []
    with open(file_path, 'rt') as fin:
        reader = csv.reader(fin, delimiter='\t')
        for row in reader:
            ingredient = row[INGREDIENT]
            ingredient = unicode(ingredient, 'utf-8')
            strs = r.findall(ingredient) # remove symbol
            ingredient = ' '.join(strs)
            ingredient = ingredient.encode('utf-8')
            ingredients.append(ingredient)

    return ingredients 


def generate_label(flow_df, ingredients, synonym):
    """
        input:  pandas dataframe of flow graph
                a list of ingredients 
                dictionary of synonym 
    """

    nodes = {} # key:index value:food or verb
    linked_idxs = {} # key:index value:a list of linked indices & hop
    idxs_per_step = {} # key:step value:indices in the step
    ACTION_TYPES = ['Ac', 'Sf']

    def _convert_by_synonym(word, partial_match=False):
        if partial_match:
            #正規表現により一番長くマッチするキーの値を返す
            match_word = None
            for synonym_word in synonym.keys():
                match_object = re.search(synonym_word, word)
                if match_object:
                    m_word = match_object.group()
                    if match_word == None or len(m_word) > len(match_word):
                        match_word = m_word
            if match_word != None:
                return synonym[match_word]
            else:
                return None
        elif not partial_match:
            if word in synonym:
                return synonym[word]
            else:
                return None

    def _make_node(rNE, NEtype, step, sentence, word_no, weight, ontology_pmatch = False):
        rNE = unicode(rNE, 'utf-8')
        node = {
                'rNE':rNE,
                'NEtype':NEtype,
                'ontology':_convert_by_synonym(rNE, partial_match=ontology_pmatch),
                'step':step,
                'sentence':sentence,
                'word_no':word_no,
                'weight':weight
                }
        return node

    def _related_food(idx):
        row = flow_df.ix[idx]
    
        # register step
        if row['step'] not in idxs_per_step:
            idxs_per_step[row['step']] = []
        idxs_per_step[row['step']].append(idx)
        if idx not in linked_idxs:
            linked_idxs[idx] = [] 

        # register previous linked indices 
        pre_idxs = row['enter_edges'].split()
        for pre_idx in pre_idxs:
            pre_idx, _ = pre_idx.split(':')
            if pre_idx not in linked_idxs:
                _related_food(pre_idx) # recursion 
            linked_idxs[idx] += [[link[0], link[1] + 1] for link in linked_idxs[pre_idx]]

        # 食材の場合
        if row['NEtype'] == 'F':
            current_node_list = []
            if row['rNE'] == '材料':
                for i, food in enumerate(ingredients):
                    node = _make_node(food, 'F',row['step'],row['sentence'],row['word_no'],1.0,True) 
                    linked_idxs[idx].append(['%s_%d'%(idx,i), 0]) # ホップ数
                    nodes['%s_%d'%(idx,i)] = node 
                    current_node_list.append(node)
            else:
                node = _make_node(row['rNE'], 'F',row['step'],row['sentence'],row['word_no'],1.0) 
                linked_idxs[idx].append([idx, 0]) # ホップ数
                nodes[idx] = node 
                current_node_list.append(node)
            # verbを追加する
            for link_idx, hop in linked_idxs[idx]:
                link_node = nodes[link_idx]
                if link_node['NEtype'] in ACTION_TYPES:
                    for node in current_node_list:
                        if link_node['NEtype'] not in node:
                            node[link_node['NEtype']] = [] 
                        node[link_node['NEtype']].append({"id":link_idx, "hop":-hop, "weight":1.0})

        # 動詞の場合 
        elif row['NEtype'] in ACTION_TYPES:
            node = _make_node(row['rNE'],row['NEtype'],row['step'],row['sentence'],row['word_no'],1.0) 
            nodes[idx] = node 
            linked_idxs[idx].append([idx, 0]) # ホップ数
            # foodに追加する
            for link in linked_idxs[idx]:
                link_idx, hop = link
                if nodes[link_idx]['NEtype'] == 'F':
                    food_node = nodes[link_idx]
                    if row['NEtype'] not in food_node:
                        food_node[row['NEtype']] = [] 
                    food_node[row['NEtype']].append({"id":idx, "hop":hop, "weight":1.0}) # index and hop


    #フローグラフの各ノードに対して遡って現れる食材を求める
    for idx in flow_df.index:
        if idx not in linked_idxs:
            _related_food(idx)

    links = set() 
    label = {}
    for step, idxs in sorted(idxs_per_step.items(), key = lambda x: x[0]):
        label[step] = {}
        links.update([link[0] for idx in idxs_per_step[step] for link in linked_idxs[idx]])
        for link in links:
            node = copy.deepcopy(nodes[link])
            if node['NEtype'] == 'F':
                for action_type in ACTION_TYPES:
                    if action_type in node:
                        # 後ろのステップで現れる動詞は除去
                        node[action_type] = [l for l in node[action_type] if l["id"] in links]
            label[step][link] = node

    return label 


