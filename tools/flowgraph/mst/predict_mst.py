# _*_ coding: utf-8 -*-

import os 
import csv
import math
import networkx as nx
from networkx.algorithms.tree.branchings import Edmonds

CHAR_POSITION = 0
NE_TYPE = 1
RECIPE_WORD = 2

STEP = 0
SENTENCE = 1
WORD_NO = 2

def extract_keywords(file_path):
    """
        input: recipeID/step_final_memos.txt
        output: list of (position, tag, r-NE) 
    """
    result = []
    with open(file_path, 'rt') as fin:
        for line in fin:
            sentence = 1
            line = line.strip()
            line = line.split('\t')
            if len(line) == 2:
                step, body = line
                step = int(float(step))
                for word_no, morph in enumerate(body.split(' ')):
                    morph = morph.split('=')
                    recipe_word = ""
                    for m in morph:
                        recipe_word += m.split(',')[0]
                    if recipe_word == u'。':
                        sentence += 1
                    else:
                        tag_split = (morph[-1].split(','))[-1]
                        if '/' in tag_split:
                            recipe_tag = (tag_split.split('/'))[-1]
                            if recipe_tag in ['Ac', 'Af']:
                                r_word = (tag_split.split('/'))[0] # dictionary form of verb
                                if r_word != 'NA':
                                    recipe_word = r_word
                            result.append( ((step, sentence, word_no), 
                                            recipe_tag,
                                            recipe_word) )
    return result


def predict_mst(keywords, file_path):
    """
        input: 
        keywords: list of keywords obtained by function 'extract_keywords'
        file_path: complete graph

        output: 
        max spaning tree (networkx DiGraph) 
    """
    def _edmonds_for_root(graph, root):
        # rootにしたいノードに入るエッジを切ることで強制的にrootにする
        for node in graph.predecessors(root):
            graph.remove_edge(node, root)
        edmonds = Edmonds(graph)
        mst = edmonds.find_optimum(attr='weight', kind = 'max', style='arborescence')
        return mst

    def _find_best_mst(graph, keywords):
        best_weights = None 
        best_mst = None
        roots = []
        max_step = keywords[-1][CHAR_POSITION][STEP]
        #### 最終stepのAcかAf なければ最終step全部をルートにする ####
        for id, keyword in enumerate(keywords):
            if keyword[CHAR_POSITION][STEP] == max_step and keyword[NE_TYPE] in ['Ac', 'Af']:
                roots.append(id)
        if len(roots) == 0:
            if keyword[CHAR_POSITION][STEP] == max_step:
                roots.append(id)
        # #### 一番下に現れるAcかAf ####
        #while len(roots) == 0 and max_step >= 0:
        #    for id, keyword in enumerate(keywords):
        #        if keyword[CHAR_POSITION][STEP] == max_step and keyword[NE_TYPE] in ['Ac', 'Af']:
        #            roots.append(id)
        #    max_step -= 1
        # #### 最終step全て(時間がかかる) ####
        #for id, keyword in enumerate(keywords):
        #    if keyword[CHAR_POSITION][STEP] == max_step:
        #        roots.append(id)
        for root in roots:
            copy_graph = graph.copy()
            mst = _edmonds_for_root(copy_graph, root)
            weights = mst.size(weight='weight')
            if best_weights == None or best_weights < weights:
                best_weights = weights
                best_mst = mst
        return best_mst

    graph = nx.DiGraph()
    with open(file_path, 'rt') as fin:
        for line in fin:
            n1, n2, weight = (line.strip()).split(' ')
            n1 = int(float(n1))
            n2 = int(float(n2))
            base, exp = weight.split('d') # fortran形式?
            weight = float(base) * math.pow(10, float(exp))
            #graph.add_edge(n1,n2)
            #graph.edge[n1][n2]['weight'] = weight 
            # --- Translate ---
            graph.add_edge(n2,n1)
            graph.edge[n2][n1]['weight'] = weight 

    # --- predicting root ---
    mst = _find_best_mst(graph, keywords)

    # --- Translate ---
    mst = mst.reverse()
    return mst


def generate_mst(ner_path, compG_path, output_dir, overwrite=False):
    """
        input: 
        ner_path: final_memos.txt 
        compG_path: recipe_id.mst_parse 
        output_dir: output directory 
        overwrite: if output_file exists and overwrite is False, this code do nothing.  

        output: 
        max spaning tree (networkx DiGraph) 
    """
    root, _ = os.path.splitext(os.path.basename(compG_path))

    savefile = os.path.join(output_dir, root + '.csv')
    if not overwrite and os.path.exists(savefile):
        #print "%s exists" % savefile 
        return 

    keywords = extract_keywords(ner_path) 
    mst = predict_mst(keywords, compG_path) 

    assert len(keywords) == nx.number_of_nodes(mst)

    # save
    with open(os.path.join(output_dir, root + '.csv'), 'wt') as fout:
        writer = csv.writer(fout, delimiter=',') 
        writer.writerow(['ID', 'char-position', 'NEtype', 'wordseq.', 'edges(node:cost)'])
        for id, keyword in enumerate(keywords):
            edges = ' ' 
            for pre in mst.predecessors(id):
                weight = mst.edge[pre][id]['weight']
                edges += u'%d:%f ' % (pre, weight) 
            writer.writerow([id, u'%03d-%02d-%03d'%keyword[CHAR_POSITION]] + list(keyword[NE_TYPE:]) + [edges])


