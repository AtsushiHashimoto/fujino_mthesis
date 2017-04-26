# _*_ coding: utf-8 -*-

"""
入力: (predict-mst).csv 
出力: dot
"""

import os 
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('flow_path', help=u'flow.csv')
    parser.add_argument('output_dir', help=u'output directory')

    params = parser.parse_args()

    return vars(params)


def main(params):
    ID = 0
    POSITION = 1
    TYPE = 2
    WORD = 3
    EDGES = 4
    flow_path = params['flow_path']
    output_dir = params['output_dir']
    root = os.path.splitext(os.path.basename(flow_path))[0] 
    save_path = os.path.join(output_dir, root + '.dot') 

    nodes = []
    edges = [] 
    rank = {}

    with open(flow_path, 'rt') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            if row[ID] != u'ID': #header does not always appear in the first line
                id = row[ID]
                step, sentence, pos = row[POSITION].split('-')
                # ids in the same step are the same rank
                if step not in rank:
                    rank[step] = []
                rank[step].append(id)
                type = row[TYPE]
                word = row[WORD]
                nodes.append('n%s [label="%s\n%s\n%s", shape=box];\n'%(id,step,word,type))
                for edge in row[EDGES].split(' '):
                    if ':' in edge:
                        flow_id, weight = edge.split(':')
                        #edges.append('n%s -> n%s [label="%.2f"];\n'%(flow_id,id,float(weight)))
                        edges.append('n%s -> n%s;\n'%(flow_id,id))

    with open(save_path, 'wt') as fout:
        fout.write('digraph G {\n')
        for node in nodes:
            fout.write(node)
        for edge in edges:
            fout.write(edge)
        for ids in rank.values():
            line = "{rank = same;"
            for id in ids:
                line += " n%s;"%id
            line += "}\n"
            fout.write(line)
        
        fout.write('}')



if __name__ == '__main__':
    params = parse()
    main(params)
