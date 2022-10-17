"""
this file is used to calculate the word, definition, example distributions. 
"""
import json

with open('../data/oxford/oxford_all.json','r') as fr, open('../data/oxford/statistic.csv','a') as fw:
    type_dict = {}
    pos_dict = {'headword':{},"phrase":{},'total':{}}
    definition_dict = {'headword':{},'phrase':{},'total':{}}
    definitions = 0
    examples = 0
    tokens = 0
    for e in fr:
        e = json.loads(e)
        _type = e['type']
        #if _type == 'phrase':
        #    continue
        tokens+=1
        type_dict[_type] = type_dict.get(_type,0) + 1
        num_definition = len(e['senses'])
        definition_dict[_type][num_definition] = definition_dict[_type].get(num_definition,0) +1
        definitions += num_definition
        for instance in e['senses']:
            len(instance['examples'])
            examples += len(instance['examples'])
            pos = instance['pos']
            pos_dict[_type][pos] = pos_dict[_type].get(pos,0) + 1
    keys = list(set(pos_dict['headword'].keys()) | set(pos_dict['phrase'].keys()))
    for k in keys:
        pos_dict['total'][k] = pos_dict['headword'].get(k,0) + pos_dict['phrase'].get(k,0)
    keys2 = list(set(definition_dict['headword'].keys()) | set(definition_dict['phrase'].keys()))
    for k in keys2:
        definition_dict['total'][k] = definition_dict['headword'].get(k,0) + definition_dict['phrase'].get(k,0)
    print(f"This dataset has {tokens} tokens, {definitions} definitions, {examples} examples.")
    print(type_dict)
    print(pos_dict)
    print(definition_dict)

    fw.write('Word type, headword, phrase, Total\n')
    l = sorted(pos_dict['total'].items(),reverse=True, key=lambda x:x[1])
    keys = [e[0] for e in l ]
    for k in keys:
        a, b, c = pos_dict['headword'].get(k,0), pos_dict['phrase'].get(k,0), pos_dict['total'].get(k,0)
        fw.write(f'{k}, {a}, {b}, {c}\n')
    a = sum(pos_dict['headword'].values())
    b = sum(pos_dict['phrase'].values())
    c = sum(pos_dict['total'].values())
    fw.write(f"Total, {a}, {b}, {c}\n\n")

    fw.write('Word type, headword, phrase, Total\n')
    for k in sorted(keys2):
        a, b, c = definition_dict['headword'].get(k,0), definition_dict['phrase'].get(k,0), definition_dict['total'].get(k,0)
        fw.write(f'{k}, {a}, {b}, {c}\n')
    a = sum(definition_dict['headword'].values())
    b = sum(definition_dict['phrase'].values())
    c = sum(definition_dict['total'].values())
    fw.write(f"Total, {a}, {b}, {c}\n")



