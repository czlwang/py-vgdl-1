import pygame
import vgdl
import os
import random

from vgdl import indent_tree_parser

def tree2string(tree):
    #input: a tree
    #return: an indented string representation of the tree
    lines = ""
    indent = " "*tree.indent
    lines += indent + tree.content + "\n"
    for c in tree.children:
        lines += tree2string(c)
    return lines

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

games_path = "./vgdl/games"
input_game = "aliens.txt"
output_game = "aliens_v2.txt"

input_path = os.path.join(games_path, input_game)
with open(input_path) as f:
    gamefile = f.read()

game = vgdl.VGDLParser().parse_game(gamefile)
tree = vgdl.indent_tree_parser(gamefile, tabsize=4).children[0]

output_path = os.path.join(games_path, output_game)
interaction_set = next(filter(lambda x: x.content=="InteractionSet", tree.children))
interaction_set = interaction_set.children

#mutate all the numerical values (just fudge them by 1 for now)
for inode in interaction_set:
    if ">" in inode.content:
        pair, edef = [x.strip() for x in inode.content.split(">")]
        objs = [x.strip() for x in pair.split(" ") if len(x) > 0]

        effects = edef.split()
        new_effects = []
        for e in effects:
            if '=' in e:
                var, value = e.split('=')
                if is_int(value):
                   new_value = int(value) + random.randint(0,1) # "mutation"
                   new_effects.append(f'{var}={new_value}')
                else:
                    new_effects.append(e)
            else:
                new_effects.append(e)
        new_effect_str = " ".join(new_effects)
        new_string = f'{pair}>{new_effect_str}'
        inode.content = new_string

string = tree2string(tree)
with open(output_path, 'w') as f:
    f.write(string)
