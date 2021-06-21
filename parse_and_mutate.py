import pygame
import vgdl
import os

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

games_path = "./vgdl/games"
input_game = "aliens.txt"
output_game = "aliens_v2.txt"

input_path = os.path.join(games_path, input_game)
with open(input_path) as f:
    gamefile = f.read()

game = vgdl.VGDLParser().parse_game(gamefile)
tree = vgdl.indent_tree_parser(gamefile, tabsize=4).children[0]
string = tree2string(tree)

output_path = os.path.join(games_path, output_game)
interaction_set = next(filter(lambda x: x.content=="InteractionSet", tree.children))

def parse_interactions(self, inodes):
    for inode in inodes:
        if ">" in inode.content:
            pair, edef = [x.strip() for x in inode.content.split(">")]
            eclass, kwargs = self._parse_args(edef)
            objs = [x.strip() for x in pair.split(" ") if len(x) > 0]

            # Create an effect for each actee
            for obj in objs[1:]:
                args = [objs[0], obj]

                if isinstance(eclass, FunctionType):
                    effect = FunctionalEffect(eclass, *args, **kwargs)
                else:
                    assert issubclass(eclass, Effect)
                    effect = eclass(*args, **kwargs)

                self.game.collision_eff.append(effect)

            if self.verbose:
                print("Collision", pair, "has effect:", effect)

with open(output_path, 'w') as f:
    f.write(string)
