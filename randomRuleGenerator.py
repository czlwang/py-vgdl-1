import pygame
import vgdl
import os
import random

from vgdl import indent_tree_parser

games_path = "./vgdl/games"
input_game = "aliens.txt"
input_level = "aliens_lvl0.txt"
output_game = "aliens_v2.txt"

input_path = os.path.join(games_path, input_game)
with open(input_path) as f:
    gamefile = f.read()

level_path = os.path.join(games_path, input_level)
with open(level_path) as f:
    levelfile = f.read()

game = vgdl.VGDLParser().parse_game(gamefile)
tree = vgdl.indent_tree_parser(gamefile, tabsize=4).children[0]
level = game.build_level(levelfile)

FIXED = 0
interactions = ["killSprite", "killAll", "killIfHasMore", "killIfHasLess",
            "killIfFromAbove", "killIfOtherHasMore", "spawnBehind", "stepBack", "spawnIfHasMore", "spawnIfHasLess",
            "cloneSprite", "transformTo", "undoAll", "flipDirection", "transformToRandomChild", "updateSpawnType",
            "removeScore", "addHealthPoints", "addHealthPointsToMax", "reverseDirection", "subtractHealthPoints",
            "increaseSpeedToAll", "decreaseSpeedToAll", "attractGaze", "align", "turnAround", "wrapAround",
            "pullWithIt", "bounceForward", "teleportToExit", "collectResource", "setSpeedForAll", "undoAll",
            "reverseDirection", "changeResource"]
            
def randomRuleGenerator(timesteps, usefulSprites, avatar, interactions):
    interaction = []
    termination = []

    num_interactions = int(len(usefulSprites)*(0.5 + 0.5 * random.random()))
    if FIXED > 0:
        num_interactions = FIXED

    for i in range(num_interactions):
        i1 = random.randint(0, len(usefulSprites) - 1)
        i2 = (i1+1+random.randint(0, len(usefulSprites)-1))%len(usefulSprites)
        scoreChange = ""
        if random.randint(0, 1):
            scoreChange += "scoreChange=" + str((random.randint(0, 5) - 2))

        interaction.append(usefulSprites[i1] + " " + usefulSprites[i2] + " > " + 
            interactions[random.randint(0, len(interactions))] + " " + scoreChange)
        for i in range(timesteps):
            try:
                tree = indent_tree_parser("\n".join(interaction))
                for c in tree.children:
                    vgdl.VGDLParser().parse_interactions(c.children)
            except:
                interaction.remove(i)
                interaction.append(usefulSprites[i1] + " " + usefulSprites[i2] + " > " + 
                    interactions[random.randint(0, len(interactions))] + " " + scoreChange)

        if random.randint(0, 1):
            termination.append("Timeout limit=" + str((800 + random.randint(0, 500))) + " win=True")
        else:
            chosen = usefulSprites[random.randint(0, len(usefulSprites))]
            try:
                tree = indent_tree_parser("\n".join(termination))
                for c in tree.children:
                    vgdl.VGDLParser().parse_terminations(c.children)
            except:
                termination.remove(len(termination) - 1)
                termination.append("SpriteCounter stype=" + chosen + " limit=0 win=True")
    termination.append("SpriteCounter stype=" + avatar + " limit=0 win=False")

    return interaction, termination

usefulSprites = list(level.init_state.data['sprites'].keys())
avatar = level.get_avatars()[random.randint(0, len(level.get_avatars()) - 1)]
avatar = str(avatar)
timesteps = 10
print(randomRuleGenerator(timesteps, usefulSprites, avatar, interactions))