import os, sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
    
import clevr_parser
import clevr_parser.utils as utils
parser = clevr_parser.Parser().get_backend(identifier='spacy', model='en_core_web_sm')

# import spacy
# from spacy import displacy
from IPython.core.display import display, Image, HTML

def visualize(doc, dep=False):
    displacy.render(doc, style='ent', jupyter=True)
    if dep:
        displacy.render(doc, style='dep', jupyter=True, options={'distance': 70})

img_fn = 'CLEVR_train_000001.png'
# img_path = f'../data/2obj/images/train/{img_fn}'
img_path = '../data/raw/CLEVR_v1.0/images/train/{}'.format(img_fn)
# display(Image(filename=img_path))

# Grounding Graph #
# img_groundings_path = "../data/2obj/scenes_parsed/train_scenes_parsed.json"
img_groundings_path = "../data/raw/CLEVR_v1.0/scenes/CLEVR_train_scenes.json"
img_grounding = utils.load_grounding_for_img(img_fn, img_groundings_path)

G_img = parser.draw_clevr_img_scene_graph(img_grounding)