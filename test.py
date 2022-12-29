from datasets.SEEDV import SEEDVFewShotLearning
from utils.parser_utils import get_args
args, device = get_args() # inside calls argparse
dataset = SEEDVFewShotLearning(args)