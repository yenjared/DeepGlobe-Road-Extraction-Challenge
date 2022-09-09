import argparse
import dataprep

parser = argparse.ArgumentParser()
parser.add_argument("inpath",help="path to masks_instances folder")

args = parser.parse_args()
dataprep.crops_to_phaseone(args.inpath)