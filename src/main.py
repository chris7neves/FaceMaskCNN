import argparse
import os
import sys

from configs.paths import paths



parser = argparse.ArgumentParser(description="Entrypoint to FaceMaskCNN model.")

subparsers = parser.add_subparsers(dest='mode')

# Training sub parser
train_parser = subparsers.add_parser("train")
train_parser.add_argument("--gen-report", action="store_true")

# Testing sub parser
test_parser = subparsers.add_parser("test")
test_parser.add_argument("--gen-report", action="store_true")

# Inference sub parser
infer_parser = subparsers.add_parser("infer")
infer_parser.add_argument("--gen-report", action="store_true")

# List models
list_parser = subparsers.add_parser("list_models")


args = parser.parse_args()

print(args.__str__())