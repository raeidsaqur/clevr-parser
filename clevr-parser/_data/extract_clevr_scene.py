#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json


def main():
    fp = 'vocab.json'
    with open(fp, 'r') as f:
        caption_token_to_idx = json.load(f)['question_token_to_idx']

    caption_tokens = list(caption_token_to_idx.keys())
    print(caption_tokens[5:])

    with open("clevr-scene-nouns.txt", "w") as output:
        for row in caption_tokens[5:]:
            output.write(str(row) + '\n')


if __name__ == '__main__':
    main()