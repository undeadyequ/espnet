#!/usr/bin/env python3
import os
import argparse


def downsample(wav_f, out_wav_f, d_sample_rate=16000):
  os.system("sox {} -r {} {}".format(wav_f, d_sample_rate, out_wav_f))
  return out_wav_f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_wav_f', type=str, help='text to be cleaned')
    parser.add_argument("out_wav_f", type=str, default="kana")
    parser.add_argument("rate", type=int)
    args = parser.parse_args()
    downsample(args.in_wav_f, args.out_wav_f, args.rate)