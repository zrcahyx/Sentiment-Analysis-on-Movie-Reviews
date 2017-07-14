#!/usr/bin/python
#-*- coding: utf-8 -*-

from os.path import abspath, dirname, join

def get_cfg_path():
    config_dir = join(dirname(dirname(abspath(__file__))), 'config')
    cfg_path = join(config_dir, 'config.cfg')
    return cfg_path

def get_file_num_line(file_path):
    num_line = 0
    with open(file_path, 'r') as f:
        for line in f:
            num_line += 1
    return num_line

def main():
    pass

if __name__ == '__main__':
    main()
