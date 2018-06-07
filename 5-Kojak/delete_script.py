with open('delete.txt') as txtfile:
	lines = txtfile.readlines()

import re

lines = [re.sub('\n','',line) for line in lines]

import os

for f in lines:
	os.remove(f)
