#!/usr/bin/python3
# -*- coding: big5 -*-
import numpy as np 

dic = {}
dic['ㄅ'] = []
dic['ㄆ'] = []
dic['ㄇ'] = []
dic['ㄈ'] = []
dic['ㄉ'] = []
dic['ㄊ'] = []
dic['ㄋ'] = []
dic['ㄌ'] = []
dic['ㄍ'] = []
dic['ㄎ'] = []
dic['ㄏ'] = []
dic['ㄐ'] = []
dic['ㄑ'] = []
dic['ㄒ'] = []
dic['ㄓ'] = []
dic['ㄔ'] = []
dic['ㄕ'] = []
dic['ㄖ'] = []
dic['ㄗ'] = []
dic['ㄘ'] = []
dic['ㄙ'] = []
dic['ㄧ'] = []
dic['ㄨ'] = []
dic['ㄩ'] = []
dic['ㄚ'] = []
dic['ㄛ'] = []
dic['ㄜ'] = []
dic['ㄝ'] = []
dic['ㄞ'] = []
dic['ㄟ'] = []
dic['ㄠ'] = []
dic['ㄡ'] = []
dic['ㄢ'] = []
dic['ㄣ'] = []
dic['ㄤ'] = []
dic['ㄥ'] = []
dic['ㄦ'] = []

with open('Big5-ZhuYin.map', 'r' , encoding='big5hkscs') as fp:
	for i in range(13009):
		a = fp.readline().replace('\n','').split(' ') 
		b = a[1].split('/')
		for j in range(len(b)):
			for x in dic:
				if(x==b[j][0]):
					dic[x].append(a[0])
for x in dic:
	dic[x]=list(set(dic[x]))
with open('ZhuYin-Big5.map', 'w' , encoding='big5hkscs') as fp: 
	for x in dic:
		fp.write('%s '%(x))
		for y in dic[x]:
			fp.write(' %s'%(y))
		fp.write('\n')
		for y in dic[x]:
			fp.write('%s %s \n'%(y,y))
 

        