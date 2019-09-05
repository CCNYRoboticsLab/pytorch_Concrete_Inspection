'''
generate train and validate list
'''

import os, sys
import numpy as np
from random import shuffle

train_list =  open('train_list.txt', 'w')
test_list =  open('test_list.txt', 'w')

train_list_1 =  open('train_list_crack_1.txt', 'w')
test_list_1 =  open('test_list_crack_1.txt', 'w')

train_list_2 =  open('train_list_crack_2.txt', 'w')
test_list_2 =  open('test_list_crack_2.txt', 'w')

accept_base = 'accept/'
noaccept_base = 'nonaccpet/'

third_path = 'crackSubImageForTraining/130BY130/'
ten_path   = 'crackSubImageForTraining/100BY100/'

base_path = 'deepLearningBridgeInspection/'

accept_file  =[]
noaccepet_file  =[]

for file in os.listdir(base_path + third_path + accept_base):
   if file.endswith('.jpg'):
       accept_file.append(third_path + accept_base + file )

for file in os.listdir(base_path + ten_path + accept_base):
   if file.endswith('.jpg'):
       accept_file.append(ten_path + accept_base + file)

for file in os.listdir(base_path + third_path + noaccept_base):
   if file.endswith('.jpg'):
       noaccepet_file.append(third_path + noaccept_base + file)

for file in os.listdir(base_path + ten_path + noaccept_base):
   if file.endswith('.jpg'):
       noaccepet_file.append(ten_path + noaccept_base + file)

###
spall_accept_file     =[]
spall_noaccepet_file  =[]

accept_base = 'accepted/'
noaccept_base = 'nonAccepted/'

spall_third_path = 'spallSUbImageForTRaining/imageClustersV130B130/'
spall_ten_path   = 'spallSUbImageForTRaining/imageClustersV2100B100/'

for file in os.listdir(base_path + spall_third_path + accept_base):
   if file.endswith('.jpg'):
       spall_accept_file.append(spall_third_path + accept_base + file )

for file in os.listdir(base_path + spall_ten_path + noaccept_base):
   if file.endswith('.jpg'):
       spall_noaccepet_file.append(spall_ten_path + noaccept_base + file )

shuffle(spall_accept_file)
shuffle(spall_noaccepet_file)

for i in range(len(spall_accept_file)):
    if i < int(len(spall_accept_file)*0.8):
        train_list.write(spall_accept_file[i] + ',' + str(1) + '\n')
    else:
        test_list.write(spall_accept_file[i] + ',' + str(1) + '\n')

for i in range(len(spall_noaccepet_file)):
    if i < int(len(spall_noaccepet_file)*0.8):
        train_list.write(spall_noaccepet_file[i] + ',' + str(0) + '\n')
    else:
        test_list.write(spall_noaccepet_file[i] + ',' + str(0) + '\n')

shuffle(accept_file)
shuffle(noaccepet_file)

for i in range(len(accept_file)):
    if i < int(len(accept_file)*0.8):
        train_list_1.write(accept_file[i] + ',' + str(1) + '\n')
        train_list_2.write(accept_file[i] + ',' + str(2) + '\n')
    else:
        test_list_1.write(accept_file[i] + ',' + str(1) + '\n')
        test_list_2.write(accept_file[i] + ',' + str(2) + '\n')

for i in range(len(noaccepet_file)):
    if i < int(len(noaccepet_file)*0.8):
        train_list_1.write(noaccepet_file[i] + ',' + str(0) + '\n')
        train_list_2.write(noaccepet_file[i] + ',' + str(0) + '\n')
    else:
        test_list_1.write(noaccepet_file[i] + ',' + str(0) + '\n')
        test_list_2.write(noaccepet_file[i] + ',' + str(0) + '\n')


train_list.close()
test_list.close()

train_list_1.close()
test_list_1.close()

train_list_2.close()
test_list_2.close()
