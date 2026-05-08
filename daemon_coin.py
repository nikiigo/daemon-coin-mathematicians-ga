import logging
import random
import re
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

random_pattern = False
random_pattern_length = 3
m1_pattern = r"0"
m2_pattern = r"0"
# pattern = r'(.)\0{1,}'

trials = 10000  # Number of experiments
serie_length = 1000 # Series length

alive = 0
eaten = 0

for a in range(trials):
    if random_pattern:
        ## Generate random pattern which will be used to find out first pattern occurence in series
        apattern = []
        for x in range(random_pattern_length):
            coin = random.SystemRandom().randint(0, 1)
            apattern.append(coin)
        pattern1 = ''.join(str(e) for e in apattern)
        logging.debug(f'Random pattern is {pattern1}')
        pattern2 = pattern1
    else:
        ## Use the following patterns to find out first pattern occurence in series
        pattern1 = m1_pattern
        pattern2 = m2_pattern

    ## Generate 2 random lists of coin flippping
    mathematist1 = []
    mathematist2 = []
    string1 = ''
    string2 = ''
    for x in range(serie_length):
        coin = random.SystemRandom().randint(0, 1)
        mathematist1.append(coin)
        coin = random.SystemRandom().randint(0, 1)
        mathematist2.append(coin)
    ## Find out first template occurence in series and check the mathematicians guess
    string1 = ''.join(str(e) for e in mathematist1)
    logging.debug(f'String1 is {string1}')
    string2 = ''.join(str(e) for e in mathematist2)
    logging.debug(f'String2 is {string2}')
    index1 = re.search(pattern1, string1).start()
    logging.debug(f'Index1 is {index1}')
    index2 = re.search(pattern2, string2).start()
    logging.debug(f'Index2 is {index2}')
    ## Calculate survival ratio
    if mathematist1[index2] == mathematist2[index1]:
        alive += 1
    else:
        eaten += 1
    percentage = alive / (alive + eaten)
    logging.info(
        f'{a}: Current patterns: [{pattern1} {pattern2}]. The survival ratio is approximately {percentage:2.2%}.')
if random_pattern:
    print(f'{a}: Random pattern with length {random_pattern_length}. The survival ratio is approximately {percentage:2.2%}.')
else:
    print(f'{a}: Patterns: [{pattern1} {pattern2}]. The survival ratio is approximately {percentage:2.2%}.')
