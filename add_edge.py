import csv
import sys

foutName = 'All_pos_padded.bed'
finName = '../deepsea_train/allTFs.pos.bed'
pad_indicator = '1337'
num_padding_windows = range(2)

def writeFrontPad(endOld, fout, chrm):
  for window_i in num_padding_windows:
  # pad the end of last one 
    towrite = [chrm, str(int(endOld)+200 * window_i), str(int(endOld)+(200 * (window_i + 1))), pad_indicator, str(0), '*']
    fout.writelines('\t'.join(towrite))
    fout.writelines('\n')

def writeBackPad(chrm, startPos, fout):
  towrite = [chrm, startPos, str(int(startPos)+200), pad_indicator, str(0), '*']
  fout.writelines('\t'.join(towrite))
  fout.writelines('\n')


with open(finName, 'r') as fin:
  with open(foutName, 'w') as fout:
   # skip the first line. Do this manually
    reader = csv.reader(fin, delimiter='\t')
    for chrmOld, startOld, endOld, dotOld, zeroOld, starOld in reader:
      break
    for window_i in reversed(num_padding_windows):
      startPos = str(int(startOld)-200 * (window_i+1))
      writeBackPad(chrmOld, startPos, fout)

    for chrm, start, end, dot, zero, star in reader:
      if chrmOld != chrm:
        print chrm
        writeFrontPad(endOld, fout, chrmOld)
        for window_i in reversed(num_padding_windows):
          writeBackPad(chrm, str(int(start)-200*(window_i+1)), fout)

      if (not int(start) == int(endOld)) and chrmOld == chrm:
        # Add a padding 
        writeFrontPad(endOld, fout, chrm)
        # pad before the start of new peak 
        for window_i in reversed(num_padding_windows):
          startPos = str(int(start)-200 * (window_i+1))
          if startPos <=int(endOld) + 200*(max(num_padding_windows)+1):
            next
          writeBackPad(chrm, startPos, fout)

      chrmOld, endOld = chrm, end
      fout.writelines('\t'.join([chrm, start, end, dot, zero, star]))
      fout.writelines('\n')


