import PreProcess as pp
import numpy as np

def main():
  vocab = pp.initVocab()
  output = pp.indexToLabel("output.mat", vocab)
  labelfile = "tag.txt"
  f = open(labelfile, "wb")
  for i in range(len(output)):
      prob = output['data'][:][i]
      idx = np.argmax(prob)
      tag = vocab[i]
      f.write(tag + '\n')
  f.close()


  f = open(labelfile, "wb")


if __name__ == '__main__':
  main()

