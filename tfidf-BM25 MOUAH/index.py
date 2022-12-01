# -*- coding: utf-8 -*-

import operator
import sys

import math # tf_idf , math.log
from stemming.porter2 import stem # stemmer

class Index :
  max_doc_id = 0 # a unique doc id && all document size 
  doc = {}
  doc_meta = {} # key : doc id , value : all term size&times
  index = {}

  def __init__(self):
    max_doc_id = 0

def analyze(content):
  content = content.replace(", ", " ")
  content = content.replace(":", " ")
  content = content.replace("\t", " ")
  content = content.replace("\'", " ")
  content = content.replace("\"", " ")
  ss = content.split(" ")
  tokens = {}
  for s in ss :
    sts = stem(s.lower().strip())
    if len(sts) > 0 :
      tokens[sts] = tokens.get(sts, 0) + 1
  return tokens


def build_index(idx):
  print("Building Index .. ")
  f = open('data/data.txt', 'r')
  for line in f:
    idx.doc[idx.max_doc_id] = line.strip()  # store doc
    tokens = analyze(line)
    term_sz = 0
    for k, v in tokens.items():
      idx_term = {}
      if k in idx.index :
        idx_term = idx.index[k]
      else :
        idx_term = {}
      idx_term[idx.max_doc_id] = v
      idx.index[k] = idx_term
      term_sz += v
    idx.doc_meta[idx.max_doc_id] = term_sz
    idx.max_doc_id = idx.max_doc_id + 1
  f.close()


# tf-idf k in document D
# stid : size of term k in  D
# atsid :  all term size in D 
# ads : all document size 
# dscct : document size contains current term
def tf_idf(stid, atsid, ads, dscct):
  #lg 10 = 2.302585
  if ads == 0 or atsid == 0 or dscct == 0 :
    return 0
  return (float(stid) / atsid) * math.log(float(ads) / (dscct))/2.302585

# ref: https://zh.wikipedia.org/wiki/Tf-idf
def get_tfidf_score(idx, term):
  scores = {}
  term_in_idx = idx.index.get(term, None)
  if term_in_idx is not None :
    for k, v in term_in_idx.items() :
      scores[k] = tf_idf(v, idx.doc_meta[k], idx.max_doc_id, len(term_in_idx))
  return scores

def query_by_sum_of_tfidf(idx, terms):
  doc_ids = {} # key : doc id , value : score
  for term in terms:
    c_doc_ids = get_tfidf_score(idx, term)
    for ck, cv in c_doc_ids.items() :
      doc_ids[ck] = doc_ids.get(ck, 0) + cv * terms[term]
  sorted_doc_ids = sorted(iter(doc_ids.items()), key=operator.itemgetter(1), reverse=True)
  lmt = 10
  curr_sz = 0
  print("Results by Sum of TF-IDFs")
  for k, v in sorted_doc_ids :
    print("[", k, "] (", v, ")", idx.doc[k])
    curr_sz += 1
    if curr_sz > 10 :
      break

# ref: https://en.wikipedia.org/wiki/Okapi_BM25
def query_by_sum_of_bm25(idx, terms):
  doc_ids = {} # key : doc id , value : score

  # k1 amd b are free parameters, ref: https://en.wikipedia.org/wiki/Okapi_BM25#cite_note-1
  k1=1.2
  b=0.75

  sumdl = 0
  dlen = 0
  for term in terms:
    cdlen = terms[term]
    sumdl += len(term) * cdlen
    dlen += cdlen
  # the average document length in the text collection
  # from which documents are drawn
  avgdl = sumdl / dlen

  for term in terms:
    # For current term
    c_doc_ids = {}
    term_in_idx = idx.index.get(term, None)
    if term_in_idx is not None :
      # stid : size of term k in  D: v
      # atsid :  all term size in D / number of words in D: idx.doc_meta[k]
      # ads : all document size: idx.max_doc_id
      # dscct : document size contains current term: len(term_in_idx)
      ads = idx.max_doc_id # aka total number of documents in the collection
      dscct = len(term_in_idx) # aka n(q_i)  the number of documents containing q_i
      for k, v in term_in_idx.items() :
        stid = v
        atsid = idx.doc_meta[k]
        # c_doc_ids[k] = tf_idf(v, idx.doc_meta[k], idx.max_doc_id, len(term_in_idx))
        idf = math.log(ads - dscct + 0.5) / math.log(dscct + 0.5)
        tf = float(stid) / atsid
        c_doc_ids[k] = idf * tf * ( k1 + 1) / ( tf + k1 * ( 1 - b + b * ( float(len(term)) / avgdl)))
    # accumulate the scores for each q
    # add 0 if not exists
    for ck, cv in c_doc_ids.items() :
      doc_ids[ck] = doc_ids.get(ck, 0) + cv * terms[term]

  sorted_doc_ids = sorted(iter(doc_ids.items()), key=operator.itemgetter(1), reverse=True)
  lmt = 10
  curr_sz = 0
  print("Results by BM25")
  for k, v in sorted_doc_ids :
    print("[", k, "] (", v, ")", idx.doc[k])
    curr_sz += 1
    if curr_sz > 10 :
      break

def query_index(idx):
  q = input("Your Query (e.g. Data Mining): ")
  terms = analyze(q)
  if len(terms) == 0:
    print("\nLength of terms is ZERO, invalid input")
  else:
    print("\nGot Input: ", q)
    print("Split to: ", terms)
    query_by_sum_of_tfidf(idx, terms)
    query_by_sum_of_bm25(idx, terms)


if __name__ == '__main__':
  idx = Index()
  build_index(idx)
  print(idx.max_doc_id , " records indexed ..")
  query_index(idx)
