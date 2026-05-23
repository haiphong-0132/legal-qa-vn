import bisect

iterators = {}

class Iterator:
    def __init__(self, postings, term, term_string, docs=None):
        self.postings = postings
        if docs is None:
            self.docs, _ = zip(*self.postings)
        else:
            self.docs = docs
        self.cursor = 0
        self.term = term
        self.term_string = term_string
    #return posting at current cursor position
    def current(self):
        return self.postings[self.cursor]
    #advance cursor position if not at last posting
    def next(self):
        if self.last():
            return False
        else:
            self.cursor += 1
            return True
    #advance cursor to nearest position less than doc
    def gallop_to(self, doc):
        if doc == self.postings[self.cursor][0]:
            return True
        if doc < self.postings[self.cursor][0]:
            return False
        elif doc > self.postings[-1][0]:
            self.cursor = len(self.postings) - 1
            return False
        else:
            i = bisect.bisect_left(self.docs, doc, self.cursor)
            self.cursor = i
            return True
    def last(self):
        if self.cursor == len(self.postings) - 1:
            return True
        else:
            return False
    def __str__(self):
        return "IT:: t:{}, cursor: {}\n postings: {}\nAt last element: {}\n".format(self.term, self.cursor, self.postings,self.last())
        

#creates iterator for term t, adds it to the list, returns first posting
def first_posting(postings, t, term_string, docs=None):
    it = Iterator(postings, t, term_string, docs=docs)
    iterators[term_string] = it
    return it.current()
#return next posting or None
def next_posting(term_string):
    it = iterators[term_string]
    if it.next():
        return it.current()
    else:
        return None
#return nearest posting that is larger than or equal to doc or None
def seek_to_document(term_string, doc):
    it = iterators[term_string]
    if it.gallop_to(doc):
        return it.current()
    else: 
        return None

def WAND_Algo(query_terms, top_k, inverted_index, term_max_score=None, docs_by_term=None):
    iterators.clear()
    max_weights = {}
    candidates = []

    if docs_by_term is None:
        docs_by_term = {}

    unique_terms = list(dict.fromkeys(query_terms))
    for t, qterm in enumerate(unique_terms):
        posting_list = inverted_index.get(qterm)
        if not posting_list:
            continue
        if term_max_score is not None and qterm in term_max_score:
            max_weights[qterm] = term_max_score[qterm]
        else:
            max_weights[qterm] = max(posting_list, key=lambda x: x[1])[1]
        docs = docs_by_term.get(qterm)
        if docs is None:
            docs = tuple(doc_id for doc_id, _score in posting_list)
            docs_by_term[qterm] = docs
        #get first candidate (first posting in the iterator) for the query term
        c_did, c_w = first_posting(posting_list, t, qterm, docs=docs)
        candidates.append((c_did, c_w, qterm))
    theta = float("-inf")
    ans = []
    fully_evaluated = 0
    while candidates:
        candidates.sort(key=lambda c: c[0])
        score_limit = 0
        pivot = 0
        pivot_found = False
        while pivot < len(candidates):
            tmp_s_lim = score_limit + max_weights[candidates[pivot][2]]
            if tmp_s_lim >= theta:
                pivot_found = True
                break
            score_limit = tmp_s_lim
            pivot += 1
        #if pivot term is the last term then DONE
        if not pivot_found:
            break
        pivot_doc = candidates[pivot][0]

        if candidates[0][0] == pivot_doc:
            fully_evaluated += 1
            s = 0
            t = 0
            cand_len = len(candidates)
            removed_candidates = []
            while t < cand_len: 
                if candidates[t][0] == pivot_doc:
                    s += candidates[t][1]
                    next_candidate = next_posting(candidates[t][2])
                    if not next_candidate: #candidate has reached end of posting list
                        removed_candidates.append(candidates[t])
                    else:
                        candidates[t] = next_candidate + (candidates[t][2],)
                    t += 1
                else:
                    break
            #remove candidates that are at the end of their posting list
            if removed_candidates:
                removed_terms = {candidate[2] for candidate in removed_candidates}
                candidates = [candidate for candidate in candidates if candidate[2] not in removed_terms]
            #if pivot doc contains all query terms needed for its accumulated upper bound to be greater than threshold
            if s > theta:
                ans.append((s,pivot_doc))
                if len(ans) > top_k:
                    #the list should be sorted by score in decrease order, if two documents have same score, smaller document id precedes larger one
                    ans.remove(min(ans, key=lambda  x: (x[0],-x[1])))
                    theta = min(ans, key=lambda  x: x[0])[0]
        else:
            removed_candidates = []
            for t in range(0,pivot):
                seeked_candidate = seek_to_document(candidates[t][2],pivot_doc)
                if not seeked_candidate: #remove candidates if they need to be advanced past their posting list bounds
                    removed_candidates.append(candidates[t])
                else:
                    candidates[t] = seeked_candidate + (candidates[t][2],)
            #remove candidates
            if removed_candidates:
                removed_terms = {candidate[2] for candidate in removed_candidates}
                candidates = [candidate for candidate in candidates if candidate[2] not in removed_terms]
    ans = sorted(ans, key=lambda x: (-x[0],x[1]))    
    return (ans,fully_evaluated)