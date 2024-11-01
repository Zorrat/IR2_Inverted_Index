import math
import nltk
from collections import OrderedDict
import re
import unicodedata
from pprint import pprint as pp



class Node:

    def __init__(self, value=None, next=None,tf_idf=None):

        """ Class to define the structure of each node in a linked list (postings list).
            Value: document id, Next: Pointer to the next node
            Add more parameters if needed.
            Hint: You may want to define skip pointers & appropriate score calculation here"""
        self.value = value  # Document ID
        self.tf_idf = tf_idf  # TF-IDF score
        self.next = next  # Pointer to the next node
        self.skip = None  # Pointer to the node after skip_length

       

    
    def __str__(self):
        return f'{str(self.value)}'


class LinkedList:
    """ Class to define a linked list (postings list). Each element in the linked list is of the type 'Node'
        Each term in the inverted index has an associated linked list object.
        Feel free to add additional functions to this class."""
    def __init__(self):
        self.head = None
        self.end_node = None
        self.length, self.n_skips, self.idf = 0, 0, 0.0
        self.skip_length = None

    def to_list(self):
        traversal = []
        if self.head is None:
            return []
        else:
            """ Write logic to traverse the linked list.
                To be implemented."""
            
            current_node = self.head
            while current_node:
                traversal.append(current_node.value)
                current_node = current_node.next
            return traversal
        
    def to_list_skip(self):
        traversal = []
        if self.head is None:
            return []
        else:
            current_node = self.head
            traversal.append(current_node.value)
            while current_node:
                if current_node.skip:
                    traversal.append(current_node.skip.value)
                current_node = current_node.skip
            return traversal


    def add_skip_connections(self):
        """ Add skip pointers in the postings lists for each term. There should be 
            floor(sqrt(L)) skip connections (floor(sqrt(L))-1 connections in case L is a 
            perfect square) in a postings list, and the length between two skips should be 
            round(sqrt(L), 0), where L = length of the postings list. """
        
        self.n_skips = math.floor(math.sqrt(self.length))
        if self.n_skips * self.n_skips == self.length:
            self.n_skips = self.n_skips - 1
        
        self.skip_length = round(math.sqrt(self.length)) # 13 -> 4
        skip_target = self.head
        current_node = self.head

        if self.length <= 2:
            return
        
        skips_added = 0
        while current_node and skips_added < self.n_skips:
            # Move `skip_target` forward by `skip_length` nodes
            for _ in range(self.skip_length):
                if skip_target and skip_target.next:
                    skip_target = skip_target.next
                else:
                    skip_target = None
                    break
            
            # Assign skip pointer and move to the next skip position
            current_node.skip = skip_target
            current_node = skip_target
            skips_added += 1

        # while skip_target:
        #     for _ in range(self.skip_length):
        #         if skip_target:
        #             skip_target = skip_target.next
        #         else:
        #             break
        #     current_node.skip = skip_target
            
        #     current_node = skip_target

        


    

    def append(self, value,tf_idf=None):
        """ Write logic to add new elements to the linked list.
            Insert the element at an appropriate position, such that elements to the left are lower than the inserted
            element, and elements to the right are greater than the inserted element.
            To be implemented. """
        # Check if the linked list is empty
        # if self.head is None:
        #     self.head = Node(value, tf_idf=tf_idf)
        #     self.end_node = self.head
        #     self.length += 1
        #     return

        # # Traverse the list to check for the document ID
        # current = self.head
        # while current:
        #     if current.value == value:
        #         return  # Document ID already exists, skip addition
        #     if current.next is None:
        #         break
        #     current = current.next
        
        # # Append new node if document ID is not found
        # new_node = Node(value, tf_idf=tf_idf)
        # current.next = new_node
        # self.end_node = new_node
        # self.length += 1

        ############################
        new_node = Node(value,tf_idf=tf_idf)
        if self.head is None:
            self.head = new_node
            self.end_node = new_node
        else:
            if self.end_node.value == value: # Check if the document ID already exists in the list
                return
            self.end_node.next = new_node
            self.end_node = new_node
        
        self.length += 1

    def merge_without_skip(self, ll2 : 'LinkedList') :
        """ Boolean AND merge of two linked lists without using skip pointers.
        Returns : LinknedList, Number of comparisons
             """
        result = LinkedList()
        p1 = self.head
        p2 = ll2.head
        comparisons = 0
        while p1 and p2:
            comparisons += 1
            if p1.value == p2.value:
                result.append(p1.value, p1.tf_idf)
                p1 = p1.next
                p2 = p2.next
            elif p1.value < p2.value:
                p1 = p1.next
            else:
                p2 = p2.next
            

        result.add_skip_connections() # Add Skip connections to the merged list

        return result, comparisons
    
    def merge_with_skip(self, ll2 : 'LinkedList') :
        result = LinkedList()
        p1 = self.head
        p2 = ll2.head
        comparisons = 0

        while p1 and p2:
            comparisons += 1
            
            if p1.value == p2.value:
                result.append(p1.value, p1.tf_idf)
                p1 = p1.next
                p2 = p2.next
            elif p1.value < p2.value:
                if p1.skip and p1.skip.value <= p2.value:
                    comparisons += 1  # Comparison before taking skip
                    # p1 = p1.skip
                    

                    while p1.skip and p1.skip.value <= p2.value:
                        comparisons += 1
                        p1 = p1.skip
                else:
                    p1 = p1.next
            else:
                if p2.skip and p2.skip.value <= p1.value:
                    comparisons += 1  # Comparison before taking skip
                    # p2 = p2.skip
                    while p2.skip and p2.skip.value <= p1.value:
                        comparisons += 1
                        p2 = p2.skip
                else:
                    p2 = p2.next

            

        result.add_skip_connections()

        return result, comparisons
        
        
    def to_list_tfidf_sorted(self):
        traversal = []
        if self.head is None:
            return []
        else:
            current_node = self.head
            while current_node:
                traversal.append((current_node.value, current_node.tf_idf))
                current_node = current_node.next
            
            traversal.sort(key=lambda x: x[1], reverse=True)
            traversal = [x[0] for x in traversal]
            return traversal

    def __str__(self) -> str:
        # For easier debugging
        traversal = []
        curr = self.head
        while curr:
            traversal.append((curr.value,curr.skip.value if curr.skip else None, curr.tf_idf))
            curr = curr.next
        return str ( " -> ".join(map(str, traversal)) )




class Preprocessor():
    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.PorterStemmer()
        self.tokens = set()
        self.tokens_stemmed = set()

    def _remove_accents(self,input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return nfkd_form

    def preprocess_text ( self, text ):
        # Lowercase
        text = text.lower()
        # Deaccent (remove accents) Convert  ß -> B
        text = self._remove_accents(text)
        # Convert brackets and hyphens to spaces
        # Remove Special Characters (Only keep alpha numeric characters) and replace with space
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)

        # Tokenize
        tokens = text.split()
        self.tokens.update(tokens)
        return text

    def preprocess_stopwords_stemming(self, text):
        # Remove stopwords
        text = ' '.join([word for word in text.split() if word not in self.stopwords])
        # Stemming
        text = ' '.join([self.stemmer.stem(word) for word in text.split()])
        
        # Tokenize
        tokens = text.split()
        self.tokens_stemmed.update(tokens)
        return text
    
    def processCorpus(self, docs):
        preprocessed_docs = {}
        for doc_id, doc_text in docs.items():
            preprocessed_docs[doc_id] = self.preprocess_stopwords_stemming(self.preprocess_text(doc_text))
        
        # Sort preprocessed_docs by key
        preprocessed_docs = OrderedDict(sorted(preprocessed_docs.items(), key=lambda x: x[0]))
        return preprocessed_docs
    
    def process_queries(self,queries):
        processed_queries = {}
        for query in queries:
            processed_queries[query] = self.preprocess_stopwords_stemming(self.preprocess_text(query))
        return processed_queries,self.tokens_stemmed

    def getTokens(self):
        return (self.tokens,self.tokens_stemmed)
    

class Indexer():
    def __init__(self,preprocessed_docs):
        self.inverted_index = OrderedDict({})
        self.tokens = set() 
        self.docs = preprocessed_docs
        self.build_inverted_index(preprocessed_docs)
        self.sort_inverted_index()
        self.add_skip_connections()
        self.compute_doc_tf_idf()

    # def build_inverted_index(self, docs):
    #     for doc_id, text in docs.items():
    #         for term in text.split():
    #             if term not in self.inverted_index:
    #                 self.inverted_index[term] = LinkedList()
    #             # Append only if the document ID is not already present in the term's postings list
    #             current_postings_list = self.inverted_index[term]
    #             if current_postings_list.head is None or doc_id not in current_postings_list.to_list():
    #                 current_postings_list.append(doc_id)


    def build_inverted_index(self, docs):
        for id, text in docs.items():
            
            for term in text.split():
                if term in self.inverted_index:
                    self.inverted_index[term].append(id)
                else:
                    self.inverted_index[term] = LinkedList()
                    self.inverted_index[term].append(id)


    def get_inverted_index(self):
        return self.inverted_index
    
    def sort_inverted_index(self):
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def compute_doc_tf_idf(self):
        """ Add tf-idf scores to each element in the postings list. The tf-idf should be 
            calculated using the following formulas:  
            Tf = (freq. of token in a doc after pre-processing / total tokens in the doc after 
            pre-processing)  
            Idf = (total num docs / length of postings list) 
            tf-idf = Tf*Idf"""
        
        for term,postings in self.inverted_index.items():

            idf = len(self.docs) / postings.length
            current = postings.head
            while current:
                doc_id = current.value
                doc = self.docs[doc_id]
                tf = doc.count(term) / len(doc.split())
                current.tf_idf = tf * idf
                current = current.next
        return self.inverted_index
    

    def add_skip_connections(self):
        for term,postings in self.inverted_index.items():
            postings.add_skip_connections()



######################### Helper Functions ################################


def get_query_postings(query_tokens, inverted_index):
    result = {}
    for token in query_tokens:
        if token not in inverted_index:
            result[token] = []
        else:
            postings = inverted_index[token].to_list()
            result[token] = postings

    return result

def get_query_postings_skips(query_tokens, inverted_index):
    result = {}
    for token in query_tokens:
        if token not in inverted_index:
            result[token] = []
            return result
        
        if inverted_index[token].length <= 2:
            result[token] = []
        else:
            postings = inverted_index[token].to_list_skip()
            result[token] = postings

    return result


def daat_and_tfidf(processed_queries, inverted_index):
    """Sort the output of DAAT without skip pointers (Part 2 step 3) using tf-idf scoring, 
    where the tf-idf should be calculated using the formula mentioned in Part 1 step 5. 
    The  retrieved  documents  &  number  of  comparisons  remain  the  same  as  DAAT 
    without skip pointers."""
    
    comparisons = 0

    response = {}

    for query,p_query in processed_queries.items():
        response[query] = {
            "num_comparisons": 0,
            "num_docs": 0,
            "results": []
        }
        
        query_terms = p_query.split()
        pl = [ inverted_index[token] if token in inverted_index else LinkedList() for token in query_terms]
        pl = sorted(pl, key=lambda x: x.length)                 # sort by length to minimize comparisons
        
        if len(pl) == 0:
            continue
        result = pl[0]
        n_comparisons = 0
        for i in range(1,len(pl)):
            comparisons += 1
            result, comparisons =  result.merge_without_skip(pl[i])
            n_comparisons += comparisons

        # Sort by tf-idf
        result = result.to_list_tfidf_sorted()
        
        response[query]["results"] = result
        response[query]["num_comparisons"] = n_comparisons
        response[query]["num_docs"] = len(result)
    
    return response


def daat_and_skip_tfidf(processed_queries, inverted_index):
    """Implement the DAAT AND algorithm with skip pointers and 
    tf-idf scoring as mentioned in Part 2 step 4."""
    
    comparisons = 0

    response = {}

    for query,p_query in processed_queries.items():
        response[query] = {
            "num_comparisons": 0,
            "num_docs": 0,
            "results": []
        }
        
        query_terms = p_query.split()
        pl = [ inverted_index[token] if token in inverted_index else LinkedList() for token in query_terms]
        pl = sorted(pl, key=lambda x: x.length)                 # sort by length to minimize comparisons
        
        if len(pl) == 0:
            continue
        result = pl[0]
        n_comparisons = 0
        for i in range(1,len(pl)):
            comparisons += 1
            result, comparisons =  result.merge_with_skip(pl[i])
            n_comparisons += comparisons

        # Sort by tf-idf
        result = result.to_list_tfidf_sorted()
        
        response[query]["results"] = result
        response[query]["num_comparisons"] = n_comparisons
        response[query]["num_docs"] = len(result)
    
    return response

def daat_and(processed_queries, inverted_index):
# Use merge algorithm and  return  a  sorted  list  of  document  ids,  along  with  the  number  of comparisons made
# Determine 

    response = {}
    
    for query,p_query in processed_queries.items():
        response[query] = {
            "num_comparisons": 0,
            "num_docs": 0,
            "results": []
        }
        
        query_terms = p_query.split()
        pl = [ inverted_index[token] if token in inverted_index else LinkedList() for token in query_terms]
        pl = sorted(pl, key=lambda x: x.length)  # Sort by length to minimize comparisons
        
        # Sort postings lists by length to minimize comparisons
        total_comparisons = 0
        pl.sort(key=lambda x: x.length)
        result = pl.pop(0)
        
        # Dynamically choose the smallest remaining list at each step
        while pl:
            # Sort remaining lists by current length to choose the shortest for next merge
            pl.sort(key=lambda x: x.length)
            next_list = pl.pop(0)
            
            # Perform merge and accumulate comparisons
            result, comparisons = result.merge_without_skip(next_list)
            total_comparisons += comparisons
        
        # Convert final result to list and add skip pointers
        result_list = result.to_list()
        result.add_skip_connections()  # Add skip pointers once after all merges
        
        # Populate response
        response[query]["results"] = result_list
        response[query]["num_comparisons"] = total_comparisons
        response[query]["num_docs"] = len(result_list)
    
    return response



def daat_and_with_skip(processed_queries, inverted_index):
# Use merge algorithm and  return  a  sorted  list  of  document  ids,  along  with  the  number  of comparisons made
# Determine 
    comparisons = 0

    response = {}

    for query,p_query in processed_queries.items():
        response[query] = {
            "num_comparisons": 0,
            "num_docs": 0,
            "results": []
        }
        
        query_terms = p_query.split()
        pl = [ inverted_index[token] if token in inverted_index else LinkedList() for token in query_terms]
        pl = sorted(pl, key=lambda x: x.length)                 # sort by length to minimize comparisons
        
        if len(pl) == 0:
            continue
        result = pl[0]
        n_comparisons = 0
        for i in range(1,len(pl)):
            comparisons += 1
            result, comparisons =  result.merge_with_skip(pl[i])
            n_comparisons += comparisons
        result = result.to_list()
        response[query]["results"] = result
        response[query]["num_comparisons"] = n_comparisons
        response[query]["num_docs"] = len(result)
    return response



            
if "__main__" == __name__:
    pass
    # # Test Preprocessor
    # preprocessor = Preprocessor()
    # text = "Hello! This
    # Load the documents in a dictionary

    docs = OrderedDict()
    with open("input_corpus.txt") as f:
        documents = f.readlines()
        # documents = sorted(documents, key=lambda x: int(x.strip().split("\t")[0]))


    for i, doc in enumerate(documents):
        doc = doc.strip().split("\t")
        docs[int(doc[0])] = doc[1]

    corpusPreprocessor = Preprocessor()
    preprocessed_docs = corpusPreprocessor.processCorpus(docs)
    tokens, tokens_stemmed = corpusPreprocessor.getTokens()

    indexer = Indexer(preprocessed_docs)
    inverted_index = indexer.get_inverted_index()

    queries = [
        "the novel coronavirus",
        "from an epidemic to a pandemic",
        "is hydroxychloroquine effective?"
    ]

    # Query Processor

    queryProcessor = Preprocessor()

    processed_queries, query_tokens = queryProcessor.process_queries(queries)


    