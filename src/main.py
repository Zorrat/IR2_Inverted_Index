from fastapi import FastAPI
import uvicorn
from collections import OrderedDict
from utils import get_query_postings, daat_and, daat_and_with_skip, daat_and_tfidf, daat_and_skip_tfidf,get_query_postings_skips
from utils import Indexer,Preprocessor
import time

app = FastAPI()


@app.post("/execute_query")
def execute_query(queries: dict):
    
    queries = queries['queries']

    response = {
    }
    st = time.time()

    inverted_index = indexer.get_inverted_index()

    # Query Processor
    queryProcessor = Preprocessor()
    processed_queries, query_tokens = queryProcessor.process_queries(queries)

    # Get Postings
    response["postingsList" ] = get_query_postings(query_tokens, inverted_index)
    response["daatAnd"] = daat_and(processed_queries, inverted_index)

    response['postingsListSkip'] = get_query_postings_skips(query_tokens, inverted_index)
    response['daatAndSkip'] = daat_and_with_skip(processed_queries, inverted_index)

    response["daatAndTfIdf"] = daat_and_tfidf(processed_queries, inverted_index)
    response["daatAndSkipTfIdf"] = daat_and_skip_tfidf(processed_queries, inverted_index)

    r = {
        "Response": response,
        "time_taken": str(time.time() - st)
    }

    return r


if __name__ == "__main__":

    docs = OrderedDict()
    with open("input_corpus.txt") as f:
        documents = f.readlines()

    for i, doc in enumerate(documents):
        doc = doc.strip().split("\t")
        docs[int(doc[0])] = doc[1]

    corpusPreprocessor = Preprocessor()
    preprocessed_docs = corpusPreprocessor.processCorpus(docs)
    tokens, tokens_stemmed = corpusPreprocessor.getTokens()

    indexer = Indexer(preprocessed_docs)
    inverted_index = indexer.get_inverted_index()


    uvicorn.run(app, host="0.0.0.0", port=9999,)