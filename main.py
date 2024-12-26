from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from loguru import logger
import sys
from sentence_transformers import SentenceTransformer, util
from test_groq import expand_query,write_argument

logger.remove()  
logger.add(sys.stdout, level="INFO")

app = FastAPI()

# model = SentenceTransformer('all-MiniLM-L6-v2')  

ARXIV_API_URL = "http://export.arxiv.org/api/query"

class ArgumentRequest(BaseModel):
    argument: str

def query_arxiv(search_query: str, max_results: int = 10):
    logger.info(f"Querying arXiv with search query: {search_query}")
    
    words = search_query.split()
    refined_query = " AND ".join([f"abs:{word}" for word in words])

    params = {
        "search_query": refined_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params)
        response.raise_for_status()  
        logger.info(f"arXiv query successful, received {len(response.text)} characters of data.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error while querying arXiv: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch data from arXiv")

    return response.text

def parse_arxiv_response(response):
    logger.info("Parsing arXiv API response...")
    from xml.etree import ElementTree
    root = ElementTree.fromstring(response)
    entries = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        link = entry.find("{http://www.w3.org/2005/Atom}id").text
        entries.append({"title": title, "summary": summary, "link": link})

    logger.info(f"Parsed {len(entries)} entries from the response.")
    return entries

@app.post("/analyze")
async def analyze_argument(request: ArgumentRequest):
    logger.info(f"Received request to analyze argument: {request.argument}")

    expanded_queries = expand_query(request.argument)
    logger.info(f"Expanded queries: {expanded_queries}")
    
    all_abstracts = []

    for query in expanded_queries:
        logger.info(f"Processing expanded query: {query}")
        raw_data = query_arxiv(query)
        abstracts = parse_arxiv_response(raw_data)
        
        if not abstracts:
            logger.warning(f"No abstracts found for query: {query}")
            continue  
        # logger.info(f"Calculating semantic similarity for {len(abstracts)} abstracts...")
        # argument_embedding = model.encode(request.argument, convert_to_tensor=True)
        # for abstract in abstracts:
        #     abstract_embedding = model.encode(abstract["summary"], convert_to_tensor=True)
        #     abstract["similarity"] = util.pytorch_cos_sim(argument_embedding, abstract_embedding).item()
        # logger.info("Semantic similarity calculation completed.")

       
        all_abstracts.extend(abstracts)

    if not all_abstracts:
        logger.info("No relevant articles found.")
        return {"message": "No relevant articles found"}

    logger.info(f"Sorting {len(all_abstracts)} abstracts by similarity score...")
    sorted_abstracts = sorted(all_abstracts, key=lambda x: x.get("similarity", 0), reverse=True)

    logger.info("Returning the sorted abstracts.")
    res=write_argument(sorted_abstracts,request.argument)
    return {
        "argument": request.argument,
        "results": res
    }
