import os
import re
from groq import Groq
from typing import List

# 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
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)



def clean_queries(queries: List[str]) -> List[str]:

    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
        'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
        'that', 'the', 'to', 'was', 'were', 'will', 'with',
        'more', 'also', 'such', 'then', 'than', 'this'
    }
    
    def clean_single_query(query: str) -> str:
        query = query.lower()
        
        words = query.split()
        
        cleaned_words = [
            word for word in words 
            if word not in stop_words and re.match(r'^[a-z]+$', word)
        ]
        
        return ' '.join(cleaned_words)
    
    cleaned_queries = [clean_single_query(re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", query)) for query in queries]
    
    seen = set()
    final_queries = []
    for query in cleaned_queries:
        if query and query not in seen:
            final_queries.append(query)
            seen.add(query)
    
    return final_queries




def parse_expanded_queries(response: str)->List[str]:
    queries = []
    for line in response.split("\n"):
        if line.strip() and line[0].isdigit() and ". " in line:
            try:
                queries.append(line.split(". ", 1)[1].strip())
            except IndexError:
                continue
    cleaned_queries =clean_queries(queries)
    
    return cleaned_queries


def expand_query(q: str):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": (
                    f"I have a search query: '{q}'. "
"Your task is to expand this query into three alternative search queries that are suitable for finding academic articles or research papers. "
"Each query should be brief, to the point, and use different phrasing or synonyms. "
"Please format your response in the following way:\n"
"1. [First query]\n"
"2. [Second query]\n"
"3. [Third query]"
                ),
            }
        ],
        model="llama3-8b-8192",
    )

    parsed_queries = parse_expanded_queries(chat_completion.choices[0].message.content)
    return parsed_queries


def write_argument(research_articles: List[dict],argument) -> str:
    # Prepare the input data for Groq
    article_info = "\n".join([
        f"Title: {article['title']}\nSummary: {article['summary']}\nLink: {article['link']}"
        for article in research_articles
    ])
    
    # Create the prompt for Groq API
    prompt = (
        f"Using the following academic papers, write a comprehensive argument on the topic of {argument}. "
        "The argument should draw from the papers provided below and reference them in a logical, coherent way.Dont use any other papers other than the ones i provided\n\n"
        "Articles:\n"
        f"{article_info}\n\n"
        "Argument:"
    )
    
    # Call the Groq client to generate the argument
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt
        }],
        model="llama3-8b-8192"
    )

    # Return the generated argument
    return chat_completion.choices[0].message.content


