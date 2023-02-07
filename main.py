"""
1) Initialize
2) Generate an index of tuples with ({language}, {country}, {topic})
3) Generate a dictionary with topics as keys and {sources} as user provided values
4) Sources should be a string that is comma separated.
5) Use langchain agent to learn how to query the newsapi.
6) Using the agent in step 5) generate a Q&A chat model that answers questions in {language} about {topic} news in {country} using {sources}.
"""
from typing import *
import os
from langchain.agents import initialize_agent, Tool
from newsapi import NewsApiClient
from fastapi import FastAPI

# initialize
app = FastAPI()
newsapi = NewsApiClient(api_key=os.env['NEWS_API_KEY'])


def load_topics_from_file(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

topics = load_topics_from_file('topics.txt')

def load_sources_from_file(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

sources = load_sources_from_file('sources.txt')

def load_countries_from_file(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

countries = load_countries_from_file('countries.txt')

def load_languages_from_file(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

languages = load_languages_from_file('languages.txt')

def generate_index(topics: List[str], sources: List[str], countries: List[str], languages: List[str]) -> List[Tuple[str, str, str]]:
    return [(language, country, topic) for language in languages for country in countries for topic in topics]

index = generate_index(topics, sources, countries, languages)

def generate_dictionary(index: List[Tuple[str, str, str]]) -> Dict[str, str]:
    return {topic: ','.join(sources) for language, country, topic in index}

dictionary = generate_dictionary(index)

def generate_agent(dictionary: Dict[str, str]) -> Tool:
    return initialize_agent(dictionary)

agent = generate_agent(dictionary)

def generate_model(agent: Tool, index: List[Tuple[str, str, str]]) -> None:
    for language, country, topic in index:
        agent.learn(language, country, topic, newsapi.get_everything(q=topic, sources=dictionary[topic], language=language, country=country))


@app.get('/topics')
def get_topics():
    return topics

@app.get('/sources')
def get_sources():
    return sources

@app.get('/countries')
def get_countries():
    return countries

@app.get('/languages')
def get_languages():
    return languages

@app.get('/chat')
def get_chat(language: str, country: str, topic: str):
    return agent.ask(language, country, topic)
