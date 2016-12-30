# general text tools

import pandas as pd
from xml.dom.minidom import parseString
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words, fetch_url

def summarize_url(url, **kwargs):
    html = fetch_url(url)
    return summarize_html(html, **kwargs)

def summarize_html(html,**kwargs):
    return summarize(html, Parser=HtmlParser, **kwargs)

def summarize_file(path, **kwargs):
    text = open(path).read()
    return summarize(text, **kwargs)

def summarize(text, n=10, lang="english", Parser=PlaintextParser, Summarizer=LexRankSummarizer, fill='\n\n'):
    tokenizer = Tokenizer(lang)
    parser = Parser(text,tokenizer)
    stemmer = Stemmer(lang)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(lang)

    sentences = [str(s) for s in summarizer(parser.document,n)]
    if fill:
        return fill.join(sentences)
    else:
        return sentences

def print_xml(src, indent='    '):
    return parseString(src).toprettyxml(indent=indent)

def parse_wiki(src, strip=False):
    df = pd.read_html(src)[0]
    if strip:
        df = df.applymap(lambda s: s.split('♠')[-1]) # for sortkey spans in wikitables
    return df
