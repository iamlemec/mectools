# general text tools

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words, fetch_url

def summarize_url(url,**kwargs):
    html = fetch_url(url)
    return summarize_html(html,**kwargs)

def summarize_html(html,**kwargs):
    return summarize(html,Parser=HtmlParser,**kwargs)

def summarize_file(path,**kwargs):
    text = open(path).read()
    return summarize(text,**kwargs)

def summarize(text,n=10,lang="english",Parser=PlaintextParser,Summarizer=LsaSummarizer,fill='\n\n'):
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
