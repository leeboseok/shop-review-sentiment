from typing import List
from collections import Counter
from konlpy.tag import Okt

okt = Okt()
stopwords = {'상품', '쇼핑몰', '정말', '도대체', '이거', '그냥', '너무', '완전', '진짜', '때문'}

def clean_and_tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = okt.pos(text, stem=True)
    keywords = [
        word for word, tag in tokens
        if tag in ('Noun', 'Adjective') and word not in stopwords and len(word) > 1
    ]
    return keywords

def extract_top_keywords(reviews: List[str], limit: int = 3):
    all_keywords = []
    for review_text in reviews:
        keywords = clean_and_tokenize(review_text)
        all_keywords.extend(keywords)
    
    keyword_counts = Counter(all_keywords)
    return keyword_counts.most_common(limit)
