#!/usr/bin/python3
# pip3 install wikipedia-api
import os
import random
import requests
import wikipediaapi
import numpy as np
import pandas as pd
import threading
import argparse

# probabilities: S -> 0.294; T,E,M -> 0.235
STEM_WEIGHTS = [1.25, 1, 1, 1]

STEM = {
    "S": ["Category:Applied_sciences", "Category:Biotechnology", "Category:Biology", "Category:Natural_history"],
    "T": [
        "Category:Technology_strategy", "Category:Technical_specifications", "Category:Technology_assessment",
        "Category:Technology_hazards", "Category:Technology_systems", "Category:Hypothetical_technology",
        "Category:Mobile_technology", "Category:Obsolete_technologies", "Category:Philosophy_of_technology",
        "Category:Real-time_technology", "Category:Software", "Category:Technology_development",
        "Category:Computing", "Category:Artificial_objects", "Category:Technological_change",
        "Category:Technical_communication", "Category:Technological_comparisons"
    ],
    "E": ["Category:Engineering_disciplines", "Category:Engineering_concepts", "Category:Industrial_equipment", "Category:Manufacturing"],
    "M": ["Category:Fields_of_mathematics", "Category:Physical_sciences"]
}

EXCLUDE_CATEGORIES = set([
    "Category:Technology", "Category:Mathematics", "Category:Works about technology",
    "Category:Technology evangelism", "Category:Artificial objects", "Category:Fictional physical scientists"
])

def split_category_members(members):
    category_list, page_list= [], []

    for member_name, member_page in members:
        if member_name.startswith('Category') and member_name not in EXCLUDE_CATEGORIES:
            category_list.append((member_name, member_page))
        else:
            page_list.append((member_name, member_page))

    return category_list, page_list

wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
def get_wiki_random_page(deep_subcategories=True):
    stem_label, stem_categories = random.choices(list(STEM.items()), weights=STEM_WEIGHTS, k=1)[0]
    category = random.choice(stem_categories)
    category_page = wiki_wiki.page(category)
    while True:
        chosen_list = list(category_page.categorymembers.items())
        if deep_subcategories:
            category_list, page_list = split_category_members(chosen_list)
            chosen_list = []
        else:
            category_list, page_list = [], []

        # 50% change to select category or page list if one of them isn't empty
        # helps to go deeper into subcategories because there're more pages than categories
        if not (category_list or page_list) and not chosen_list:
            continue
        elif not category_list:
            chosen_list = page_list
        elif not page_list:
            chosen_list = category_list
        else:
            chosen_list = random.choice([category_list, page_list])

        # select random page from chosen list
        selected_page_name, selected_page = random.choice(chosen_list)

        if not selected_page_name.startswith("Category"):
            break

        category_page = selected_page

    return selected_page, stem_label

def get_wiki_text(seen_pages, min_page_length=6, sentences_include=3):
    while True:
      try:
        wiki_page, stem_label = get_wiki_random_page()

        if wiki_page.pageid in seen_pages:
            continue

        page_sentences = wiki_page.text.split(". ")

        # check is the page is long enought
        if len(page_sentences) >= min_page_length:
            # main information about the topic usualy described within first 3 sentences
            wiki_text = ". ".join(page_sentences[:sentences_include]) + "."
            break
      except Exception as e:
        print(f'Ex: {e}')

    return wiki_text, wiki_page.pageid, wiki_page.title, stem_label

from tqdm import tqdm
class Worker(threading.Thread):
  def __init__(self, id, n_page, seen_pages, path):
    super().__init__()
    self.id = id
    self.n_page = n_page
    self.min_page_length=6
    self.sentences_include=5
    self.seen_pages = seen_pages

  def run(self):
    cp = 300
    pages = []
    for i in tqdm(range(self.n_page)):
        try:
            page = get_wiki_text(self.seen_pages,self.min_page_length,self.sentences_include)
            pages.append(page)
            self.seen_pages.append(page[1])
        except Exception as e:
            print(f'Ex: {e}')
        if (i+1) % cp == 0:
            df = pd.DataFrame(pages,columns=['text', 'pageid','title', 'stem_label'])
            df.to_csv(os.path.join(path, f'r_{self.id}_{i}.csv'), index=False)
            print(f'Finish {self.id}: {i}, saving to file')
            pages = []

def fetch(path, resume):
    n_worker = 1
    n_page = 50000
    seen_pages = []

    if resume:
        df = pd.read_csv(os.path.join(path,'merged'))
        n_page -= len(df)
        seen_pages += df['pageid'].tolist()
        assert len(df) == len(seen_pages)
        print(f'Resume fetching. Adding {len(df)} records')

    workers = [Worker(i,n_page,seen_pages,path) for i in range(n_worker)]
    for w in workers:
      w.start()
    for w in workers:
      w.join()

def merge(path):
    files = [f for _,_,files in os.walk(path) for f in files]
    dfs = [pd.read_csv(os.path.join(path, file)) for file in files]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(path,'merged'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str)
    parser.add_argument('--path', '-p', type=str)
    parser.add_argument('--resume', '-r', action='store_true')
    args = parser.parse_args()
    mode = args.mode
    path = args.path
    resume = args.resume

    if mode == 'fetch':
        fetch(path, resume)
    elif mode == 'merge':
        merge(path)
