from pathlib import Path
from user_agents import parse as parse_ua
import requests
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas()

def load_ja4plus_json(json_file_path) -> pd.DataFrame:
    return pd.read_json(json_file_path)

def load_crawler_list():
    url = 'https://raw.githubusercontent.com/monperrus/crawler-user-agents/master/crawler-user-agents.json'
    crawlers_df = pd.read_json(url)
    return crawlers_df.to_dict(orient='records')

def load_bad_bot_list():
    response = requests.get('https://raw.githubusercontent.com/mitchellkrogza/nginx-ultimate-bad-bot-blocker/master/_generator_lists/bad-user-agents.list')
    response.raise_for_status()
    bad_bots = response.text.splitlines()

    exclude_list = {
        'ChatGPT-User',
        'openai.com',
        'Netcraft',
        'MicroMessenger',
        'Bytespider',
        'GPTBot',
        'MJ12bot',
        'MQQBrowser'
    }
    bad_bots = [bot for bot in bad_bots if bot not in exclude_list]
     
    return bad_bots

def is_known_crawler(df,row_index):
    df = df.copy()

    df['is_known_crawler'] = False
    df['crawler_match'] = None

    crawlers = load_crawler_list()

    if 'is_known_bad_bot' in df.columns and 'is_known_browser' in df.columns:
        mask = (df['is_known_bad_bot'] == False) & (df['is_known_browser'] == False)
    
    if mask.any():
        first_indices = df[mask].index[:row_index]
        mask = df.index.isin(first_indices)

    def _check_row(ua_string):
        ua_lower = ua_string.lower()
        for crawler in crawlers:
            instances = crawler.get("instances") or []
            for pattern in instances:
                try:
                    if re.search(pattern, ua_string, re.IGNORECASE):
                        return True, crawler.get("pattern")
                except re.error:
                    continue

            pat = crawler.get("pattern")
            if pat:
                try:
                    if re.search(pat, ua_string, re.IGNORECASE):
                        return True, pat
                except re.error:
                    if pat.lower() in ua_lower:
                        return True, pat
        return False, None
    
    if mask.any():
        print('Processing known crawler...')
        results = df.loc[mask, 'user_agent_string'].progress_apply(_check_row)
        df.loc[mask, 'is_known_crawler'] = [r[0] for r in results]
        df.loc[mask, 'crawler_match'] = [r[1] for r in results]
        df.loc[df['is_known_crawler'] == True, 'bad_origin'] = 0

    return df

def is_known_browser(df,row_index):
   df = df.copy()

   df['is_known_browser'] = False
   df['browser_match'] = None
   
   if 'is_known_bad_bot' in df.columns:
       mask = df['is_known_bad_bot'] == False
       

   if mask.any():
       first_indices = df[mask].index[:row_index]
       mask = df.index.isin(first_indices)

   def _check_row(ua_string):
    parsed = parse_ua(ua_string)
    if parsed.is_pc or parsed.is_mobile or parsed.is_tablet:
        if not parsed.is_bot:
            return True, parsed.browser.family
    return False, None
   
   if mask.any():
    print('Processing known browsers...')
    results = df.loc[mask, 'user_agent_string'].progress_apply(_check_row)
    df.loc[mask, 'is_known_browser'] = [r[0] for r in results]
    df.loc[mask, 'browser_match'] = [r[1] for r in results]
    df.loc[df['is_known_browser'] == True, 'bad_origin'] = 0    
   return df

def is_known_bad_bot(df):
    df = df.copy()

    df['is_known_bad_bot'] = False
    df['bad_bot_match'] = None
    df['bad_origin'] = None

    bad_bots = load_bad_bot_list()
    bad_bots_paired = [(bot.lower(), bot) for bot in bad_bots]

    # if 'is_known_crawler' in df.columns and 'is_known_browser' in df.columns:
    #     mask = (df['is_known_crawler'] == False) & (df['is_known_browser'] == False)

    def check_ua(ua):
        ua_lower = str(ua).lower()
        for bot_lower, bot_orig in bad_bots_paired:
            if bot_lower in ua_lower:
                return True, bot_orig
        return False, None   
    
    print('Processing known bad bots...')
    results = df['user_agent_string'].progress_apply(check_ua)
    df['is_known_bad_bot'] = [r[0] for r in results]
    df['bad_bot_match'] = [r[1] for r in results]
    df.loc[df['is_known_bad_bot'] == True, 'bad_origin'] = 1

    return df

def main():
    INPUT_PATH = Path('data/test/ja4+_db.json')
    OUTPUT_PATH = Path('data/test/ja4+_labeled.csv')

    df = load_ja4plus_json(INPUT_PATH)
    pd.options.display.max_colwidth = 300
    df = df.dropna(subset=['ja4_fingerprint', 'ja4h_fingerprint'])
    df = df.drop_duplicates(subset=['ja4_fingerprint', 'ja4h_fingerprint'])
    df = df.dropna(subset=['user_agent_string'])
    
    df = is_known_bad_bot(df)
    df= is_known_browser(df,250)
    df = is_known_crawler(df,500) 

    print(df['bad_bot_match'].value_counts())

    print('Total bad bots:',len(df[df['is_known_bad_bot'] == True]))
    print('Unknown:',len(df[(df['is_known_bad_bot'] == False)&(df['is_known_crawler'] == False) & (df['is_known_browser'] == False)]))

    df.dropna(subset=['bad_origin'], inplace=True)
    df = df[['ja4_fingerprint','ja4h_fingerprint','bad_origin']]
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == '__main__':
    main()
