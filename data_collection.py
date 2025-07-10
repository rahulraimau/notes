import requests
from bs4 import BeautifulSoup
import pandas as pd

def collect_from_api(api_url):
    try:
        response = requests.get(api_url)
        data = response.json()
        df = pd.DataFrame(data)
        print("API data collected successfully")
        return df
    except Exception as e:
        print(f"Error collecting API data: {e}")
        return None

def scrape_web(url, tag, class_name):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        elements = soup.find_all(tag, class_=class_name)
        data = [element.text for element in elements]
        df = pd.DataFrame(data, columns=["scraped_data"])
        print("Web data scraped successfully")
        return df
    except Exception as e:
        print(f"Error scraping web data: {e}")
        return None

# Example usage
if __name__ == "__main__":
    api_url = "https://api.example.com/data"
    web_url = "https://example.com"
    df_api = collect_from_api(api_url)
    df_web = scrape_web(web_url, tag="div", class_name="content")
    if df_api is not None:
        df_api.to_csv("api_data.csv", index=False)
    if df_web is not None:
        df_web.to_csv("web_data.csv", index=False)