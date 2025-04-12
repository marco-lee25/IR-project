import requests
import xml.etree.ElementTree as ET


def fetch_arxiv_results(query, categories=None, max_results=10):
    """
    Query arXiv API and return parsed results.
    """
    base_url = "http://export.arxiv.org/api/query"

    # Prepare category filter
    if categories:
        cat_filter = "+OR+".join([f"cat:{cat}" for cat in categories])
        search_query = f"({cat_filter})+AND+all:{query.replace(' ', '+')}"
    else:
        search_query = f"all:{query.replace(' ', '+')}"

    # Compose full URL
    url = f"{base_url}?search_query={search_query}&start=0&max_results={max_results}&sortBy=relevance"

    # Make request
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}")

    return parse_arxiv_feed(response.text)


def parse_arxiv_feed(feed):
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(feed)
    results = []

    for entry in root.findall('atom:entry', ns):
        arxiv_id = entry.find('atom:id', ns).text.strip().split('/')[-1]
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        abstract = entry.find('atom:summary', ns).text.strip().replace('\n',
                                                                       ' ')
        results.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract
        })
    return results


if __name__ == "__main__":
    categories = ["cs.AI", "cs.LG", "cs.CL", "cs.IR"]
    results = fetch_arxiv_results("transformer models", categories,
                                  max_results=10)
    for paper in results:
        print(f"ID: {paper['arxiv_id']}\nTitle: {paper['title']}\n")
