import requests

from bs4 import BeautifulSoup

nations = ["Sweden", "Poland", "Germany", "U.S.S.R.", "U.S.A.", "China", "France", "U.K.", "Japan", "Czechoslovakia", "Italy"]

with open("tank_names.txt", "w") as f:
    for nation in nations:
        api_url = "https://wiki.wargaming.net/api.php"
        params = {
            "action": "parse",
            "page": f"Tank:{nation}",
            "format": "json",
            "prop": "text"
        }

        # Fetch the HTML content
        response = requests.get(api_url, params=params)
        data = response.json()

        html_content = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html_content, 'html.parser')

        # because all the tank links on the wiki page start with Tank:
        tank_links = soup.find_all('a', href=lambda href: href and href.startswith('/en/Tank:'))

        # Print the links
        for link in tank_links:
            # print(link['href'].replace('/en/', ''))
            f.write(link['href'].replace('/en/', '') + "\n")
print("Write complete.")