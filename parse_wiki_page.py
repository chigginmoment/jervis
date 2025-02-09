import requests
from bs4 import BeautifulSoup

def page_parse(link="Tank:F38_Bat_Chatillon155_58"):
    """Downloads a page from the Wargaming wiki and formats it into only the important text-based parts"""
    try:
        ctx_string = ""

        api_url = "https://wiki.wargaming.net/api.php"
        params = {
            "action": "parse",
            "page": f"{link}",
            "format": "json",
            "prop": "text"
        }

        response = requests.get(api_url, params=params)
        data = response.json()

        html_content = data["parse"]["text"]["*"]

        soup = BeautifulSoup(html_content, "html.parser")

        # get rid of the tooltips, don't care
        tooltip_sections = soup.find_all("div", class_="wiki_tooltip")
        for tooltip in tooltip_sections:
            tooltip.decompose()

        # get rid of stock, don't care
        stock_sections = soup.find_all("span", class_="stock")
        for stock in stock_sections:
            stock.decompose()

        description = soup.find("div", class_="b-description")
        ctx_string += description.get_text(separator=" ", strip=True) + "\n"

        performance_section = soup.find("div", class_="b-performance")

        for subsection in performance_section.find_all("div", class_="b-performance_text"):
            row_entries = subsection.find_all("td")
            for row in row_entries:
                metric = row.find("span", class_="t-performance_left")
                value = row.find("span", class_="t-performance_right")
                ctx_string += metric.get_text() + value.get_text(separator=" ", strip=True) + "\n"
                
        span = soup.find('span', id="Pros_and_Cons")

        parent_div = span.find_parent("div")
        ctx_string += parent_div.get_text(separator=" ", strip=True) + "\n"

        return ctx_string
    except Exception as e:
        print(e)
        return("Error in fetching page")

if __name__ == "__main__":
    with open("tank_names.txt") as file:
        for line in file:
            with open("pages/" + line.replace("Tank:", "").strip(), 'w', encoding="utf-8") as tank_file:
                tank_file.write(page_parse(line))
                print("Wrote to", "pages/" + line.replace("Tank:", "").strip())
