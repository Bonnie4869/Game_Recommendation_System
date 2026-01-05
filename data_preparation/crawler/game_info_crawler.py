import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import random
import pandas as pd

# -----------------------------
# config：set the range of rows to crawl
# -----------------------------
start_row = 20000  # start row index
end_row = 32747  # end row index (exclusive)

# -----------------------------
# initialize database
# -----------------------------
conn = sqlite3.connect("steam_dataset.db")
cur = conn.cursor()

cur.execute(
    """
CREATE TABLE IF NOT EXISTS steam_games (
    app_id INTEGER PRIMARY KEY,
    title TEXT,
    genres TEXT,
    popular_tags TEXT,
    about TEXT,
    high_stakes TEXT
)
"""
)
conn.commit()


# -----------------------------
# check if app_id exists in database
# -----------------------------
def exists_in_db(app_id):
    cur.execute("SELECT 1 FROM steam_games WHERE app_id = ?", (app_id,))
    return cur.fetchone() is not None


# -----------------------------
#   fetch game info from steam store page
# -----------------------------
def fetch_steam_page(app_id):
    url = f"https://store.steampowered.com/app/{app_id}/"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        r = requests.get(url, headers=headers, timeout=15)
    except Exception as e:
        print(f"Request failed for {app_id}: {e}")
        return None

    if r.status_code != 200:
        print(f"Failed to fetch {app_id}: Status code {r.status_code}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # ---- Title ----
    title_block = soup.select_one("#appHubAppName")
    title = title_block.get_text(strip=True) if title_block else ""

    # ---- Genres ----
    genre_block = soup.select_one("#genresAndManufacturer span")
    genres = (
        [g.get_text(strip=True) for g in genre_block.find_all("a")]
        if genre_block
        else []
    )

    # ---- About ----
    about_block = soup.select_one("#game_area_description")

    def clean(tag):
        if not tag:
            return ""
        for br in tag.find_all("br"):
            br.replace_with("\n")
        return tag.get_text("\n", strip=True)

    about_text = clean(about_block)

    # ---- High Stakes ----
    high_h2 = about_block.find("h2", class_="bb_tag") if about_block else None
    high_stakes = ""
    if high_h2:
        nxt = high_h2.next_sibling
        while nxt:
            txt = ""
            if isinstance(nxt, str):
                txt = nxt.strip()
            elif nxt.name not in ["h2", "br"]:
                txt = nxt.get_text(" ", strip=True)
            if txt:
                high_stakes = txt
                break
            nxt = nxt.next_sibling

    # ---- Popular Tags ----
    tag_block = soup.select_one(".glance_tags.popular_tags")
    popular_tags = (
        [a.get_text(strip=True) for a in tag_block.select("a.app_tag")]
        if tag_block
        else []
    )

    return {
        "app_id": app_id,
        "title": title,
        "genres": "|".join(genres),
        "popular_tags": "|".join(popular_tags),
        "about": about_text,
        "high_stakes": high_stakes,
    }


# -----------------------------
#   save game info to database
# -----------------------------
def save_to_db(data):
    if data is None:
        return
    cur.execute(
        """
    INSERT OR REPLACE INTO steam_games(app_id, title, genres, popular_tags, about, high_stakes)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            data["app_id"],
            data["title"],
            data["genres"],
            data["popular_tags"],
            data["about"],
            data["high_stakes"],
        ),
    )
    conn.commit()


# -----------------------------
#   main crawler function
# -----------------------------
app_ids_all = pd.read_csv("unique_appids_2022.csv")["app_id"].tolist()
app_ids = app_ids_all[start_row:end_row]
failed_ids = []

for cnt, app_id in enumerate(app_ids, start=start_row + 1):
    if exists_in_db(app_id):
        print(f"{cnt}/{len(app_ids_all)}: App ID {app_id} already in DB, skipping.")
        continue

    print(f"{cnt}/{len(app_ids_all)}: Processing App ID = {app_id}")
    data = fetch_steam_page(app_id)

    if data is None:
        failed_ids.append(app_id)
        print(f"Failed to fetch {app_id}.")
    else:
        save_to_db(data)
        print(f"Saved {app_id}")

    time.sleep(random.uniform(0.5, 0.8))  # random delay to avoid overloading the server

# -----------------------------
#   save failed app_ids to csv
# -----------------------------
if failed_ids:
    pd.DataFrame({"app_id": failed_ids}).to_csv("failed_appids.csv", index=False)
    print(f"Saved {len(failed_ids)} failed app_ids to failed_appids.csv")

conn.close()
