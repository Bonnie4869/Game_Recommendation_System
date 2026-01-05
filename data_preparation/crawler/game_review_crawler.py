import requests
from bs4 import BeautifulSoup
import json
import sqlite3
import time
import random
import pandas as pd
import os

# -----------------------------
# config: set the range of app_ids to crawl
# -----------------------------
start_row = 0
end_row = 32747

# config
INPUT_CSV_PATH = "dataset_2022/unique_appids_2022.csv"
DB_NAME = "steam_review_merged.db"
LANGUAGE = "english"
initial_cursor = "*"
BASE_URL = "https://store.steampowered.com/appreviews/{}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# -----------------------------
# manage db
# -----------------------------


def setup_database():
    """create db"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS steam_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_id TEXT NOT NULL,
            username TEXT,
            recommendation TEXT,
            review_text TEXT,
            hours REAL,
            date_posted TEXT,
            -- 唯一约束，用于防止重复插入
            UNIQUE(app_id, username, date_posted, review_text) 
        )
    """
    )
    conn.commit()
    return conn


def exists_in_db(app_id, conn):
    """check if app_id exists in db"""
    cursor = conn.cursor()
    # check if app_id exists in db
    cursor.execute(
        "SELECT 1 FROM steam_reviews WHERE app_id = ? LIMIT 1", (str(app_id),)
    )
    return cursor.fetchone() is not None


def insert_review(app_id, review_data, conn):
    """insert review into db"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR IGNORE INTO steam_reviews 
            (app_id, username, recommendation, review_text, hours, date_posted) 
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                str(app_id),
                review_data["username"],
                review_data["recommendation"],
                review_data["text"],
                float(review_data["hours"]),
                review_data["date_posted"],
            ),
        )
    except sqlite3.Error as e:
        print(f"insert error for {app_id}: {e}")


def save_to_db(app_id, reviews_list, conn):
    """save reviews_list into db"""
    if not reviews_list:
        return 0

    for review in reviews_list:
        insert_review(app_id, review, conn)

    conn.commit()  # commit transaction
    return len(reviews_list)


# -----------------------------
# crawl and parse html
# -----------------------------


def parse_html_reviews(html_content):
    """parse html content, extract reviews, using 'review_box' as selector"""
    soup = BeautifulSoup(html_content, "html.parser")
    reviews = []
    review_blocks = soup.find_all("div", class_="review_box")

    for block in review_blocks:
        review_data = {}
        # extract username
        user_name_div = block.find("div", class_="persona_name")
        review_data["username"] = user_name_div.text.strip() if user_name_div else "N/A"
        # extract recommendation
        recommendation_div = block.find("div", class_="title")
        review_data["recommendation"] = (
            recommendation_div.text.strip() if recommendation_div else "N/A"
        )
        # extract review text
        review_text_div = block.find("div", class_="content")
        review_text = review_text_div.text.strip() if review_text_div else "N/A"
        # clean text
        review_data["text"] = (
            review_text.split("Read More")[0].strip()
            if "Read More" in review_text
            else review_text
        )

        # extract playtime
        playtime_div = block.find("div", class_="hours")
        if playtime_div:
            # clean text, extract number part
            hours_text = (
                playtime_div.text.strip().replace(" hrs on record", "").replace(",", "")
            )
            try:
                review_data["hours"] = float(hours_text)
            except ValueError:
                review_data["hours"] = 0.0
        else:
            review_data["hours"] = 0.0

        # extract date posted
        date_posted_div = block.find("div", class_="postedDate")
        if date_posted_div:
            full_date_text = date_posted_div.text.strip()
            date_part = full_date_text.replace("Posted:", "").strip()
            review_data["date_posted"] = date_part.split("Direct from Steam")[0].strip()
        else:
            review_data["date_posted"] = "N/A"

        reviews.append(review_data)

    return reviews


def fetch_steam_page(app_id):
    """send request, fetch and parse first page of reviews. return parsed list or None/[]."""
    url = BASE_URL.format(app_id)
    params = {
        "use_review_quality": "1",
        "cursor": initial_cursor,
        "day_range": "30",
        "filter": "summary",
        "language": LANGUAGE,
        "review_type": "all",
        "filter_offtopic_activity": "1",
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()  # check for 4xx/5xx errors
        data = response.json()

        if data.get("success") != 1:
            print(f"API return error: success={data.get('success')}。")
            return None

        html_content = data.get("html")
        if not html_content:
            return []  # no review content

        return parse_html_reviews(html_content)

    except requests.exceptions.Timeout:
        print("request timeout.")
        return None
    except requests.RequestException as e:
        print(f"request exception: {e}")
        return None
    except json.JSONDecodeError:
        print("API return content is not valid JSON.")
        return None


# -----------------------------
#   main function
# -----------------------------
def main_scraper():
    print("starting Steam review crawler...")

    # 1. init db connection
    conn = setup_database()

    # 2. read app id list
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"error: cannot find {INPUT_CSV_PATH}")
        conn.close()
        return

    try:
        # read all app ids
        app_ids_all = pd.read_csv(INPUT_CSV_PATH)["app_id"].astype(str).tolist()
    except KeyError:
        print(f"error: cannot find 'app_id' column in {INPUT_CSV_PATH}")
        conn.close()
        return

    # 3. slice app id list
    app_ids = app_ids_all[start_row:end_row]
    total_ids_all = len(app_ids_all)

    print(f"successfully loaded {total_ids_all} app ids.")
    print(f"this run slice: row {start_row} to {end_row-1} (total {len(app_ids)} ids)")
    print("-" * 60)

    failed_ids = []

    # 4. main loop: traverse app ids
    # cnt 从 start_row + 1
    for cnt, app_id in enumerate(app_ids, start=start_row + 1):

        # 5. check if exists in db
        if exists_in_db(app_id, conn):
            print(f"[{cnt}/{total_ids_all}]: App ID {app_id} has been processed, skip.")
            continue

        print(f"[{cnt}/{total_ids_all}]: processing App ID = {app_id}")

        # 6. fetch reviews data
        reviews_data = fetch_steam_page(app_id)

        # 7. handle results
        if reviews_data is None:
            failed_ids.append(app_id)
            print(f"failed: cannot fetch data for App ID {app_id}.")
        elif reviews_data == []:
            print(f"✅ App ID {app_id} has no reviews for this filter.")
        else:
            # 8. save data to db
            count = save_to_db(app_id, reviews_data, conn)
            print(f"✅ successfully saved {count} reviews for App ID {app_id}.")

        # 9. random delay between requests
        delay = random.uniform(0.5, 0.8)
        time.sleep(delay)

    print("-" * 60)

    # 10. save error app_id
    if failed_ids:
        # 10. save error app_id to csv
        failed_df = pd.DataFrame({"app_id": failed_ids})
        if os.path.exists("failed_appids.csv"):
            failed_df.to_csv("failed_appids.csv", mode="a", header=False, index=False)
        else:
            failed_df.to_csv("failed_appids.csv", index=False)

        print(
            f"successfully saved {len(failed_ids)} failed app ids to failed_appids.csv"
        )
    else:
        print("all app ids processed successfully or already exists in db.")

    # 11. close db connection
    conn.close()
    print("database connection closed. program ends.")


if __name__ == "__main__":
    main_scraper()
