#!/usr/bin/env python3
"""Sync articles from alteredcraft.com RSS feed.

Fetches the RSS feed and downloads any articles not already saved locally.

Usage:
    uv run sync_articles.py                  # Sync weekly_review and deep_dive only
    uv run sync_articles.py --all            # Sync all articles (no filter)
"""

import argparse
import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import frontmatter
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# =============================================================================
# CONFIGURATION
# =============================================================================

FEED_URL = "https://alteredcraft.com/feed"
ARTICLES_DIR = Path("./articles")

# Article types to sync by default (use --all to override)
# see 'Custom tags' https://alteredcraft.com/publish/settings
SYNC_TYPES = {"weekly_review", "deep_dive"}


def detect_article_type(title: str) -> str:
    """Detect article type from title."""
    title_lower = title.lower()
    if title_lower.startswith("weekly review") or title_lower.startswith("weekly ai review"):
        return "weekly_review"
    # Known deep dive patterns
    deep_dive_keywords = [
        "a second look",
        "antigravity ide",
        "testing mozilla",
        "countering pr",
        "developers, ai changed",
        "claude skills",
        "mapping ai agent",
        "engineering ai generated",
        "context trap",
        "memory illusion",
        "ai agent can't remember",
        "building for the agentic era",
        "when ai writes code",
    ]
    for keyword in deep_dive_keywords:
        if keyword in title_lower:
            return "deep_dive"
    return "other"


def get_existing_urls() -> set[str]:
    """Get all URLs from existing articles."""
    urls = set()
    for f in ARTICLES_DIR.glob("*.md"):
        try:
            post = frontmatter.load(f)
            url = post.get("url", "")
            if url:
                urls.add(url)
        except Exception:
            pass
    return urls


def fetch_article_content(url: str) -> str | None:
    """Fetch and convert article content to markdown."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"  ERROR fetching: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Find content element
    content_elem = None
    for selector in ["div.body", "div.post-content", "article"]:
        tag, _, class_name = selector.partition(".")
        if class_name:
            content_elem = soup.find(tag, class_=class_name)
        else:
            content_elem = soup.find(tag)
        if content_elem:
            break

    if not content_elem:
        return None

    # Clean up
    for unwanted in content_elem.find_all(["script", "style", "button"]):
        unwanted.decompose()

    # Convert to markdown
    content = md(str(content_elem), heading_style="ATX")
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def generate_filename(title: str, publish_date: datetime | None) -> str:
    """Generate filename from title and date."""
    if publish_date:
        date_prefix = publish_date.strftime("%Y-%m-%d")
    else:
        date_prefix = "undated"

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    return f"{date_prefix}_{slug}.md"


def extract_slug(url: str) -> str:
    """Extract slug from article URL."""
    if "/p/" in url:
        return url.split("/p/")[-1].rstrip("/")
    return ""


def save_article(title: str, url: str, publish_date: datetime | None, content: str, article_type: str) -> str:
    """Save article with frontmatter. Returns filename."""
    filename = generate_filename(title, publish_date)
    filepath = ARTICLES_DIR / filename

    date_str = publish_date.strftime("%B %d, %Y") if publish_date else "Unknown"
    date_prefix = publish_date.strftime("%Y-%m-%d") if publish_date else "undated"
    slug = extract_slug(url)
    article_id = f"{date_prefix}_{slug}" if slug else ""
    title_yaml = f'"{title}"' if ":" in title else title

    article = f"""---
author: Sam Keen
id: {article_id}
publish_date: {date_str}
title: {title_yaml}
type: {article_type}
url: {url}
---

# {title}

{content}
"""
    filepath.write_text(article)
    return filename


def sync(sync_all: bool = False):
    """Sync articles from RSS feed.

    Args:
        sync_all: If True, sync all articles. If False, only sync DEFAULT_TYPES.
    """
    ARTICLES_DIR.mkdir(exist_ok=True)

    print(f"Fetching RSS feed: {FEED_URL}")
    feed = feedparser.parse(FEED_URL)
    print(f"Found {len(feed.entries)} entries in feed")

    existing_urls = get_existing_urls()
    print(f"Found {len(existing_urls)} existing articles")

    # Show filter info
    if sync_all:
        print("Filter: --all (syncing all types)")
    else:
        print(f"Filter: {', '.join(sorted(SYNC_TYPES))}")
    print()

    new_count = 0
    skip_count = 0
    filtered_count = 0

    for entry in feed.entries:
        url = entry.link
        title = entry.title
        article_type = detect_article_type(title)

        if url in existing_urls:
            print(f"SKIP (exists): [{article_type}] {title[:40]}...")
            skip_count += 1
            continue

        if not sync_all and article_type not in SYNC_TYPES:
            print(f"🚫 SKIP (filter): [{article_type}] {title[:40]}...")
            filtered_count += 1
            continue

        print(f"NEW: [{article_type}] {title[:40]}...")

        # Parse date
        pub_date = None
        if hasattr(entry, "published"):
            try:
                pub_date = parsedate_to_datetime(entry.published)
            except Exception:
                pass

        # Fetch content
        content = fetch_article_content(url)
        if not content:
            content = getattr(entry, "summary", "")

        if content:
            filename = save_article(title, url, pub_date, content, article_type)
            print(f"  Saved: {filename}")
            new_count += 1
        else:
            print("  ERROR: No content found")

        time.sleep(0.5)

    print(f"\nDone! New: {new_count}, Skipped: {skip_count}, Filtered: {filtered_count}")


def main():
    parser = argparse.ArgumentParser(description="Sync articles from alteredcraft.com RSS feed")
    parser.add_argument("--all", action="store_true", help="Sync all articles (default: weekly_review and deep_dive only)")

    args = parser.parse_args()
    sync(sync_all=args.all)


if __name__ == "__main__":
    main()
