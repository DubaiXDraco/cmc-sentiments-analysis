#!/usr/bin/env python3
"""
scripts/03c_crawler_playwright.py
Tuần 1/3: Crawler trang web công khai bằng Playwright có giới hạn tốc độ (rate-limiting)
và ẩn danh hóa PII ngay lập tức qua `src/anonymize.py` trước khi ghi ra đĩa.
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anonymize import anonymize_id, anonymize_text
from src.preprocess import clean_text
from src.topics import classify_topic


async def crawl_public_page(
    url: str,
    selector: str = "article, .post-content, p",
    max_scrolls: int = 3,
    delay_sec: float = 2.0,
) -> List[Dict[str, str]]:
    from playwright.async_api import async_playwright

    rows: List[Dict[str, str]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        for _ in range(max_scrolls):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(int(delay_sec * 1000))

        elements = await page.query_selector_all(selector)
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        now_dt = datetime.now().strftime("%Y-%m-%d")

        for idx, el in enumerate(elements, start=1):
            raw_text = " ".join((await el.inner_text()).split())
            if len(raw_text) < 15:
                continue
            anon_text = anonymize_text(raw_text)
            if len(clean_text(anon_text)) < 5:
                continue
            rows.append(
                {
                    "id": anonymize_id(f"{url}_{idx}_{now_dt}"),
                    "type": "post",
                    "source": "public_web_crawler",
                    "timestamp": now_ts,
                    "date": now_dt,
                    "likes": 0,
                    "topic": classify_topic(anon_text),
                    "text": anon_text,
                }
            )
            time.sleep(0.02)

        await browser.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Playwright crawler công khai có ẩn danh hóa PII.")
    parser.add_argument("--url", type=str, required=True, help="URL trang công khai cần thu thập.")
    parser.add_argument("--selector", type=str, default="article, .post-content, p")
    parser.add_argument("--scrolls", type=int, default=3)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--out", type=str, default=str(ROOT / "data" / "crawled_public.csv"))
    args = parser.parse_args()

    records = asyncio.run(
        crawl_public_page(
            url=args.url,
            selector=args.selector,
            max_scrolls=args.scrolls,
            delay_sec=args.delay,
        )
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Đã thu thập và ẩn danh hóa {len(records)} bản ghi -> {out_path}")


if __name__ == "__main__":
    main()
