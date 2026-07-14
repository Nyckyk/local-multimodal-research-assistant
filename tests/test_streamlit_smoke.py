from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest
import requests

from settings import BASE_DIR


pytestmark = pytest.mark.smoke


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(url, process, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("Streamlit exited before becoming ready")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError("Streamlit did not respond before the smoke-test timeout")


def test_streamlit_page_and_chat_smoke():
    playwright = pytest.importorskip(
        "playwright.sync_api",
        reason="optional smoke test requires Playwright; scientific regressions do not",
    )
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless=true", f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_server(url, process)
        with playwright.sync_playwright() as context:
            browser = context.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            page.get_by_text("Local Research Assistant", exact=True).wait_for(timeout=120_000)
            chat = page.locator('[data-testid="stChatInputTextArea"]')
            chat.wait_for(state="visible", timeout=30_000)
            chat.fill("What papers are available?")
            chat.press("Enter")
            page.get_by_text("What papers are available?", exact=True).wait_for(timeout=30_000)
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
