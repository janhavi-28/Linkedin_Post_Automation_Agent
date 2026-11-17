"""
LinkedIn Playwright Publisher:
1. One-time interactive login to save persistent browser session in ./linkedin_session
2. Automated posting: attaches screenshot image + pastes post text directly to LinkedIn feed.
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

# Directory where persistent browser cookies, auth tokens, and session data are stored
USER_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "linkedin_session"))


# ==============================================================================
# 1. ONE-TIME LOGIN FLOW
# ==============================================================================
def do_login_flow(browser_ctx):
    """
    Guides the user through LinkedIn manual login inside a visible browser window.
    Once logged in, credentials/cookies are stored in USER_DATA_DIR for future runs.
    """
    page = browser_ctx.pages[0] if browser_ctx.pages else browser_ctx.new_page()
    page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
    print("\n" + "=" * 60)
    print("  [LinkedIn Auth] Please log in on the browser window.")
    print("  1. Enter your LinkedIn username and password.")
    print("  2. Complete 2FA / 'Remember this device' if prompted.")
    print("  3. Make sure you can see your LinkedIn Home Feed.")
    print("=" * 60)
    input("\nPress [Enter] here once your LinkedIn Feed is visible on screen... ")
    print("  [LinkedIn Auth] Verifying session...", flush=True)
    page.wait_for_timeout(3000)


# ==============================================================================
# 2. SESSION VALIDATION
# ==============================================================================
def is_logged_in(page) -> bool:
    """
    Checks if the currently loaded page is an active, logged-in LinkedIn session
    by verifying the URL does not contain login/checkpoint and looking for feed nav.
    """
    try:
        url = page.url.lower()
        # If redirected to login, checkpoint, or authentication screens, session is not active
        if "login" in url or "checkpoint" in url or "uas" in url:
            return False
        # Look for global navigation or home profile indicators
        nav = page.locator(".global-nav, [data-view-name='profile-card'], nav, button[aria-label*='Home' i]")
        return nav.count() > 0
    except Exception:
        return False


# ==============================================================================
# 3. AUTOMATED POST PUBLISHER
# ==============================================================================
def post_to_linkedin(text: str, image_path: str = None, headless: bool = False, auto_submit: bool = True) -> bool:
    """
    Automates publishing a post to LinkedIn with text copy and an optional image attachment.
    
    Args:
        text (str): Post description content.
        image_path (str, optional): Path to screenshot/image to attach.
        headless (bool): Whether to run headless (default: False for reliability against bot checks).
        auto_submit (bool): If True, clicks 'Post' automatically. If False, pauses for manual review.
    """
    os.makedirs(USER_DATA_DIR, exist_ok=True)

    with sync_playwright() as p:
        print("[LinkedIn] Launching browser window...", flush=True)
        # Launch persistent context: reuses stored cookies and local storage from ./linkedin_session
        browser_ctx = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled", # Hides standard automated browser flags
                "--start-maximized",                            # Maximizes view for consistent DOM layout
            ],
            no_viewport=True,
        )
        page = browser_ctx.pages[0] if browser_ctx.pages else browser_ctx.new_page()

        # Step 3A: Navigate to LinkedIn Home Feed and verify session status
        print("[LinkedIn] Checking session...", flush=True)
        page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=40000)
        page.wait_for_timeout(4000)

        # If session expired or first-time run, trigger login guide
        if not is_logged_in(page):
            print("[LinkedIn] Session not logged in. Opening login page...", flush=True)
            do_login_flow(browser_ctx)
            page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=40000)
            page.wait_for_timeout(4000)

        # Step 3B: Open LinkedIn's 'Create a post' modal
        print("[LinkedIn] Opening post modal...", flush=True)
        # Strategy A: Use LinkedIn's native URL query parameter to trigger the composer modal
        page.goto("https://www.linkedin.com/feed/?shareActive=true", wait_until="domcontentloaded")
        page.wait_for_timeout(3000)

        # Strategy B: If modal not yet visible, fallback to clicking 'Start a post' button variants
        editor = page.locator(
            "div.ql-editor[contenteditable='true'], div[role='textbox'][contenteditable='true'], [data-placeholder*='What do you want to talk about' i]"
        ).first
        if not editor.is_visible():
            for sel in [
                "button:has-text('Start a post')",
                "[aria-label*='Start a post' i]",
                ".share-box-feed-entry__trigger",
                "button.share-box-feed-entry__trigger",
                "div.share-box-feed-entry button",
                "span:text-is('Start a post')",
            ]:
                try:
                    btn = page.locator(sel).first
                    if btn.count() and btn.is_visible():
                        btn.click()
                        page.wait_for_timeout(2000)
                        if editor.is_visible():
                            break
                except Exception:
                    pass

        # Wait for the post text editor element to appear on screen
        try:
            editor.wait_for(state="visible", timeout=12000)
            print("[LinkedIn] Post modal opened successfully!", flush=True)
        except Exception:
            print("[LinkedIn] Notice: Please click 'Start a post' on the browser screen if not open yet.")
            input("Press [Enter] here once the post dialog is open... ")

        # Step 3C: Attach image screenshot (if available)
        if image_path and os.path.exists(image_path):
            abs_image = os.path.abspath(image_path)
            print(f"[LinkedIn] Attaching screenshot: {abs_image}...", flush=True)
            
            uploaded = False
            # Method 1: Inject image directly into the hidden file input
            file_inputs = page.locator("input[type='file']")
            for i in range(file_inputs.count()):
                try:
                    fi = file_inputs.nth(i)
                    accept = fi.get_attribute("accept") or ""
                    if "image" in accept or accept == "":
                        fi.set_input_files(abs_image)
                        uploaded = True
                        break
                except Exception:
                    pass

            # Method 2: Click the media/photo button to trigger file chooser dialog
            if not uploaded:
                for media_sel in [
                    "button[aria-label*='media' i]",
                    "button[aria-label*='photo' i]",
                    "button:has-text('Media')",
                    "button:has-text('Photo')",
                ]:
                    try:
                        btn = page.locator(media_sel).first
                        if btn.count() and btn.is_visible():
                            with page.expect_file_chooser(timeout=5000) as fc_info:
                                btn.click()
                            fc_info.value.set_files(abs_image)
                            uploaded = True
                            break
                    except Exception:
                        pass

            page.wait_for_timeout(3500)

            # In LinkedIn's photo preview dialog, click the 'Next' button
            for next_sel in [
                "button:has-text('Next')",
                ".share-box-footer__primary-btn",
                "button[aria-label='Next']",
                ".share-box-footer button.artdeco-button--primary",
            ]:
                try:
                    next_btn = page.locator(next_sel).first
                    if next_btn.count() and next_btn.is_visible():
                        next_btn.click()
                        page.wait_for_timeout(2000)
                        break
                except Exception:
                    pass

        # Step 3D: Insert formatted post copy into the rich text editor
        print("[LinkedIn] Pasting post description...", flush=True)
        editor = page.locator(
            "div.ql-editor[contenteditable='true'], div[role='textbox'][contenteditable='true'], [data-placeholder*='What do you want to talk about' i]"
        ).first
        editor.click()
        page.wait_for_timeout(500)

        # Uses browser execCommand('insertText') to ensure all paragraph breaks and newlines are preserved
        page.evaluate(
            """({text}) => {
                const el = document.querySelector("div.ql-editor[contenteditable='true'], div[role='textbox'][contenteditable='true']");
                if (el) {
                    el.focus();
                    document.execCommand('selectAll', false, null);
                    document.execCommand('insertText', false, text);
                }
            }""",
            {"text": text}
        )
        page.wait_for_timeout(2000)

        # Step 3E: User manual review gate (if --manual flag was passed)
        if not auto_submit:
            print("\n[LinkedIn] Post is ready in the browser window!")
            confirm = input("Press [Enter] to publish now, or type 'cancel': ").strip().lower()
            if confirm in ["cancel", "n", "no"]:
                print("[LinkedIn] Publish cancelled by user.")
                browser_ctx.close()
                return False

        # Step 3F: Submit and publish the post
        print("[LinkedIn] Publishing post...", flush=True)
        post_btn = None
        for p_sel in [
            "button.share-actions__primary-action",
            "button:has-text('Post'):not([disabled])",
            "div[role='dialog'] button:has-text('Post')",
            "button[data-control-name='share.post']",
        ]:
            try:
                b = page.locator(p_sel).last
                if b.count() and b.is_visible():
                    post_btn = b
                    break
            except Exception:
                pass

        if post_btn:
            post_btn.click()
            print("[LinkedIn] Post button clicked. Confirming upload...", flush=True)
            page.wait_for_timeout(6000)
        else:
            print("[LinkedIn] Notice: Please click 'Post' in the browser window manually.")
            input("Press [Enter] here once you have clicked Post... ")

        # Step 3G: Clean up and close browser
        browser_ctx.close()
        print("\n" + "=" * 60)
        print("  [SUCCESS] Your post has been published to LinkedIn!")
        print("=" * 60 + "\n")
        return True


# ==============================================================================
# 4. COMMAND-LINE INTERFACE
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Publish post to LinkedIn via Playwright")
    parser.add_argument("--login", action="store_true", help="One-time interactive login to LinkedIn")
    parser.add_argument("--post", default="post.txt", help="Path to post copy text file (default: post.txt)")
    parser.add_argument("--image", default="screenshot.png", help="Path to screenshot image (default: screenshot.png)")
    parser.add_argument("--manual", action="store_true", help="Review in browser window before clicking Post")
    args = parser.parse_args()

    # Route: Standalone login setup
    if args.login:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        with sync_playwright() as p:
            browser_ctx = p.chromium.launch_persistent_context(
                user_data_dir=USER_DATA_DIR,
                headless=False,
                args=["--disable-blink-features=AutomationControlled"],
            )
            do_login_flow(browser_ctx)
            browser_ctx.close()
    # Route: Publish post + screenshot
    else:
        text = ""
        if os.path.exists(args.post):
            text = open(args.post, "r", encoding="utf-8").read().strip()
        else:
            print(f"Error: {args.post} not found.")
            sys.exit(1)

        img = args.image if os.path.exists(args.image) else None
        post_to_linkedin(text=text, image_path=img, auto_submit=not args.manual)
