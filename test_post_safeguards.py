"""Offline regression tests: no API calls or publishing."""
import ast
from datetime import date, timedelta
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import Mock
import requests

source = Path(__file__).with_name("post_agent2.py").read_text(encoding="utf-8")
tree = ast.parse(source)
ns = dict(re=re, json=json, os=os, date=date, requests=requests,
          GITHUB_MAX_AGE_DAYS=30, GITHUB_MIN_STARS=20, GITHUB_MAX_STARS=5000,
          POST_MAX_WORDS=120, STYLE_EXAMPLE="Short informative text.")
names = {"parse_candidate_index", "search_web_for_tool", "resolve_repo",
         "github_candidate", "filter_web_github_candidates", "review_post_claims", "write_post"}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in names], type_ignores=[]), "<checks>", "exec"), ns)


class SafeguardTests(unittest.TestCase):
    def setUp(self):
        ns["requests"] = Mock(RequestException=requests.RequestException)
        self.body = "Review documents.\n\nTool reads text.\n\nTool scores regions."
        self.evidence = "Tool reads text. Tool scores regions."
        self.approval = {"supported": True, "issues": [], "paragraphs": [
            {"index": 0, "evidence": []}, {"index": 1, "evidence": ["Tool reads text."]},
            {"index": 2, "evidence": ["Tool scores regions."]}]}

    def test_index_is_strict_and_bounded(self):
        for reply in ["-1", "99", "choose 0", "0.5", ""]:
            self.assertEqual(ns["parse_candidate_index"](reply, 3), -1)
        self.assertEqual(ns["parse_candidate_index"](" 2 ", 3), 2)

    def test_web_lookup_does_not_guess_or_accept_bad_index(self):
        engine = Mock()
        ns["DDGS"] = lambda: engine
        engine.text.return_value = []
        with self.assertRaises(ValueError):
            ns["search_web_for_tool"]("unknown tool")
        engine.text.return_value = [{"title": "Tool", "href": "https://example.com"}]
        ns["ask"] = lambda *a, **k: "100"
        with self.assertRaises(ValueError):
            ns["search_web_for_tool"]("tool")

    def test_web_github_links_use_age_check_and_cached_metadata(self):
        old = {"created_at": "2020-01-01", "pushed_at": date.today().isoformat(),
               "stargazers_count": 100, "html_url": "https://github.com/owner/tool", "full_name": "owner/tool"}
        response = Mock()
        response.json.return_value = old
        ns["requests"].get.return_value = response
        item = {"url": old["html_url"], "type": "Hacker News", "badge": "20 points", "desc": "Tool"}
        self.assertEqual(ns["filter_web_github_candidates"]([item, dict(item, url=item["url"]+"/issues/1")], []), [])
        self.assertEqual(ns["requests"].get.call_count, 1)
        response.json.return_value = dict(old, created_at=(date.today()-timedelta(days=3)).isoformat())
        result = ns["filter_web_github_candidates"]([item], [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "Hacker News")
        ns["requests"].get.side_effect = requests.RequestException("rate limit")
        self.assertEqual(ns["filter_web_github_candidates"]([item], []), [])

    def test_review_requires_real_source_evidence(self):
        ns["ask"] = lambda *a, **k: json.dumps(self.approval)
        self.assertEqual(ns["review_post_claims"](self.body, self.evidence), [])
        self.approval["paragraphs"][1]["evidence"] = ["Tool is 100x faster."]
        self.assertTrue(ns["review_post_claims"](self.body, self.evidence))
        ns["ask"] = lambda *a, **k: "invalid json"
        self.assertTrue(ns["review_post_claims"](self.body, self.evidence))
        self.assertTrue(ns["review_post_claims"]("I tested the tool.", self.evidence))

    def test_failed_review_rewrites_then_verifies(self):
        responses = iter([self.body, json.dumps({"supported": False, "issues": ["Unsupported"]}),
                          self.body, json.dumps(self.approval)])
        ns["ask"] = lambda *a, **k: next(responses)
        result = ns["write_post"]({"name": "Tool"}, "https://example.com", self.evidence)
        self.assertIn("Explore: https://example.com", result)

    def test_repeated_review_failure_never_returns_post(self):
        responses = iter([self.body, "bad review"] * 3)
        ns["ask"] = lambda *a, **k: next(responses)
        with self.assertRaises(ValueError):
            ns["write_post"]({"name": "Tool"}, "https://example.com", self.evidence)


if __name__ == "__main__":
    unittest.main()
