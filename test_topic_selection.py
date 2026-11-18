"""Offline selection checks; do not initialize Gemini or run discovery."""
import ast
from collections import Counter
import json
from pathlib import Path
import re
import sys
import unittest

source = Path(__file__).with_name("post_agent2.py").read_text(encoding="utf-8")
compile(source, "post_agent2.py", "exec")
tree = ast.parse(source)
namespace = {"json": json, "re": re, "sys": sys}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {"balanced_shortlist", "select_topic"}]
exec(compile(ast.Module(body=functions, type_ignores=[]), "<selection>", "exec"), namespace)
shortlist = namespace["balanced_shortlist"]


def candidate(source, number, score=0):
    return {"type": source, "name": f"{source}-{number}",
            "url": f"https://example.com/{source}/{number}", "badge": "test",
            "desc": "A useful developer tool. " * 15, "source_score": score}


class SelectionTests(unittest.TestCase):
    def test_every_source_has_three_slots_in_both_mode(self):
        sources = ["GitHub", "Hacker News", "Hugging Face Paper", "Hugging Face Model", "Google News Tech", "Community / Blog"]
        pool = [candidate(s, i) for s in sources for i in range(20)]
        result = shortlist(pool, limit=18)
        self.assertEqual(Counter(x["type"] for x in result), dict.fromkeys(sources, 3))
        reverse_sources = [candidate(s, i) for s in reversed(sources) for i in range(20)]
        self.assertEqual(result, shortlist(reverse_sources, limit=18))

    def test_redistributes_and_supports_github_only(self):
        pool = [candidate("A", 0)] + [candidate("B", i) for i in range(20)]
        result = shortlist(pool)
        self.assertEqual(len(result), 15)
        self.assertIn(pool[0], result)
        self.assertEqual(len(shortlist([candidate("GitHub", i) for i in range(20)])), 15)
        self.assertEqual(shortlist([]), [])

    def test_ranks_only_within_source(self):
        pool = [candidate("Papers", i, i) for i in range(6)]
        pool += [candidate("News", i, 10000) for i in range(6)]
        result = shortlist(pool, limit=6)
        self.assertEqual([x["name"] for x in result if x["type"] == "Papers"], ["Papers-5", "Papers-4", "Papers-3"])

    def test_deduplicates_tracking_links(self):
        first = candidate("News", 1)
        duplicate = dict(first, url=first["url"] + "/?utm_source=feed#section")
        self.assertEqual(shortlist([first, duplicate]), [first])

    def test_selection_uses_shortlist_index_and_rich_details(self):
        pool = [candidate("News", i) for i in range(3)]
        prompts = []
        def ask(prompt, **kwargs):
            prompts.append(prompt)
            return json.dumps({"index": 2, "reason": "Useful debugger."})
        namespace["ask"] = ask
        self.assertEqual(namespace["select_topic"](pool), (pool[2], "Useful debugger."))
        self.assertIn(pool[2]["desc"], prompts[0])

    def test_retries_bad_response_and_honors_rejection(self):
        responses = iter(['{"index": 30, "reason": "Invalid"}', '{"index": 0, "reason": "Useful"}'])
        namespace["ask"] = lambda *args, **kwargs: next(responses)
        pool = [candidate("News", 0)]
        self.assertEqual(namespace["select_topic"](pool)[0], pool[0])
        namespace["ask"] = lambda *args, **kwargs: '{"index": -1, "reason": "No useful candidates"}'
        with self.assertRaises(SystemExit):
            namespace["select_topic"](pool)
        namespace["ask"] = lambda *args, **kwargs: '[]'
        with self.assertRaises(SystemExit):
            namespace["select_topic"](pool)


if __name__ == "__main__":
    unittest.main()
