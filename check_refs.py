"""
Bibliographic audit: query Crossref for every DOI given on the command line
or found as \\doi{...} in a LaTeX file, and print authors, title, journal,
volume(issue), pages and year as registered.

Usage:  python3 check_refs.py ../01_revised_manuscript/Manuscript_CMA.tex
        python3 check_refs.py 10.1088/1361-6544/aaf513 ...
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request


def fetch(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/:()")
    req = urllib.request.Request(url, headers={"User-Agent": "check_refs/1.0 (mailto:yy.y@zig.univ.sn)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]


def describe(msg):
    authors = ", ".join(a.get("family", a.get("name", "")) for a in msg.get("author", []))
    date = (msg.get("published-print") or msg.get("published-online") or msg.get("issued"))["date-parts"][0]
    return (f"{authors} | {msg['title'][0]} | {(msg.get('container-title') or [''])[0]} "
            f"{msg.get('volume', '')}({msg.get('issue', '')}) {msg.get('page', '')} | {date[0]}")


if __name__ == "__main__":
    dois = []
    for arg in sys.argv[1:]:
        if arg.endswith(".tex"):
            dois += re.findall(r"\\doi\{([^}]*)\}", open(arg, encoding="utf8").read())
        else:
            dois.append(arg)
    bad = 0
    for d in dois:
        try:
            print(f"{d} | {describe(fetch(d))}")
        except Exception as e:                       # noqa: BLE001
            print(f"{d} | NOT FOUND ({e})"); bad += 1
        time.sleep(0.3)
    sys.exit(1 if bad else 0)
