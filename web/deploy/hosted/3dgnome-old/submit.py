"""Submit the request form of the old site with its own default values and one region."""
import sys, urllib.request, urllib.parse, http.cookiejar
from html.parser import HTMLParser

base, region = sys.argv[1], sys.argv[2]
extra = dict(a.split("=", 1) for a in sys.argv[3:])


class Form(HTMLParser):
    def __init__(self):
        super().__init__(); self.f = {}; self.sel = None; self.ta = None
    def handle_starttag(self, tag, a):
        a = dict(a)
        if tag == "input" and a.get("name"):
            t = a.get("type", "text")
            if t in ("checkbox", "radio"):
                if "checked" in a: self.f[a["name"]] = a.get("value", "y")
            elif t != "file" and t != "submit":
                self.f[a["name"]] = a.get("value", "")
        elif tag == "select": self.sel = a.get("name"); self.first = None
        elif tag == "option" and self.sel:
            if self.first is None: self.first = a.get("value", ""); self.f.setdefault(self.sel, self.first)
            if "selected" in a: self.f[self.sel] = a.get("value", "")
        elif tag == "textarea": self.ta = a.get("name"); self.f[self.ta] = ""
    def handle_endtag(self, tag):
        if tag == "select": self.sel = None
        if tag == "textarea": self.ta = None
    def handle_data(self, d):
        if self.ta: self.f[self.ta] += d


op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
p = Form(); p.feed(op.open(base + "/request/").read().decode())
p.f["region"] = region; p.f.update(extra)
print({k: v for k, v in p.f.items() if k != "csrf_token"})
r = op.open(urllib.request.Request(base + "/request/", urllib.parse.urlencode(p.f).encode(), {"Referer": base + "/request/"}))
body = r.read().decode()
import re
print(r.status, re.findall(r"(Your job request[^<]*|There was a problem[^<]*|/request/[A-Za-z0-9]{32})", body)[:6])
print([m.strip() for m in re.findall(r'class="[^"]*(?:error|alert)[^"]*"[^>]*>([^<]+)', body)][:8])
