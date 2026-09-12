# -*- coding: utf-8 -*-
import os,re
root=r"D:\Deepseek\minicode"
skip=("MiniCode-reference",)
key=re.compile(r"sk-[A-Za-z0-9]{16,}|(api_?key|token|secret|password)\s*[:=]\s*[\"'][^\"']{8,}",re.I)
pii=re.compile(r"15537573795|frszzu|C:\\\\Users\\\\AW|D:\\\\|/Users/AW",re.I)
for dp,dn,fns in os.walk(root):
    if any(s in dp for s in skip): continue
    if "__pycache__" in dp: continue
    for fn in fns:
        ext=os.path.splitext(fn)[1].lower()
        if ext not in (".py",".md",".txt",".json",".bat",".toml",".yaml",".yml"): continue
        p=os.path.join(dp,f)
        try: t=open(p,encoding="utf-8",errors="ignore").read()
        except: continue
        for i,line in enumerate(t.splitlines(),1):
            if key.search(line) and not re.search(r"os.environ|getenv|input\(|session|your|placeholder|text_input",line,re.I):
                print(f"[KEY] {p.replace(root,'.')}:{i}: {line.strip()[:90]}")
            if pii.search(line):
                print(f"[PII] {p.replace(root,'.')}:{i}: {line.strip()[:90]}")
print("---- 外层 skills 目录结构 ----")
sk=os.path.join(root,"skills")
if os.path.isdir(sk):
    for dp,dn,fns in os.walk(sk):
        lvl=dp.replace(sk,"").count(os.sep)
        if lvl<=1:
            print(("  "*lvl)+os.path.basename(dp)+"/", f"({len(fns)} files)")
