"""将 techrxiv-paper.md 转换为学术论文风格 PDF"""
import markdown
import subprocess
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
md_path = os.path.join(_root, "project-docs/research/marketing/techrxiv-paper.md")
html_path = os.path.join(_dir, "paper.html")
pdf_path = os.path.join(_root, "project-docs/research/marketing/ssrn-paper.pdf")

# 读取 markdown
with open(md_path, "r", encoding="utf-8") as f:
    md_text = f.read()

# 转 HTML
md_html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

# 学术论文 CSS
css = """
@page {
    size: A4;
    margin: 2.5cm 2cm;
}
body {
    font-family: "Palatino", "Palatino Linotype", "Charter", Georgia, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #111;
    max-width: 100%;
    margin: 0;
    padding: 0;
}
h1 {
    font-size: 18pt;
    text-align: center;
    margin-bottom: 0.3em;
    line-height: 1.3;
}
/* 作者信息居中 */
h1 + p, h1 + p + p, h1 + p + p + p {
    text-align: center;
}
h2 {
    font-size: 13pt;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    border-bottom: none;
}
h3 {
    font-size: 11pt;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
}
p {
    margin: 0.5em 0;
    text-align: justify;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 1.5em 0;
}
table {
    border-collapse: collapse;
    margin: 1em auto;
    font-size: 10pt;
    width: auto;
}
th, td {
    border: 1px solid #999;
    padding: 4px 10px;
    text-align: left;
}
th {
    background: #f0f0f0;
    font-weight: bold;
}
ul, ol {
    margin: 0.5em 0;
    padding-left: 2em;
}
li {
    margin: 0.2em 0;
}
code {
    font-family: "Courier New", monospace;
    font-size: 10pt;
    background: #f5f5f5;
    padding: 1px 3px;
    border-radius: 2px;
}
em {
    font-style: italic;
}
strong {
    font-weight: bold;
}
a {
    color: #1a0dab;
    text-decoration: none;
}
/* 隐藏第一个 hr（标题下方的分隔线） */
hr:first-of-type {
    display: none;
}
"""

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>robotmem: Persistent Episodic Memory for Robot Reinforcement Learning</title>
<style>{css}</style>
</head>
<body>
{md_html}
</body>
</html>"""

# 写 HTML
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_doc)
print(f"✅ HTML: {html_path}")

# Chrome headless 打印 PDF（无页眉页脚）
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cmd = [
    chrome,
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    f"--print-to-pdf={pdf_path}",
    "--no-pdf-header-footer",
    html_path,
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
if result.returncode == 0:
    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"✅ PDF: {pdf_path} ({size_kb:.0f} KB)")
else:
    print(f"❌ Chrome 错误: {result.stderr}")
