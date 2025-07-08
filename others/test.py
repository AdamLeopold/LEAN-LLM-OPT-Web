from flask import Flask, render_template_string
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension

app = Flask(__name__)


@app.route("/")
def index():
    # md_text = r""" $x_l = 1$ """
    md_text = r"""

    $x_l = 1$
    """

    html = markdown.markdown(md_text, extensions=["fenced_code", CodeHiliteExtension()])
    
    return render_template_string(TEMPLATE, content=html)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Markdown with KaTeX</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" />
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ]
            });">
    </script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/pygments/2.17.2/styles/default.min.css">
    <style>
        body { max-width: 800px; margin: auto; font-family: sans-serif; padding: 2em; }
        pre { background: #f5f5f5; padding: 1em; overflow-x: auto; }
    </style>
</head>
<body>
    <div>{{ content|safe }}</div>
</body>
</html>
"""
if __name__ == "__main__":
    app.run(debug=True)

#     md_text = md_text.replace(r'\[', '$$').replace(r'\]', '$$') 
#     # md_text = md_text.replace(r'$', '$').replace(r'$', '$') 
#     md_text = md_text.replace(r'\(', '$').replace(r'\)', '$')
#     html = markdown.markdown(md_text, extensions=["fenced_code", CodeHiliteExtension()])
#     return render_template_string(TEMPLATE, content=html)


# TEMPLATE = """
# <!DOCTYPE html>
# <html>
# <head>
#     <meta charset="UTF-8">
#     <title>Markdown with KaTeX</title>
#     <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" />
#     <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
#     <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
#             onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}]});">
#     </script>
#     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/pygments/2.17.2/styles/default.min.css">
#     <style>
#         body { max-width: 800px; margin: auto; font-family: sans-serif; padding: 2em; }
#         pre { background: #f5f5f5; padding: 1em; overflow-x: auto; }
#     </style>
# </head>
# <body>
#     <div>{{ content|safe }}</div>
# </body>
# </html>
# """
# if __name__ == "__main__":
#     app.run(debug=True)
