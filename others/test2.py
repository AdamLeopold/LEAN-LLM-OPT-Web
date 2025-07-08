from flask import Flask, render_template_string

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Optimization Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" />
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ],
                macros: {
                    '\\text': '\\text',
                    '\\frac': '\\frac',
                    '--': '\\cdot'
                }
            });">
    </script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/languages/python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js"></script>
    <script>MathJax = {tex: {inlineMath: [['$', '$'],['$$', '$$'], ['\\(', '\\)']]}}</script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    <style>
        /* Adjusting fonts and colors */
        body {
            font-family: "Times New Roman", Times, serif;
            background-color: #fffefe;
        }
        /* Additional styles */
    </style>
</head>
<body>
    <div>{{ content|safe }}</div>
</body>
</html>
"""

@app.route("/")
def index():
    md_text = r"""
    $$\begin{aligned}
    \text{Maximize} \quad & \sum_{i} A_i x_i \\
    \text{Subject To} \quad & \\
    & \text{inventory_constraint: } x_i \leq I_i, \quad \forall i \\
    & \text{demand_constraint: } x_i \leq d_i, \quad \forall i \\
    \text{where} \quad & I = [70, 30, 80, 500, 990, 250, 900, 690, 300, 40, 340, 190, 240, 440, 760, 270, 680, 440, 800, 290, 480, 580, 240, 890, 50, 670, 880, 510, 700, 590, 600, 850, 260, 760, 670, 850, 350, 170, 210, 790, 200, 830, 40, 310, 920, 820, 470, 150, 950, 170, 170, 790, 570, 680, 560, 360, 890, 600, 960, 290, 160, 620, 580, 880, 600, 470, 610, 760, 350, 830] \\
    & d = [9, 4, 10, 65, 125, 31, 116, 98, 40, 5, 48, 28, 32, 55, 102, 40, 101, 66, 102, 43, 60, 81, 29, 121, 7, 97, 131, 64, 100, 76, 78, 112, 33, 102, 96, 112, 52, 26, 30, 109, 25, 125, 6, 43, 138, 121, 59, 21, 125, 25, 24, 109, 80, 90, 75, 45, 132, 87, 130, 42, 20, 90, 75, 110, 84, 62, 79, 95, 48, 122] \\
    & A = [1283.7, 371.56, 1003.56, 321.11, 956.78, 641.22, 894.34, 1316.82, 951.85, 875.91, 1088.57, 467.93, 1211.35, 348.43, 1409.21, 1237.37, 867.3, 178.22, 518.67, 701.97, 1068.1, 178.52, 377.8, 287.36, 1118.63, 728.62, 992.14, 513.64, 153.61, 1175.92, 479.44, 1196.95, 167.99, 478.36, 256.42, 805.82, 1279.24, 498.77, 1183.94, 338.6, 872.16, 753.55, 549.92, 1128.07, 973.62, 1285.81, 156.02, 1064.72, 1134.19, 1030.8, 1136.87, 965.34, 507.13, 462.26, 169.7, 245.81, 1184.54, 887.06, 1269.71, 483.26, 448.61, 971.93, 139.14, 1042.66, 678.72, 972.1, 1318.33, 367.91, 869.95, 1385.88]
    \end{aligned}$$
    """
    return render_template_string(TEMPLATE, content=md_text)

if __name__ == "__main__":
    app.run(debug=True)
