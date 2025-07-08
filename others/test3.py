from flask import Flask, render_template_string
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension

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
                    {left: '$', right: '$', display: false},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false}
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

        .sidebar {
            background-color: #e3f6f5;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            width: 400px;
            color: #000000;
            border-radius: 1rem;
            margin-right: 2rem;
        }

        .slide-up {
            animation: slideUp 0.5s ease-out;
        }
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .preview-table {
            max-height: 400px;
            overflow-y: auto;
        }
        .preview-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.7rem;
            color: #272343;
        }
        .preview-table th {
            position: sticky;
            top: 0;
            background: #fffefe;
            z-index: 10;
            padding: 0.25rem 0.5rem;
            font-weight: 500;
            border-radius: 0.5rem;
        }
        .preview-table td {
            padding: 0.25rem 0.5rem;
            border-top: 1px solid #e5e7eb;
            border-radius: 0.5rem;
        }
        .progress-stepper {
            position: fixed;
            left: 1rem;
            bottom: 1rem;
            z-index: 50;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            pointer-events: none;
        }
        .progress-stepper-inner {
            background: rgba(255,255,255,0.95);
            border-radius: 0.5rem;
            box-shadow: 0 2px 8px 0 rgba(0,0,0,0.08);
            padding: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            pointer-events: auto;
        }
        .step-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
            color: #6b7280;
            transition: all 0.5s ease;
        }
        .step-item.completed {
            opacity: 0;
            transform: translateY(-10px);
        }
        .step-item.completed-keep {
            opacity: 1;
            transform: none;
        }
        .step-item.current {
            color: #ffd803;
        }
        .step-item.future {
            color: #d1d5db;
        }
        .circle-spin {
            transform-origin: 50% 50%;
            animation: circleSpin 1s linear infinite;
        }
        @keyframes circleSpin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner-circle {
            stroke-dasharray: 283;
            stroke-dashoffset: 75;
            stroke-width: 4;
            stroke-linecap: round;
            animation: spinnerDash 1.5s ease-in-out infinite;
        }
        @keyframes spinnerDash {
            0% {
                stroke-dashoffset: 283;
            }
            50% {
                stroke-dashoffset: 75;
            }
            100% {
                stroke-dashoffset: 283;
            }
        }
        .chat-container {
            max-height: calc(100vh - 400px);
            overflow-y: auto;
        }
        .chat-message {
            max-width: 80%;
            margin-bottom: 1rem;
            border-radius: 1rem;
        }
        .chat-message.user {
            margin-left: auto;
            background-color: #bae8e8;
            color: #272343;
        }
        .chat-message.assistant {
            margin-right: auto;
            background-color: #fffefe;
        }
        .chat-message-content {
            padding: 1rem;
            border-radius: 1rem;
        }
        .problem-description-container {
            position: fixed;
            bottom: 0;
            left: 420px;
            background: #fffefe;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
            padding: 1.5rem;
            z-index: 40;
            width: calc(100% - 420px);
            border-radius: 1rem;
        }
        .problem-description-inner {
            margin: 0 auto;
        }
        .main-content {
            padding-bottom: 180px;
            padding-left: 420px;
        }
        .resize-handle {
            position: absolute;
            right: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: #e3f6f5;
            cursor: col-resize;
            transition: background 0.2s;
            border-radius: 0 1rem 1rem 0;
        }
        .resize-handle:hover {
            background: #ffd803;
        }
        .auto-resize-textarea {
            min-height: 60px;
            max-height: 200px;
            resize: none;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: #cbd5e1 #f1f5f9;
            border-radius: 1rem;
        }
        .auto-resize-textarea::-webkit-scrollbar {
            width: 6px;
        }
        .auto-resize-textarea::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 3px;
        }
        .auto-resize-textarea::-webkit-scrollbar-thumb {
            background-color: #cbd5e1;
            border-radius: 3px;
        }
        .welcome-message {
            text-align: center;
            padding: 2rem;
            color: #272343;
            border-radius: 1rem;
        }
        .welcome-message h2 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .welcome-message p {
            font-size: 1.2rem;
            line-height: 1.6;
            color: #2d334a;
        }
        .button {
            background-color: #ffd803;
            color: #272343;
            padding: 12px 24px;
            border-radius: 1rem;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        .button:hover {
            background-color: #bae8e8;
        }
        .chat-message-content h1, 
        .chat-message-content h2, 
        .chat-message-content h3, 
        .chat-message-content h4, 
        .chat-message-content h5, 
        .chat-message-content h6 {
            margin-top: 1em;
            margin-bottom: 0.5em;
            font-weight: bold;
            color: #272343;
        }
        
        .chat-message-content h1 { font-size: 1.5em; }
        .chat-message-content h2 { font-size: 1.3em; }
        .chat-message-content h3 { font-size: 1.1em; }
        
        .chat-message-content p {
            margin-bottom: 1em;
            line-height: 1.6;
        }
        
        .chat-message-content ul, 
        .chat-message-content ol {
            margin-bottom: 1em;
            padding-left: 2em;
        }
        
        .chat-message-content li {
            margin-bottom: 0.5em;
        }
        
        .chat-message-content code:not(.hljs) {
            background-color: #f3f4f6;
            padding: 0.2em 0.4em;
            border-radius: 0.3em;
            font-family: monospace;
            color: #d63384;
        }
        
        .chat-message-content pre {
            background-color: #f8f8f8;
            padding: 1em;
            border-radius: 0.5em;
            overflow-x: auto;
            margin-bottom: 1em;
        }
        
        .chat-message-content pre code.hljs {
            padding: 1em;
            border-radius: 0.5em;
            font-size: 0.9em;
            line-height: 1.5;
            background-color: #f8f8f8;
        }
        
        .chat-message-content blockquote {
            border-left: 4px solid #bae8e8;
            padding-left: 1em;
            margin-left: 0;
            color: #4a5568;
            margin-bottom: 1em;
        }
        
        .chat-message-content table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 1em;
        }
        
        .chat-message-content th, 
        .chat-message-content td {
            border: 1px solid #e2e8f0;
            padding: 0.5em;
            text-align: left;
        }
        
        .chat-message-content th {
            background-color: #f7fafc;
        }
        
        .chat-message-content tr:nth-child(even) {
            background-color: #f7fafc;
        }
    
        /* Math expression styling */
        .math-display {
            overflow-x: auto;
            overflow-y: hidden;
            margin: 1em 0;
        }
        
        .math-inline {
            color: #2b6cb0;
        }
        
        /* Specific styling for input */
        input[type="password"] {
            color: #272343;
            border-radius: 1rem;
        }
    </style>
</head>
<body>
    <div>{{ content|safe }}</div>
</body>
</html>
"""

@app.route("/")
def index():
    # Prepare the LaTeX content
    # before = """\\["""
    # Define the LaTeX expression here, ensuring correct usage of '&' within align environments
    rs = '''\\[
\\begin{align*}
 \\text{Minimize} \quad & \\sum_{i=1}^{n} c_i x_i \quad \\text{(Objective function)} \\\\
& x_i \\leq I_i, \\quad \\forall i \\quad \\text{(Inventory constraint)} \\\\
& x_i \\leq D_i, \\quad \\forall i \\quad \\text{(Demand constraint)} \\\\
& x_i \in \\mathbb{Z}_{\\geq 0}, \\quad \\forall i \\quad \\text{(Non-negativity and integrality)}
\\end{align*}
\\]
    


This is a math expression:

$$
\int_0^1 x^2 \, dx = \frac{1}{3}
$$

And here's some Python code:

```python
def greet(name):
    return f"Hello, {name}"
```
    
'''
    # rs = rs.replace("\\", "\\\\")
    # end = """\\]"""
    
    # Replace single backslashes with double backslashes
    # rs = rs.replace('\\\\', '\\\\\\\\')

    # Concatenate LaTeX parts
    md_text = rs

    # Render the template with the LaTeX content
    return render_template_string(TEMPLATE, content=md_text)

if __name__ == "__main__":
    app.run(debug=True)
