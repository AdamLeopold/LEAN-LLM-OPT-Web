import markdown2
# from flask import Flask, render_template, request, jsonify
import tempfile
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
import markdown,markdown2
from bleach import clean
import re
import openai
import os
import pandas as pd
import json
import webbrowser
import threading
from langchain.vectorstores import FAISS
from langchain.schema import Document
from datetime import time as time_class  
from datetime import datetime, time
from flask import Flask, render_template_string, render_template, request, jsonify
import tempfile
# Your markdown content
markdown_content = '''
### Objective Function

Let \( x_{(l,k,j)} \) be the number of tickets sold for OD pair \( l \), departure time \( k \), and ticket type \( j \) (where \( j = f \) for Eco_flex, \( j = l \) for Eco_lite). Let \( x_o[l] \) be the number of outside option passengers for OD \( l \).

The objective is to maximize total revenue:

\[
\max \quad \sum_{(l,k,j)} \text{avg\_price}[(l,k,j)] \cdot x_{(l,k,j)}
\]

where the sum is over all ticket options for the specified flights.
'''

# Convert markdown to HTML
html_content = markdown2.markdown(markdown_content, extras=["fenced-code-blocks"])

# Wrap the HTML content with MathJax script
html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                ignoreHtmlClass: 'tex2jax_ignore',
                processHtmlClass: 'tex2jax_process'
            }
        };
    </script>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Save or serve the HTML output as needed
with open("output.html", "w") as file:
    file.write(html_output)
