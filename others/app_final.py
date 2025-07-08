# from flask import Flask, render_template, request, jsonify
import tempfile
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from markdown.extensions.codehilite import CodeHiliteExtension
from langchain.agents import initialize_agent, AgentType, Tool
import markdown
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
# 其他导入保持不变

app = Flask(__name__)

# Global variable to store uploaded files
uploaded_files = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global uploaded_files
    uploaded_files = []
    
    if 'files[]' not in request.files:
        return jsonify({'status': 'fail', 'error': 'No files uploaded'}), 400
        
    files = request.files.getlist('files[]')
    
    for file in files:
        if file.filename.endswith('.csv'):
            # Use temporary file to handle upload content
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            
            # Read CSV file
            df = pd.read_csv(tmp_file_path)
            uploaded_files.append((file.filename, df))
            
    return jsonify({'status': 'success', 'message': 'Files uploaded successfully', 'count': len(uploaded_files)})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    query = data.get('query')
    api_key = data.get('api_key')
    
    if not query or not api_key:
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        # Initialize first LLM for question classification
        llm1 = ChatOpenAI(
            temperature=0.0,
            model_name="gpt-4",
            openai_api_key=api_key
        )
        
        # Load reference documents
        loader = CSVLoader(file_path="RefData.csv", encoding="utf-8")
        documents = loader.load()
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_documents(documents, embeddings)
        retriever = vectors.as_retriever(search_kwargs={'k': 5})
        
        # Build retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm1,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        
        # Create QA tool
        qa_tool = Tool(
            name="FileQA",
            func=qa_chain.invoke,
            description="Use this tool to answer questions about the problem type of the text."
        )
        
        # Initialize Agent
        agent1 = initialize_agent(
            tools=[qa_tool],
            llm=llm1,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        # Get problem type
        category_result = agent1.invoke(f"What is the problem type in operation of the text? text:{query}")
        problem_type = extract_problem_type(category_result.get('output', ''))

        return jsonify({
            'problem_type': problem_type,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    query = data.get('query')
    api_key = data.get('api_key')

#     result = '''

# Maximize the total ticket sale revenue across the selected flights and ticket types:
# $$
# \max \quad \sum_{(l,k,j)} p[(l,k,j)] \cdot x[(l,k,j)]
# $$

# where:
# - $(l,k,j)$ indexes over all OD pairs $l$, departure times $k$, and ticket types $j$ (Eco_flex = f, Eco_lite = l).
# - $p[(l,k,j)]$ is the average price for ticket type $j$ on flight $(l,k)$.
# - $x[(l,k,j)]$ is the number of tickets sold for ticket type $j$ on flight $(l,k)$.

# ---

# ### Constraints

# #### 1. Capacity Constraints

# For each flight (OD pair $l$, departure time $k$):
# $$
# c_f \cdot x[(l,k,f)] + x[(l,k,l)] \leq C
# $$
# where:
# - $c_f = 1.2$ (Eco_flex ticket capacity consumption)
# - $C = 187$ (flight capacity)

# #### 2. Balance Constraints

# For each OD pair $l$:
# $$
# r_0[l] \cdot x_o[l] + \sum_{k} \sum_{j} r[(l,k,j)] \cdot x[(l,k,j)] = d[l]
# $$
# where:
# - $r_0[l]$ is the base ratio for OD $l$
# - $x_o[l]$ is the base sales for OD $l$
# - $r[(l,k,j)]$ is the ratio for ticket type $j$ on flight $(l,k)$
# - $d[l]$ is the demand for OD $l$

# #### 3. Scale Constraints

# For each ticket option \((l,k,j)\):
# $$
# \frac{x[(l,k,j)]}{v[(l,k,j)]} - \frac{x_o[l]}{v_0[l]} \leq 0
# $$
# where:
# - $v[(l,k,j)]$ is the value coefficient for ticket type $j$ on flight $(l,k)$
# - $v_0[l]$ is the base value for OD $l$

# #### 4. Big M Constraints

# For each ticket option \((l,k,j)\):
# $$
# x[(l,k,j)] \leq M \cdot y[(l,k)]
# $$
# where:
# - $y[(l,k)]$ is a binary variable indicating if flight $(l,k)$ is selected
# - $M = 10,000,000$

# #### 5. Cardinality Constraint

# Select at most $n_o = 2$ flights:
# $$
# \sum_{(l,k)} y[(l,k)] \leq 2
# $$

# #### 6. Flow Conservation Constraints

# At each airport, the number of incoming and outgoing selected flights must balance:

# - For airport A:
#   $$
#   \sum_{(l,k) \in \sigma_A^+} y[(l,k)] = \sum_{(l,k) \in \sigma_A^-} y[(l,k)]
#   $$
# - For airport B:
#   $$
#   \sum_{(l,k) \in \sigma_B^+} y[(l,k)] = \sum_{(l,k) \in \sigma_B^-} y[(l,k)]
#   $$
# - For airport C:
#   $$
#   \sum_{(l,k) \in \sigma_C^+} y[(l,k)] = \sum_{(l,k) \in \sigma_C^-} y[(l,k)]
#   $$

# #### 7. Nonnegativity Constraints

# $$
# x[(l,k,j)] \geq 0, \quad x_o[l] \geq 0
# $$

# #### 8. Binary Constraints

# $$
# y[(l,k)] \in \{0,1\}
# $$

# ---

# ### Retrieved Information

# - Prices:
#   ```
#   p = {
#     '(AB,11:20,f)': '1140.3', '(AB,11:20,l)': '429.26',
#     '(AB,06:40,f)': '1142.6', '(AB,06:40,l)': '482.92',
#     '(AB,07:55,f)': '1138.19', '(AB,07:55,l)': '441.83',
#     '(AC,09:45,f)': '1292.44', '(AC,09:45,l)': '485.92',
#     '(BA,06:25,f)': '1131.33', '(BA,06:25,l)': '424.19',
#     '(BA,09:05,f)': '1123.69', '(BA,09:05,l)': '419.78',
#     '(CA,07:40,f)': '1285.82', '(CA,07:40,l)': '558.55',
#     '(CA,08:15,f)': '1284.5', '(CA,08:15,l)': '477.98'
#   }
#   ```
# - Value coefficients:
#   ```
#   v = {
#     '(AB,11:20,f)': 3.896123449, '(AB,11:20,l)': 1,
#     '(AB,06:40,f)': 2.236888293, '(AB,06:40,l)': 0.521872683,
#     '(AB,07:55,f)': 2.236888293, '(AB,07:55,l)': 0.521872683,
#     '(AC,09:45,f)': 3.297109885, '(AC,09:45,l)': 1,
#     '(BA,06:25,f)': 1.813282947, '(BA,06:25,l)': 0.505261036,
#     '(BA,09:05,f)': 2.600413693, '(BA,09:05,l)': 1,
#     '(CA,07:40,f)': 0.865902564, '(CA,07:40,l)': 0.666898117,
#     '(CA,08:15,f)': 2.440052228, '(CA,08:15,l)': 1
#   }
#   ```
# - Ratio coefficients:
#   ```
#   r = {
#     '(AB,11:20,f)': 2.48264215, '(AB,11:20,l)': 32.96333763,
#     '(AB,06:40,f)': 1.91965885, '(AB,06:40,l)': -22.81661357,
#     '(AB,07:55,f)': 1.91965885, '(AB,07:55,l)': -22.81661357,
#     '(AC,09:45,f)': 1.0, '(AC,09:45,l)': 1.0,
#     '(BA,06:25,f)': 5.327693214, '(BA,06:25,l)': 22.66580704,
#     '(BA,09:05,f)': -9.210371325, '(BA,09:05,l)': -3.045572418,
#     '(CA,07:40,f)': 1.0, '(CA,07:40,l)': 1.0,
#     '(CA,08:15,f)': 1.0, '(CA,08:15,l)': 1.0
#   }
#   ```
# - Base value and ratio:
#   ```
#   v_0 = {'AB': 0.022767048, 'AC': 0.024033093, 'BA': 0.133469692, 'CA': 0.126816434}
#   r_0 = {'AB': 0.002224984, 'AC': 1.0, 'BA': 0.012916147, 'CA': 1.0}
#   ```
# - Demand:
#   ```
#   d = {'CA': '4807.43', 'AC': '4812.5', 'AB': '38965.86', 'BA': '33210.71'}
#   ```
# - Capacity consumption: $c_f = 1.2$
# - Flight capacity: $C = 187$
# - Big M: $M = 10,000,000$
# - Number of flights to select: $n_o = 2$
# - Flow sets:
#   ```
#   \sigma_A^+ = ['(BA,06:25)', '(BA,09:05)', '(CA,07:40)', '(CA,08:15)']
#   \sigma_A^- = ['(AB,11:20)', '(AB,06:40)', '(AB,07:55)', '(AC,09:45)']
#   \sigma_B^+ = ['(AB,11:20)', '(AB,06:40)', '(AB,07:55)']
#   \sigma_B^- = ['(BA,06:25)', '(BA,09:05)']
#   \sigma_C^+ = ['(AC,09:45)']
#   \sigma_C^- = ['(CA,07:40)', '(CA,08:15)']
#   ```

# ---

# ### Generated Code

# ```python
# import gurobipy as gp
# from gurobipy import GRB

# # Data
# p = {
#     '(AB,11:20,f)': 1140.3, '(AB,11:20,l)': 429.26,
#     '(AB,06:40,f)': 1142.6, '(AB,06:40,l)': 482.92,
#     '(AB,07:55,f)': 1138.19, '(AB,07:55,l)': 441.83,
#     '(AC,09:45,f)': 1292.44, '(AC,09:45,l)': 485.92,
#     '(BA,06:25,f)': 1131.33, '(BA,06:25,l)': 424.19,
#     '(BA,09:05,f)': 1123.69, '(BA,09:05,l)': 419.78,
#     '(CA,07:40,f)': 1285.82, '(CA,07:40,l)': 558.55,
#     '(CA,08:15,f)': 1284.5, '(CA,08:15,l)': 477.98
# }
# v = {
#     '(AB,11:20,f)': 3.896123449, '(AB,11:20,l)': 1,
#     '(AB,06:40,f)': 2.236888293, '(AB,06:40,l)': 0.521872683,
#     '(AB,07:55,f)': 2.236888293, '(AB,07:55,l)': 0.521872683,
#     '(AC,09:45,f)': 3.297109885, '(AC,09:45,l)': 1,
#     '(BA,06:25,f)': 1.813282947, '(BA,06:25,l)': 0.505261036,
#     '(BA,09:05,f)': 2.600413693, '(BA,09:05,l)': 1,
#     '(CA,07:40,f)': 0.865902564, '(CA,07:40,l)': 0.666898117,
#     '(CA,08:15,f)': 2.440052228, '(CA,08:15,l)': 1
# }
# r = {
#     '(AB,11:20,f)': 2.48264215, '(AB,11:20,l)': 32.96333763,
#     '(AB,06:40,f)': 1.91965885, '(AB,06:40,l)': -22.81661357,
#     '(AB,07:55,f)': 1.91965885, '(AB,07:55,l)': -22.81661357,
#     '(AC,09:45,f)': 1.0, '(AC,09:45,l)': 1.0,
#     '(BA,06:25,f)': 5.327693214, '(BA,06:25,l)': 22.66580704,
#     '(BA,09:05,f)': -9.210371325, '(BA,09:05,l)': -3.045572418,
#     '(CA,07:40,f)': 1.0, '(CA,07:40,l)': 1.0,
#     '(CA,08:15,f)': 1.0, '(CA,08:15,l)': 1.0
# }
# v_0 = {'AB': 0.022767048, 'AC': 0.024033093, 'BA': 0.133469692, 'CA': 0.126816434}
# r_0 = {'AB': 0.002224984, 'AC': 1.0, 'BA': 0.012916147, 'CA': 1.0}
# d = {'CA': 4807.43, 'AC': 4812.5, 'AB': 38965.86, 'BA': 33210.71}
# c_f = 1.2
# n_o = 2
# M = 10000000
# C = 187

# sigma_A_plus = ['(BA,06:25)', '(BA,09:05)', '(CA,07:40)', '(CA,08:15)']
# sigma_A_minus = ['(AB,11:20)', '(AB,06:40)', '(AB,07:55)', '(AC,09:45)']
# sigma_B_plus = ['(AB,11:20)', '(AB,06:40)', '(AB,07:55)']
# sigma_B_minus = ['(BA,06:25)', '(BA,09:05)']
# sigma_C_plus = ['(AC,09:45)']
# sigma_C_minus = ['(CA,07:40)', '(CA,08:15)']

# # All unique (l,k) pairs
# flight_pairs = set()
# for key in p.keys():
#     l, k, j = key.strip('()').split(',')
#     flight_pairs.add(f'({l},{k})')
# flight_pairs = list(flight_pairs)

# model = gp.Model("sales_based_lp")

# # Decision variables
# y = model.addVars(flight_pairs, vtype=GRB.BINARY, name="y")
# x = model.addVars(p.keys(), lb=0, name="x")
# x_o = model.addVars(v_0.keys(), lb=0, name="x_o")

# # Objective
# model.setObjective(gp.quicksum(p[key] * x[key] for key in p.keys()), GRB.MAXIMIZE)

# # Capacity constraints
# for l_k in flight_pairs:
#     l, k = l_k.strip('()').split(',')
#     model.addConstr(
#         c_f * x[f'({l},{k},f)'] + x[f'({l},{k},l)'] <= C,
#         name=f"capacity_{l}_{k}"
#     )

# # Balance constraints
# for l in r_0.keys():
#     model.addConstr(
#         r_0[l] * x_o[l] + gp.quicksum(r[key] * x[key] for key in p.keys() if key.startswith(f'({l},')) == d[l],
#         name=f"balance_{l}"
#     )

# # Scale constraints
# for key in p.keys():
#     l, k, j = key.strip('()').split(',')
#     model.addConstr(
#         x[key] / v[key] - x_o[l] / v_0[l] <= 0,
#         name=f"scale_{key}"
#     )

# # Big M constraints
# for key in p.keys():
#     l, k, j = key.strip('()').split(',')
#     model.addConstr(
#         x[key] <= M * y[f'({l},{k})'],
#         name=f"bigM_{key}"
#     )

# # Cardinality constraint
# model.addConstr(
#     gp.quicksum(y[l_k] for l_k in flight_pairs) <= n_o,
#     name="cardinality"
# )

# # Flow conservation constraints
# model.addConstr(
#     gp.quicksum(y[l_k] for l_k in sigma_A_plus) == gp.quicksum(y[l_k] for l_k in sigma_A_minus),
#     name="flow_A"
# )
# model.addConstr(
#     gp.quicksum(y[l_k] for l_k in sigma_B_plus) == gp.quicksum(y[l_k] for l_k in sigma_B_minus),
#     name="flow_B"
# )
# model.addConstr(
#     gp.quicksum(y[l_k] for l_k in sigma_C_plus) == gp.quicksum(y[l_k] for l_k in sigma_C_minus),
#     name="flow_C"
# )

# model.optimize()

# if model.status == GRB.OPTIMAL:
#     print("Optimal solution found:")
#     for v in model.getVars():
#         print(v.varName, v.x)
#     print("Optimal objective value:", model.objVal)
# else:
#     print("No optimal solution found.")
# ```

# ---

# **Note:** All required parameters and sets are included above. The model enforces all constraints as described, and the code is ready to run with Gurobi. Adjustments may be needed for v
# '''
    # result = convert_to_typora_markdown(result)
    # rendered_solution = markdown.markdown(result, extensions=["fenced_code", CodeHiliteExtension()])

    # return jsonify({
    #     'solution': rendered_solution,
    #     'status': 'success'
    # })
    problem_type = data.get('problem_type')
    if not all([query, api_key, problem_type]):
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        # Process based on problem type
        result = process_problem_type(query, api_key, problem_type)
        print(result)
        with open("NP_Flow.md", "w") as file:
            file.write(result)
        # result.write('1.md')
        # Render the result with proper markdown and code highlighting
        # rendered_solution = render_markdown_with_highlight(result)
        # rendered_solution = result
        rendered_solution = markdown.markdown(result, extensions=["fenced_code", CodeHiliteExtension()])


        return jsonify({
            'solution': rendered_solution,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# def render_markdown_with_highlight(md_text):
#     # First, process LaTeX-style math formulas to HTML
#     # Convert $$ $$ to <div class="math"> and $\) to <span class="math">
#     md_text = re.sub(r'\\$$(.*?)\\$$', r'<div class="math">\1</div>', md_text, flags=re.DOTALL)
#     md_text = re.sub(r'\\\((.*?)\\\)', r'<span class="math">\1</span>', md_text)
    
#     # Define patterns for code blocks
#     python_code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
#     generic_code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    
#     # Function to replace Python code blocks
#     def replace_python_code(match):
#         code = match.group(1).strip()
#         return f'<pre><code class="language-python">{code}</code></pre>'
    
#     # Function to replace generic code blocks
#     def replace_generic_code(match):
#         code = match.group(1).strip()
#         return f'<pre><code>{code}</code></pre>'
    
#     # Replace Python code blocks first
#     html_content = python_code_pattern.sub(replace_python_code, md_text)
#     # Then replace any remaining generic code blocks
#     html_content = generic_code_pattern.sub(replace_generic_code, html_content)
    
#     # Convert the rest of the markdown to HTML using extensions for extra features and code highlighting
#     html_content = markdown.markdown(html_content, extensions=['extra', 'codehilite'])
    
#     return html_content
import re
import markdown
from bleach import clean

def render_markdown_with_highlight(md_text):
    # Convert Typora-style math formulas to HTML
    # Convert $$ $$ to <div class="math"> and $ $ to <span class="math">
    md_text = re.sub(r'\$\$(.*?)\$\$', r'<div class="math">\1</div>', md_text, flags=re.DOTALL)
    md_text = re.sub(r'\$(.*?)\$', r'<span class="math">\1</span>', md_text)

    # Define patterns for code blocks
    python_code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    generic_code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    
    # Function to replace Python code blocks
    def replace_python_code(match):
        code = match.group(1).strip()
        return f'<pre><code class="language-python">{code}</code></pre>'
    
    # Function to replace generic code blocks
    def replace_generic_code(match):
        code = match.group(1).strip()
        return f'<pre><code>{code}</code></pre>'
    
    # Replace Python code blocks first
    html_content = python_code_pattern.sub(replace_python_code, md_text)
    # Then replace any remaining generic code blocks
    html_content = generic_code_pattern.sub(replace_generic_code, html_content)
    
    # Convert the rest of the markdown to HTML using extensions for extra features and code highlighting
    # html_content = markdown.markdown(html_content, extensions=['extra', 'codehilite'])
    html_content = markdown.markdown(html_content, ["fenced_code", CodeHiliteExtension()])

    # Sanitize the HTML content to prevent XSS attacks
    sanitized_content = clean(html_content, tags=['div', 'span', 'pre', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'blockquote', 'table', 'thead', 'tbody', 'tr', 'th', 'td'], attributes={'div': ['class'], 'span': ['class'], 'code': ['class']})

    return sanitized_content

# Ensure other parts of your application remain unchanged

# def render_markdown_with_highlight(md_text):
#     # Process LaTeX-style math formulas to HTML
#     # Convert $$ $$ to <div class="math"> and \( \) to <span class="math">
#     md_text = re.sub(r'\$\$(.*?)\$\$', r'<div class="math">\1</div>', md_text, flags=re.DOTALL)
#     md_text = re.sub(r'\$(.*?)\$', r'<span class="math">\1</span>', md_text)

#     # Define patterns for code blocks
#     python_code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
#     generic_code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    
#     # Function to replace Python code blocks
#     def replace_python_code(match):
#         code = match.group(1).strip()
#         return f'<pre><code class="language-python">{code}</code></pre>'
    
#     # Function to replace generic code blocks
#     def replace_generic_code(match):
#         code = match.group(1).strip()
#         return f'<pre><code>{code}</code></pre>'
    
#     # Replace Python code blocks first
#     html_content = python_code_pattern.sub(replace_python_code, md_text)
#     # Then replace any remaining generic code blocks
#     html_content = generic_code_pattern.sub(replace_generic_code, html_content)
    
#     # Convert the rest of the markdown to HTML using extensions for extra features and code highlighting
#     html_content = markdown.markdown(html_content, extensions=['extra', 'codehilite'])

#     # Sanitize the HTML content to prevent XSS attacks
#     sanitized_content = clean(html_content, tags=['div', 'span', 'pre', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'blockquote', 'table', 'thead', 'tbody', 'tr', 'th', 'td'], attributes={'div': ['class'], 'span': ['class'], 'code': ['class']})

#     return sanitized_content

# def solve():
#     data = request.get_json()
#     query = data.get('query')
#     api_key = data.get('api_key')
#     problem_type = data.get('problem_type')
    
#     if not all([query, api_key, problem_type]):
#         return jsonify({'error': 'Missing required parameters'}), 400
        
#     try:
#         # Process based on problem type
#         result = process_problem_type(query, api_key, problem_type)

#         # Use markdown library to render the solution
#         rendered_solution = markdown.markdown(result)

#         return jsonify({
#             'solution': rendered_solution,
#             'status': 'success'
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

def convert_to_typora_markdown(content):
    content = content.replace(r'\[', '$$').replace(r'\]', '$$') 
    content = content.replace(r'\( ', '$').replace(r' \)', '$') 
    content = content.replace(r'\(', '$').replace(r'\)', '$')
    content = content.replace(r'\t ', '\\t').replace(r' \f', '\\f') 
    content = content.replace(r'\{ ', '\\{').replace(r' \}', '\\}') 

    return content

def extract_problem_type(output_text):
    pattern = r'(Network Revenue Management|Resource Allocation|Transportation|Sales-Based Linear Programming|SBLP|Facility Location|Others without CSV|Others without csv)'
    match = re.search(pattern, output_text, re.IGNORECASE)
    return match.group(0) if match else "Others with CSV"

def process_problem_type(query, api_key, problem_type):
    # Read global uploaded files
    global uploaded_files

    # 2. Select RAG reference file and few-shot template based on problem type
    if problem_type == "Network Revenue Management":
        # 1. Preprocess uploaded CSV data
        data = []
        for df_index, (file_name, df) in enumerate(uploaded_files):
            data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")
            for i, r in df.iterrows():
                description = ""
                description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                data.append(description + "\n")
        documents = [content for content in data]

        rag_file = "RAG_Example_NRM.csv"
        model_name = "gpt-4o"
        few_shot_examples = """
Question: Based on the following description and data, please formulate a linear programming model. [Problem Description]

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file.

Action: CSVQA

Action Input: Retrieve the product data related to [Related Content] to formulate the linear programming model.

Observation: [Example Data]

Final Answer: 
[Linear Programming Model]
"""

        # 3. Build vector database
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_texts(documents, embeddings)
        retriever = vectors.as_retriever(search_kwargs={'k': 250})

        # 4. Build RetrievalQA chain
        llm2 = ChatOpenAI(temperature=0.0, model_name=model_name, openai_api_key=api_key)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm2,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )
        qa_tool = Tool(
            name="CSVQA",
            func=qa_chain.run,
            description="Use this tool to answer questions based on the provided CSV data and retrieve product data similar to the input query."
        )

        # 5. Build Agent prompt
        prefix = f"""You are an assistant that generates linear programming models based on the user's description and provided CSV data.

    Please refer to the following example and generate the answer in the same format:

    {few_shot_examples}

    When you need to retrieve information from the CSV file, use the provided tool.

    """
        suffix = """

    Begin!

    User Description: {input}
    {agent_scratchpad}"""

        agent2 = initialize_agent(
            tools=[qa_tool],
            llm=llm2,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={
                "prefix": prefix,
                "suffix": suffix,
            },
            verbose=True,
            handle_parsing_errors=True,
        )

        # 6. Execute Agent reasoning
        result = agent2.invoke(query)
    elif problem_type == "Resource Allocation":
        # 1. Preprocess uploaded CSV data
        data = []
        for df_index, (file_name, df) in enumerate(uploaded_files):
            data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")
            for i, r in df.iterrows():
                description = ""
                description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                data.append(description + "\n")
        documents = [content for content in data]

        rag_file = "RAG_Example_RA.csv"
        model_name = "gpt-4o"
        few_shot_examples = """
Question: Based on the following description and data, please formulate a linear programming model. [Problem Description]

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. Particularly, I should figure out which product and how many of them in total in the CSV should be considered.

Action: CSVQA

Action Input: [Problem Description] Retrieve the capacity data and products data to formulate the linear programming model.

Observation: [Example Data]

Thought: Now that I have the necessary data, I can construct the objective function and constraints. And I should generate the answer only using the format similar to the result from the observation. The expressions should not be simplified or abbreviated. I need to retrieve products similar to [Related Content] from the CSV file to formulate the linear programming model.

Final Answer: 
[Linear Programming Model]
"""

        # 3. Build vector database
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_texts(documents, embeddings)
        retriever = vectors.as_retriever(search_kwargs={'k': 250})

        # 4. Build RetrievalQA chain
        llm2 = ChatOpenAI(temperature=0.0, model_name=model_name, openai_api_key=api_key)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm2,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )
        qa_tool = Tool(
            name="CSVQA",
            func=qa_chain.run,
            description="Use this tool to answer questions based on the provided CSV data and retrieve product data similar to the input query."
        )

        # 5. Build Agent prompt
        prefix = f"""You are an assistant that generates linear programming models based on the user's description and provided CSV data.

    Please refer to the following example and generate the answer in the same format:

    {few_shot_examples}

    When you need to retrieve information from the CSV file, use the provided tool.

    """
        suffix = """

    Begin!

    User Description: {input}
    {agent_scratchpad}"""

        agent2 = initialize_agent(
            tools=[qa_tool],
            llm=llm2,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={
                "prefix": prefix,
                "suffix": suffix,
            },
            verbose=True,
            handle_parsing_errors=True,
        )

        # 6. Execute Agent reasoning
        result = agent2.invoke(query)
    elif problem_type == "Transportation":
        # 1. Preprocess uploaded CSV data
        data = []
        for df_index, (file_name, df) in enumerate(uploaded_files):
            data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")
            for i, r in df.iterrows():
                description = ""
                description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                data.append(description + "\n")
        documents = [content for content in data]

        rag_file = "RAG_Example_TP.csv"
        model_name = "gpt-4"
        few_shot_examples = """
Question: Based on the following transportation problem description and data, please formulate a complete linear programming model using real data from retrieval. [Problem Description]

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. Particularly, I should figure out which product and how many of them in total in the CSV should be considered.

Action: CSVQA

Action Input: Retrieve the relevant data to formulate the linear programming model [Problem Description].

Observation: [Example Data]

Thought: Now that I have the necessary data, I can construct the objective function and constraints. And the answer I generate should only be similar to the format below. The expressions should not be simplified or abbreviated.

Final Answer: 
[Linear Programming Model]
"""

        # 3. Build vector database
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_texts(documents, embeddings)
        retriever = vectors.as_retriever(search_kwargs={'k': 250})

        # 4. Build RetrievalQA chain
        llm2 = ChatOpenAI(temperature=0.0, model_name=model_name, openai_api_key=api_key)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm2,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )
        qa_tool = Tool(
            name="CSVQA",
            func=qa_chain.run,
            description="Use this tool to answer questions based on the provided CSV data and retrieve product data similar to the input query."
        )

        # 5. Build Agent prompt
        prefix = f"""You are an assistant that generates linear programming models based on the user's description and provided CSV data.

    Please refer to the following example and generate the answer in the same format:

    {few_shot_examples}

    When you need to retrieve information from the CSV file, use the provided tool.

    """
        suffix = """

    Begin!

    User Description: {input}
    {agent_scratchpad}"""

        agent2 = initialize_agent(
            tools=[qa_tool],
            llm=llm2,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={
                "prefix": prefix,
                "suffix": suffix,
            },
            verbose=True,
            handle_parsing_errors=True,
        )

        # 6. Execute Agent reasoning
        result = agent2.invoke(query)
    elif problem_type == "Sales-Based Linear Programming" or problem_type == "SBLP":
        def LoadFiles():
            for df_index, (file_name, df) in enumerate(uploaded_files):
                if 'v1.csv' in file_name:
                    v1 = df
                elif 'v2.csv' in file_name:
                    v2 = df
                elif 'od_demand.csv' in file_name:
                    demand = df
                elif 'flight.csv' in file_name:
                    flight = df
            return v1,v2,demand,flight
        
        def New_Vectors_Flight(query):
            v1,v2,demand,df = LoadFiles()
            new_docs = []
            for _, row in df.iterrows():
                new_docs.append(Document(
                    page_content=f"OD={row['Oneway_OD']},Departure_Time_Flight1={row['Departure Time']},Oneway_Product={row['Oneway_Product']},avg_price={row['Avg Price']}",
                    metadata={
                        'OD': row['Oneway_OD'],
                        'time': row['Departure Time'],
                        'product': row['Oneway_Product'],
                        'avg_price': row['Avg Price']
                    }
                ))

            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            new_vectors = FAISS.from_documents(new_docs, embeddings)
            return new_vectors

        def New_Vectors_Demand(query):
            v1,v2,demand,df = LoadFiles()
            new_docs = []
            for _, row in demand.iterrows():
                new_docs.append(Document(
                    page_content=f"OD={row['Oneway_OD']}, avg_pax={row['Avg Pax']}",
                    metadata={
                        'OD': row['Oneway_OD'],
                        'avg_pax': row['Avg Pax']
                    }
                ))

            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            new_vectors = FAISS.from_documents(new_docs, embeddings)
            return new_vectors


        def retrieve_key_information(query):
            flight_pattern = r"\(OD\s*=\s*\(\s*'(\w+)'\s*,\s*'(\w+)'\s*\)\s+AND\s+Departure\s+Time='(\d{1,2}:\d{2})'\)"
            matches = re.findall(flight_pattern, query)

            flight_strings = [f"(OD = ('{origin}', '{destination}') AND Departure Time='{departure_time}')" for origin, destination, departure_time in matches]

            new_query = "Retrieve avg_price, avg_pax, value_list, ratio_list, value_0_list, ratio_0_list, and flight_capacity for the following flights:\n" + ", ".join(flight_strings) + "."

            return new_query
        def retrieve_time_period(departure_time):
            intervals = {
                '12pm~6pm': (time_class(12, 0), time_class(18, 0)),
                '6pm~10pm': (time_class(18, 0), time_class(22, 0)),
                '10pm~8am': (time_class(22, 0), time_class(8, 0)),
                '8am~12pm': (time_class(8, 0), time_class(12, 0))
            }

            if isinstance(departure_time, str):
                try:
                    hours, minutes = map(int, departure_time.split(':'))
                    departure_time = time_class(hours, minutes)
                except ValueError:
                    raise ValueError("Time format should be: 'HH:MM'")

            for interval_name, (start, end) in intervals.items():
                if start < end:
                    if start <= departure_time < end:
                        return interval_name
                else:
                    if departure_time >= start or departure_time < end:
                        return interval_name

            return "Unknown"
        def retrieve_parameter(O,time_interval,product):
            v1,v2,df,demand = LoadFiles()
            time_interval = f'({time_interval})'
            key = product + '*' + time_interval
            _value_ = 0
            _ratio_ = 0
            no_purchase_value = 0
            no_purchase_value_ratio = 0
            subset = v1[v1['OD Pairs'] == O]
            if key in subset.columns and not subset.empty:
                _value_ = subset[key].values[0]

            subset2 = v2[v2['OD Pairs'] == O]
            if key in subset2.columns and not subset2.empty:
                _ratio_ = subset2[key].values[0]

            if 'no_purchase' in subset.columns and not subset.empty:
                no_purchase_value = subset['no_purchase'].values[0]

            if 'no_purchase' in subset2.columns and not subset2.empty:
                no_purchase_value_ratio = subset2['no_purchase'].values[0]
            return _value_,_ratio_,no_purchase_value,no_purchase_value_ratio

        def generate_coefficients(OD,time):
            value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
            value_0_list, ratio_0_list = [], []

            departure_time = datetime.strptime(time, '%H:%M').time()
            time_interval = retrieve_time_period(departure_time)
            value_1,ratio_1,value_0,ratio_0 = retrieve_parameter(OD,time_interval,'Eco_flexi')

            value_2,ratio_2,value_0,ratio_0 = retrieve_parameter(OD,time_interval,'Eco_lite')

            return value_1,ratio_1,value_2,ratio_2,value_0,ratio_0


        def clean_text_preserve_newlines(text):
            cleaned = re.sub(r'\x1b$$[0-9;]*[mK]', '', text)
            cleaned = re.sub(r'[^\x20-\x7E\n]', '', cleaned)
            cleaned = re.sub(r'(\n\s+)(\w+\s*=)', r'\n\2', cleaned)
            cleaned = re.sub(r'$$\s+', '[', cleaned)
            cleaned = re.sub(r'\s+$$', ']', cleaned)
            cleaned = re.sub(r',\s+', ', ', cleaned)

            return cleaned

        def csv_qa_tool_flow(query: str):
            new_vectors = New_Vectors_Flight(query)
            matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
            num_match = re.search(r"optimal (\d+) flights", query)
            num_flights = int(num_match.group(1)) if num_match else None  # 3
            capacity_match = re.search(r"Eco_flex ticket consumes (\d+\.?\d*)\s*units", query)
            if matches == []:
                pattern = r"\(\('(\w+)','(\w+)'\),\s*'(\d{1,2}:\d{2})'\)"
                matches_2 = re.findall(pattern, query)

            if capacity_match:
                eco_flex_capacity = capacity_match.group(1)
            else:
                eco_flex_capacity = 1.2

            sigma_inflow_A = []
            sigma_outflow_A = []
            sigma_inflow_B = []
            sigma_outflow_B = []
            sigma_inflow_C = []
            sigma_outflow_C = []

            if matches == []:
                matches = matches_2
                for origin,destination,time in matches:
                    if origin == 'A':
                        flight_name = f"({origin}{destination},{time})"
                        sigma_inflow_A.append(flight_name)
                    elif origin == 'B':
                        flight_name = f"({origin}{destination},{time})"
                        sigma_inflow_B.append(flight_name)
                    elif origin == 'C':
                        flight_name = f"({origin}{destination},{time})"
                        sigma_inflow_C.append(flight_name)
                    elif destination == 'A':
                        flight_name = f"({origin}{destination},{time})"
                        sigma_outflow_A.append(flight_name)
                    elif destination == 'B':
                        flight_name = f"({origin}{destination},{time})"
                        sigma_outflow_B.append(flight_name)
                    elif destination == 'C':
                        flight_name = f"({origin}{destination},{time})"
                        sigma_outflow_C.append(flight_name)
            
            else:
                a_origin_flights_A_out = [
                    (od, time)
                    for (od, time) in matches
                    if od[2] == 'A'
                ]

                a_origin_flights_B_out = [
                    (od, time)
                    for (od, time) in matches
                    if od[2] == 'B'
                ]

                a_origin_flights_C_out = [
                    (od, time)
                    for (od, time) in matches
                    if od[2] == 'C'
                ]

                a_origin_flights_A = [
                    (od, time)
                    for (od, time) in matches
                    if od[7] == 'A'
                ]

                a_origin_flights_B = [
                    (od, time)
                    for (od, time) in matches
                    if od[7] == 'B'
                ]

                a_origin_flights_C = [
                    (od, time)
                    for (od, time) in matches
                    if od[7] == 'C'
                ]

                for od, time in a_origin_flights_A:
                    origin = od[2]
                    destination = od[7]
                    flight_name = f"({origin}{destination},{time})"
                    sigma_inflow_A.append(flight_name)

                for od, time in a_origin_flights_B:
                    origin = od[2]
                    destination = od[7]
                    flight_name = f"({origin}{destination},{time})"
                    sigma_inflow_B.append(flight_name)

                for od, time in a_origin_flights_C:
                    origin = od[2]
                    destination = od[7]
                    flight_name = f"({origin}{destination},{time})"
                    sigma_inflow_C.append(flight_name)

                for od, time in a_origin_flights_A_out:
                    origin = od[2]
                    destination = od[7]
                    flight_name = f"({origin}{destination},{time})"
                    sigma_outflow_A.append(flight_name)

                for od, time in a_origin_flights_B_out:
                    origin = od[2]
                    destination = od[7]
                    flight_name = f"({origin}{destination},{time})"
                    sigma_outflow_B.append(flight_name)

                for od, time in a_origin_flights_C_out:
                    origin = od[2]
                    destination = od[7]
                    flight_name = f"({origin}{destination},{time})"
                    sigma_outflow_C.append(flight_name)

            avg_price, x, x_o, ratio_0_list, ratio_list, value_list, value_0_list, avg_pax = {}, {}, {}, {}, {}, {}, {}, {}
            y = {}
            N_l = {"AB":[],"AC":[],"BA":[],"CA":[]}

            if matches == []:
                matches = matches_2
                for origin,destination,time in matches:
                    od = str((origin, destination))
                code_f = f"({origin}{destination},{time},f)"
                code_l = f"({origin}{destination},{time},l)"
                code_o = f"{origin}{destination}"
                x[code_f] = f"x_{origin}{destination}_{time}_f"
                x[code_l] = f"x_{origin}{destination}_{time}_l"
                code_y = f"({origin}{destination},{time})"
                y[code_y] = f"y_{origin}{destination}_{time}"
                retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})

                doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_price=")
                for doc in doc_1:
                    content = doc.page_content
                    pattern = r',\s*(?=\w+=)'
                    parts = re.split(pattern, content)

                    pairs = [p.strip().replace('"', "'") for p in parts]
                    for pair in pairs:
                        key, value = pair.split('=')
                        if key == 'avg_price':
                            avg_price[code_f] = value


                doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_price=")
                for doc in doc_2:
                    content = doc.page_content
                    pattern = r',\s*(?=\w+=)'
                    parts = re.split(pattern, content)

                    pairs = [p.strip().replace('"', "'") for p in parts]
                    for pair in pairs:
                        key, value = pair.split('=')
                        if key == 'avg_price':
                            avg_price[code_l] = value

                value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(od,time)

                ratio_list[code_f] = ratio_1
                ratio_list[code_l] = ratio_2
                value_list[code_f] = value_1
                value_list[code_l] = value_2
                ratio_0_list[code_o] = ratio_0
                value_0_list[code_o] = value_0
            else:
                for match in matches:
                    origin = match[0][2]
                    destination = match[0][7]
                    time = match[1]
                    od = str((origin, destination))
                    code_f = f"({origin}{destination},{time},f)"
                    code_l = f"({origin}{destination},{time},l)"
                    code_o = f"{origin}{destination}"
                    x[code_f] = f"x_{origin}{destination}_{time}_f"
                    x[code_l] = f"x_{origin}{destination}_{time}_l"
                    code_y = f"({origin}{destination},{time})"
                    y[code_y] = f"y_{origin}{destination}_{time}"
                    retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})

                    doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_price=")
                    for doc in doc_1:
                        content = doc.page_content
                        pattern = r',\s*(?=\w+=)'
                        parts = re.split(pattern, content)

                        pairs = [p.strip().replace('"', "'") for p in parts]
                        for pair in pairs:
                            key, value = pair.split('=')
                            if key == 'avg_price':
                                avg_price[code_f] = value


                    doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_price=")
                    for doc in doc_2:
                        content = doc.page_content
                        pattern = r',\s*(?=\w+=)'
                        parts = re.split(pattern, content)

                        pairs = [p.strip().replace('"', "'") for p in parts]
                        for pair in pairs:
                            key, value = pair.split('=')
                            if key == 'avg_price':
                                avg_price[code_l] = value

                    value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(od,time)

                    ratio_list[code_f] = ratio_1
                    ratio_list[code_l] = ratio_2
                    value_list[code_f] = value_1
                    value_list[code_l] = value_2
                    ratio_0_list[code_o] = ratio_0
                    value_0_list[code_o] = value_0


            od_matches = re.findall(
            r"OD\s*=\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", 
            query
            )
            if od_matches == []:
                pattern = r"\(\('(\w+)','(\w+)'\)"
                matches = re.findall(pattern, query)
                od_matches = matches
            
            od_matches = list(set(od_matches))

            new_vectors_demand = New_Vectors_Demand(query)
            for origin, dest in od_matches:
                od = str((origin, dest))
                code_o = f"{origin}{dest}"
                x_o[code_o] = f"x_{origin}{dest}_o"
                retriever = new_vectors_demand.as_retriever(search_kwargs={'k': 1})
            
                doc_1= retriever.get_relevant_documents(f"OD={od}, avg_pax=")
                content = doc_1[0].page_content

                pattern = r',\s*(?=\w+=)'
                parts = re.split(pattern, content)
                pairs = [p.strip().replace('"', "'") for p in parts]

                for pair in pairs:
                    key, value = pair.split('=')
                    if key == 'avg_pax':
                        avg_pax[code_o] = value
                    
            doc = f"y = {y}\n"
            doc = f"p={avg_price} \n v ={value_list}\n r={ratio_list}\n"
            doc += f"v_0={value_0_list}\n r_0={ratio_0_list}\n"
            doc += f"d={avg_pax}\n"
            doc += f"c_f={eco_flex_capacity}\n"
            doc += f"n_o={num_flights}\n"
            doc += f"\sigma_A^+={sigma_inflow_A}\n"
            doc += f"\sigma_A^-={sigma_outflow_A}\n"
            doc += f"\sigma_B^+={sigma_inflow_B}\n"
            doc += f"\sigma_B^-={sigma_outflow_B}\n"
            doc += f"\sigma_C^+={sigma_inflow_C}\n"
            doc += f"\sigma_C^-={sigma_outflow_C}\n"
            doc += f"M = 10000000\n"
            doc += f"C=187\n"

            return doc

        def retrieve_similar_docs(query,retriever):
            
            similar_docs = retriever.get_relevant_documents(query)

            results = []
            for doc in similar_docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            return results

        def FlowAgent(query):
            loader = CSVLoader(file_path="RAG_Example_SBLP_Flow.csv", encoding="utf-8")
            data = loader.load()
            documents = data
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectors = FAISS.from_documents(documents, embeddings)
            retriever = vectors.as_retriever(search_kwargs={'k': 1})
            similar_results = retrieve_similar_docs(query,retriever)
            problem_description = similar_results[0]['content'].replace("prompt:", "").strip()  
            example_matches = retrieve_key_information(problem_description)
            example_data_description = csv_qa_tool_flow(example_matches)
            example_data_description = example_data_description.replace('{', '{{')
            example_data_description = example_data_description.replace('}', '}}')    
            fewshot_example = f'''
            Question: {problem_description}

            Thought: I need to retrive the required OD and Departure Time information.

            Action: CName

            Action Input: {problem_description}

            Observation: {example_matches}

            Thought: I need to retrieve relevant information from 'information1.csv' for the given OD and Departure Time values. Next I need to retrieve the relevant coefficients from v1 and v2 based on the retrieved ticket information. 
            
            Action: CSVQA

            Action Input:  {example_matches}

            Observation: {example_data_description}

            Thought: I now have all the required information from the CSV, including avg_price, avg_pax, value_list, ratio_list, value_0_list, ratio_0_list, flight_capacity, capacity_consum, option_num and so on for the specified flights. I will now formulate the SBLP model in markdown format, following the example provided, including all constraints, retrieved information, and generated code.


            Final Answer:
### Objective Function

$$
\max \quad \sum_(l,k,j) p[(l,k,j)] \cdot x[(l,k,j)]
$$

### Constraints

#### 1. Capacity Constraints

For each flight (OD pair $l$, departure time $k$):

$$
c_f \cdot x[(l,k,f)] + x[(l,k,l)] \leq C
$$

#### 2. Balance Constraints

For each OD pair $l$:

$$
r_0[l] \cdot x_o[l] + \sum_k \sum_j r[(l,k,j)] \cdot x[(l,k,j)] = d[l]
$$

#### 3. Scale Constraints

For each ticket option \((l,k,j)\):

$$
x[(l,k,j)] \cdot v_0[l]  - x_o[l] \cdot v[(l,k,j)] \leq 0
$$

#### 4. Big M Constraints

For each ticket option \((l,k,j)\):

$$
x[(l,k,j)] \leq 10000000 \cdot y[(l,k)]
$$

#### 5. Cardinality Constraint

$$
\sum_(l,k) y[(l,k)] \leq n_o
$$


#### 6. Flow Conservation Constraints

$$
\sum_(l,k)  \in \sigma_A^+ y[(l,k)] = \sum_(l,k)  \in \sigma_A^- y[(l,k)]
$$
$$
\sum_(l,k)  \in \sigma_B^+ y[(l,k)] = \sum_(l,k)  \in \sigma_B^- y[(l,k)]
$$
$$
\sum_(l,k)  \in \sigma_C^+ y[(l,k)] = \sum_(l,k)  \in \sigma_C^- y[(l,k)]
$$

#### 7. Nonnegativity Constraints

$$
x[(l,k,j)] \geq 0, \quad x_o[l] \geq 0
$$
- x = p.keys()
- x_o = d.keys()

#### 8. Binary Constraints

$$
y[(l,k)] is binary
$$

### Retrieved Information
{example_data_description}

### Generated Code

```python
import gurobipy as gp
from gurobipy import GRB
{example_data_description}
total_set = \sigma_A^+ + \sigma_A^- + \sigma_B^+ + \sigma_B^- + \sigma_C^+ + \sigma_C^-
total_set = list(set(total_set)) 
model = gp.Model("sales_based_lp")
y = model.addVars(total_set, vtype=GRB.BINARY, name="y")  
x = model.addVars(p.keys(), lb=0, name="x") 
x_o = model.addVars(value_0_list.keys(), lb=0, name="x_o")  

model.setObjective(gp.quicksum(p[key] * x[key] for key in p.keys()), GRB.MAXIMIZE)

paired_keys = []
for i in range(0, len(p.keys()), 2):  
    if i + 1 < len(p.keys()):  
        names = list(p.keys())
        model.addConstr(
            c_f* x[names[i]] + x[names[i + 1]] <= 187,
            name=f"capacity_constraint"
        )

for l in r_0.keys():
    temp = 0
    for key in r.keys():
        if l in key:
            temp += r[key] * x[key]
    model.addConstr(
        float(r_0[l]) * x_o[l] + temp == float(d[l])
    )

for i in v.keys():
    for l in r_0.keys():
        if l in i:
            model.addConstr(
                v_0[l]  * x[i] <= v[i] * x_o[l]
            )


for key in p.keys():
    for lk in total_set:
        if lk.split(',')[0].strip('(').strip(')')==key.split(',')[0].strip('(').strip(')') and lk.split(',')[1].strip('(').strip(')')==key.split(',')[1].strip('(').strip(')'):
            model.addConstr(
                x[key] <= 10000000 * y[lk],  
                name=f"big_m_constraint"
            )

model.addConstr(gp.quicksum(y[lk] for lk in total_set) <= n_o,
        name=f"cardinality_constraint"
    )


model.addConstr(
    gp.quicksum(y[inflow] for inflow in \sigma_A^+) == gp.quicksum(y[outflow] for outflow in \sigma_A^-),
    name=f"flow_conservation_A"
)


model.addConstr(
    gp.quicksum(y[inflow] for inflow in \sigma_B^+) == gp.quicksum(y[outflow] for outflow in \sigma_B^-),
    name=f"flow_conservation_B"
)
    
model.addConstr(
    gp.quicksum(y[inflow] for inflow in \sigma_C^+) == gp.quicksum(y[outflow] for outflow in \sigma_C^-),
    name=f"flow_conservation_C"
)

model.optimize()


if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for v in model.getVars():
        print(v.varName, v.x)
    print("Optimal objective value:", model.objVal)
else:
    print("No optimal solution found.")
            '''

            tools = [Tool(name="CSVQA", func=csv_qa_tool_flow, description="Retrieve flight data."),Tool(name="CName", func=retrieve_key_information, description="Retrieve flight information.")]

            llm = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=api_key)
            prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

            Please refer to the following example and generate the answer in the same format:

            {fewshot_example}

            Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

            When you need to retrieve information from the CSV file, and write SBLP formulation by using the provided tools.

            """

            suffix = """

            Begin!

            User Description: {input}
            {agent_scratchpad}"""


            agent2 = initialize_agent(
                tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                agent_kwargs={
                    "prefix": prefix,
                    "suffix": suffix,
                },
                verbose=True,
                handle_parsing_errors=True
            )
            return agent2

        def policy_sblp_flow_model_code(query):
            agent2 = FlowAgent(query)
            llm_code = ChatOpenAI(
                temperature=0.0, model_name="gpt-4.1", openai_api_key=api_key
            )
            result = agent2.invoke({"input": query})
            output_model = result['output']

            return output_model

        def ProcessPolicyFlow(query):
            output_model = policy_sblp_flow_model_code(query)
            return output_model        
        
        def csv_qa_tool_CA(query: str):
                           
            new_vectors = New_Vectors_Flight(query)
            matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
            capacity_match = re.search(r"Eco_flex ticket consumes (\d+\.?\d*)\s*units", query)

            if capacity_match:
                eco_flex_capacity = capacity_match.group(1)
            else:
                eco_flex_capacity = 1.2

            avg_price, x, x_o, ratio_0_list, ratio_list, value_list, value_0_list, avg_pax = {}, {}, {}, {}, {}, {}, {}, {}

            if matches == []:
                pattern = r"\('(\w+)'\s*,\s*'(\d{1,2}:\d{2})'\)"
                matches_2 = re.findall(pattern, query)
                matches = matches_2
                for od,time in matches:
                    origin = od[0]
                    destination = od[1]
                    od = str((origin, destination))
                    code_f = f"({origin}{destination},{time},f)"
                    code_l = f"({origin}{destination},{time},l)"
                    code_o = f"{origin}{destination}"
                    x[code_f] = f"x_{origin}{destination}_{time}_f"
                    x[code_l] = f"x_{origin}{destination}_{time}_l"
                    retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})

                    doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_price=")
                    for doc in doc_1:
                        content = doc.page_content
                        pattern = r',\s*(?=\w+=)'
                        parts = re.split(pattern, content)

                        pairs = [p.strip().replace('"', "'") for p in parts]
                        for pair in pairs:
                            key, value = pair.split('=')
                            if key == 'avg_price':
                                avg_price[code_f] = value


                    doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_price=")
                    for doc in doc_2:
                        content = doc.page_content
                        pattern = r',\s*(?=\w+=)'
                        parts = re.split(pattern, content)

                        pairs = [p.strip().replace('"', "'") for p in parts]
                        for pair in pairs:
                            key, value = pair.split('=')
                            if key == 'avg_price':
                                avg_price[code_l] = value

                    value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(od,time)

                    ratio_list[code_f] = ratio_1
                    ratio_list[code_l] = ratio_2
                    value_list[code_f] = value_1
                    value_list[code_l] = value_2
                    ratio_0_list[code_o] = ratio_0
                    value_0_list[code_o] = value_0
            else:
                for match in matches:
                    origin = match[0][2]
                    destination = match[0][7]
                    time = match[1]
                    od = str((origin, destination))
                    code_f = f"({origin}{destination},{time},f)"
                    code_l = f"({origin}{destination},{time},l)"
                    code_o = f"{origin}{destination}"
                    x[code_f] = f"x_{origin}{destination}_{time}_f"
                    x[code_l] = f"x_{origin}{destination}_{time}_l"

                    retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})
                    doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_price=")
                    for doc in doc_1:
                        content = doc.page_content
                        pattern = r',\s*(?=\w+=)'
                        parts = re.split(pattern, content)

                        pairs = [p.strip().replace('"', "'") for p in parts]
                        for pair in pairs:
                            key, value = pair.split('=')
                            if key == 'avg_price':
                                avg_price[code_f] = value


                    doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_price=")
                    for doc in doc_2:
                        content = doc.page_content
                        pattern = r',\s*(?=\w+=)'
                        parts = re.split(pattern, content)

                        pairs = [p.strip().replace('"', "'") for p in parts]
                        for pair in pairs:
                            key, value = pair.split('=')
                            if key == 'avg_price':
                                avg_price[code_l] = value

                    value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(od,time)

                    ratio_list[code_f] = ratio_1
                    ratio_list[code_l] = ratio_2
                    value_list[code_f] = value_1
                    value_list[code_l] = value_2
                    ratio_0_list[code_o] = ratio_0
                    value_0_list[code_o] = value_0

            od_matches = re.findall(
            r"OD\s*=\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", 
            query
            )

            if od_matches == []:
                pattern = r"\(\('(\w+)','(\w+)'\)"
                matches = re.findall(pattern, query)
                od_matches = matches
            
            od_matches = list(set(od_matches))

            new_vectors_demand = New_Vectors_Demand(query)
            for origin, dest in od_matches:
                od = str((origin, dest))
                code_o = f"{origin}{dest}"
                x_o[code_o] = f"x_{origin}{dest}_o"
                retriever = new_vectors_demand.as_retriever(search_kwargs={'k': 1})
            
                doc_1= retriever.get_relevant_documents(f"OD={od}, avg_pax=")
                content = doc_1[0].page_content

                pattern = r',\s*(?=\w+=)'
                parts = re.split(pattern, content)
                pairs = [p.strip().replace('"', "'") for p in parts]

                for pair in pairs:
                    key, value = pair.split('=')
                    if key == 'avg_pax':
                        avg_pax[code_o] = value
                
            doc = f"p={avg_price} \n v ={value_list}\n r={ratio_list}\n"
            doc += f"v_0={value_0_list}\n r_0={ratio_0_list}\n"
            doc += f"d={avg_pax}\n"
            doc += f"c_f = {eco_flex_capacity}\n"
            doc += f"C = 187 \n"


            return doc

        def CA_Agent(query):

            loader = CSVLoader(file_path="RAG_Example_SBLP_CA.csv", encoding="utf-8")
            data = loader.load()
            documents = data
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectors = FAISS.from_documents(documents, embeddings)
            retriever = vectors.as_retriever(search_kwargs={'k': 1})
            similar_results = retrieve_similar_docs(query,retriever)
            problem_description = similar_results[0]['content'].replace("prompt:", "").strip()  
            example_matches = retrieve_key_information(problem_description)
            example_data_description = csv_qa_tool_CA(example_matches)
            example_data_description = example_data_description.replace('{', '{{')
            example_data_description = example_data_description.replace('}', '}}')    
            fewshot_example = f'''
            Question: {problem_description}

            Thought: I need to retrive the required OD and Departure Time information.

            Action: CName

            Action Input: {problem_description}

            Observation: {example_matches}

            Thought: I need to retrieve relevant information from 'information1.csv' for the given OD and Departure Time values. Next I need to retrieve the relevant coefficients from v1 and v2 based on the retrieved ticket information. 
            
            Action: CSVQA

            Action Input:  {example_matches}

            Observation: {example_data_description}

            Thought: I now have all the required information from the CSV, including p,d,v,r_0,r,v_0,v,c_f,C and so on for the specified flights. I will now formulate the SBLP model in markdown format, following the example provided, including all constraints, retrieved information, and generated code.


            Final Answer:
### Objective Function

$$
\max \quad \sum_(l,k,j) p[(l,k,j)] \cdot x[(l,k,j)]
$$

### Constraints

#### 1. Capacity Constraints

For each flight (OD pair $l$, departure time $k$):

$$
c_f \cdot x[(l,k,f)] + x[(l,k,l)] \leq C
$$

#### 2. Balance Constraints

For each OD pair $l$:

$$
r_0[l] \cdot x_o[l] + \sum_k \sum_j r[(l,k,j)] \cdot x[(l,k,j)] = d[l]
$$

#### 3. Scale Constraints

For each ticket option \((l,k,j)\):

$$
x[(l,k,j)] \cdot v_0[l]  - x_o[l] \cdot v[(l,k,j)] \leq 0
$$

#### 4. Nonnegativity Constraints

$$
x[(l,k,j)] \geq 0, \quad x_o[l] \geq 0
$$
- x = p.keys()
- x_o = d.keys()

### Retrieved Information
{example_data_description}

### Generated Code

```python
import gurobipy as gp
from gurobipy import GRB
{example_data_description}
model = gp.Model("sales_based_lp")
x = model.addVars(avg_price.keys(), lb=0, name="x") 
x_o = model.addVars(value_0_list.keys(), lb=0, name="x_o")  

model.setObjective(gp.quicksum(avg_price[key] * x[key] for key in avg_price.keys()), GRB.MAXIMIZE)

paired_keys = []
for i in range(0, len(avg_price.keys()), 2):  
    if i + 1 < len(avg_price.keys()):  
        names = list(avg_price.keys())
        model.addConstr(
            capacity_consum* x[names[i]] + x[names[i + 1]] <= 187,
            name=f"capacity_constraint"
        )

for l in ratio_0_list.keys():
    temp = 0
    for key in ratio_list.keys():
        if l in key:
            temp += ratio_list[key] * x[key]
    model.addConstr(
        float(ratio_0_list[l]) * x_o[l] + temp == float(avg_pax[l])
    )

for i in value_list.keys():
    for l in ratio_0_list.keys():
        if l in i:
            model.addConstr(
                value_0_list[l]  * x[i] <= value_list[i] * x_o[l]
            )

model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for v in model.getVars():
        print(v.varName, v.x)
    print("Optimal objective value:", model.objVal)
else:
    print("No optimal solution found.")
    '''

            tools = [Tool(name="CSVQA", func=csv_qa_tool_CA, description="Retrieve flight data."),Tool(name="CName", func=retrieve_key_information, description="Retrieve flight information.")]

            llm = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=api_key)
            prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

            Please refer to the following example and generate the answer in the same format:

            {fewshot_example}

            Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

            When you need to retrieve information from the CSV file, and write SBLP formulation by using the provided tools.

            """

            suffix = """

                Begin!

                User Description: {input}
                {agent_scratchpad}"""


            agent2 = initialize_agent(
            tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={
                "prefix": prefix,
                "suffix": suffix,
            },
            verbose=True,
            handle_parsing_errors=True
            )

            return agent2

        def get_answer(query):
            agent2 = CA_Agent(query)
            result = agent2.invoke({"input": query})
            return result
        def ProcessCA(query):
            result = get_answer(query)
            return result
        if "flow conservation constraints" in query:
            # print("Recommend Optimal Flights With Flow Conervation Constraints")
            result = ProcessPolicyFlow(query)
            output = convert_to_typora_markdown(result)
            Type = "Policy_Flow"
        else:
            # print("Only Develop Mathematic Formulations. No Recommendation for Flights.")
            result = ProcessCA(query)
            # output = result['output']

            output = convert_to_typora_markdown(result['output'])
            # code = "This SBLP Problem Type is to develop mathematical model. Therefore, no code in this part."
            Type = "CA"
        # ai_response = f'It is a {Type} SBLP Problem,\n model: \n {result},\n code for this model: \n{code}'
    else:
        return "Automatic modeling for this problem type is not supported yet."
    
    return output

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Timer(1.0, open_browser).start()  # Delay 1 second to open browser
    app.run(debug=True) 