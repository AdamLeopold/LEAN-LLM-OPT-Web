# from flask import Flask, render_template, request, jsonify
import tempfile
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage
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
import numpy as np
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
        llm1 = ChatOpenAI(
            temperature=0.0, model_name="gpt-4", openai_api_key=api_key
        )

        # Load and process the data
        loader = CSVLoader(file_path="Large_Scale_Or_Files/RefData.csv", encoding="utf-8")
        data = loader.load()

        # Each line is a document
        documents = data

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_documents(documents, embeddings)

        # Create a retriever
        retriever = vectors.as_retriever(search_kwargs={'k': 5})

        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm1,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        qa_tool = Tool(
            name="FileQA",
            func=qa_chain.invoke,
            description=(
                "Use this tool to answer questions about the problem type of the text. "
            ),
        )

        # Define few-shot examples as a string
        few_shot_examples_csv = """

        Query: What is the problem type in operation of the text? Please give the answer directly. Text:There are three best-selling items (P1, P2, P3) on Amazon with the profit w_1,w_2,w_3.There is an independent demand stream for each of the products. The objective of the company is to decide which demands to be fufilled over a ﬁnite sales horizon [0,10] to maximize the total expected revenue from ﬁxed initial inventories. The on-hand inventories for the three items are c_1,c_2,c_3 respectively. During the sales horizon, replenishment is not allowed and there is no any in-transit inventories. Customers who want to purchase P1,P2,P3 arrive at each period accoring to a Poisson process with a_1,a_2,a_3 the arrival rates respectively. Decision variables y_1,y_2,y_3 correspond to the number of requests that the firm plans to fulfill for product 1,2,3. These variables are all positive integers.

        Thought: I need to determine the problem type of the text. The Query contains descriptions like '.csv' or 'column'. I'll use the FileQA tool to retrieve the relevant information.

        Action: FileQA

        Action Input: "What is the problem type in operation of the text? text:There are three best-selling items (P1, P2, P3) on Amazon with the profit w_1, w_2, w_3. ..."

        Observation: The problem type of the text is Network Revenue Management.

        Thought: The problem type Network Revenue Management is in the allowed list [Network Revenue Management, Resource Allocation, Transportation, Facility Location Problem, Assignment Problem]. I could get the final answer and finish.

        Final Answer: Network Revenue Management.

        ---
        Query: What is the problem type in operation of the text? Please give the answer directly. Text:A supermarket needs to allocate various products, including high-demand items like the Sony Alpha Refrigerator, Sony Bravia XR, and Sony PlayStation 5, across different retail shelves. The product values and space requirements are provided in the "Products.csv" dataset. Additionally, the store has multiple shelves, each with a total space limit and specific space constraints for Sony and Apple products, as outlined in the "Capacity.csv" file. The goal is to determine the optimal number of units of each Sony product to place on each shelf to maximize total value while ensuring that the space used by Sony products on each shelf does not exceed the brand-specific limits. The decision variables x_ij represent the number of units of product i to be placed on shelf j.

        Thought: I need to determine the problem type of the text. The Query contains descriptions like '.csv' or 'column'. I'll use the FileQA tool to retrieve the relevant information.

        Action: FileQA

        Action Input: "What is the problem type in operation of the text? Text:A supermarket needs to allocate various products, including high-demand items like the Sony Alpha Refrigerator, Sony Bravia XR, ...."

        Observation: The problem type of the text is Inventory Management.

        Thought: The problem type Inventory Management is not in the allowed list [Network Revenue Management, Resource Allocation, Transportation, Facility Location Problem, Assignment Problem]. I need to review the query again and classify it to a type in the allowed list. According to the text, the problem type should be Resource Allocation. 

        Final Answer: Resource Allocation

        """

        few_shot_examples_without_csv = """
        Query: A book distributor needs to shuffle a bunch of books from two warehouses (supply points: W1, W2) to libraries (demand points: L1, L2), using a pair of sorting centers (transshipment points: C1, C2). W1 has a stash of up to p_1 books per day it can send out. W2 can send out up to p_2 books daily. Library L1 needs a solid d_1 books daily. L2 requires d_2 books daily. Storage at the sorting centers has no cap. Transportation costs: From W1 to C1 is t_11 dollars, to C2 is t_12 dollars. From W2 to C1 is t_21 dollars, and to C2 it__ t_22 dollars. From the centers to the libraries: From C1 to L1, it__l cost t_31 dollars, to L2 it__ t_32 dollars. From C2 to L1, it__ t_41 dollars, to L2 it__ t_42 dollars. The strategy here is all about minimizing transportation spend while making sure those libraries get their books on time. We__l use x_11 and x_12 to track shipments from W1 to C1 and C2, and x_21 and x_22 for shipments from W2. For the books going out to the libraries, y_11 and y_12 will handle the flow from C1 to L1 and L2, and y_21 and y_22 from C2. Variables are all positive integers.

        Thought: I need to determine the problem type of the text. The Query doesn't contain any descriptions like '.csv' and 'column'. I'll direcrly classify the problem type as 'Others without CSV'.

        Final Answer: Others without CSV

        """
        prefix = f"""I am a helpful assistant that can answer Querys about operation problems. My response must align with one of the following categories: Network Revenue Management, Resource Allocation, Transportation, Facility Location Problem, SBLP, Others with CSV, and Others without CSV. Firstly you need to identify whether the text contains any descriptions like '.csv' and 'column'.

        Always remember! If the input does not contain any description like '.csv' and 'column', and the values for all the variables are given directly, I will directly classify the problem type as 'Others without CSV'. Like the example {few_shot_examples_without_csv}. 

        However, if the text contains descriptions like '.csv' or 'column', and the values for all the variables are not given directly, I will use the following examples {few_shot_examples_csv} as a guide. And answer the Query by given the answer directly.

        """

        suffix = """

        Begin!

        Query: {input}
        {agent_scratchpad}"""

        classification_agent = initialize_agent(
            tools=[qa_tool],
            llm=llm1,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={
                "prefix": prefix,
                "suffix": suffix,
            },
            verbose=True,
            handle_parsing_errors=True,  # Enable error handling
        )
        openai.api_request_timeout = 60  
        # Get problem type
        category_result = classification_agent.invoke(f"What is the problem type in operation of the text? text:{query}")
        # TODO: Problem Type
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
    problem_type = data.get('problem_type')
    if not all([query, api_key, problem_type]):
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        result = process_problem_type(query, api_key, problem_type)
        # result = result.replace('\\', '\\\\')
        # result = result.replace('\\\\', '\\\\\\\\')
    # rendered_solution = render_markdown_with_highlight(result)
        rendered_solution = markdown.markdown(result, extensions=["fenced_code", CodeHiliteExtension()])

        rendered_solution = result
        return jsonify({
            'solution': rendered_solution,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
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

def convert_to_typora_markdown(content):
    content = content.replace(r'\[', '$$').replace(r'\]', '$$') 
    content = content.replace(r'\( ', '$').replace(r' \)', '$') 
    content = content.replace(r'\(', '$').replace(r'\)', '$')
    content = content.replace(r'\text{Maximize}', '\\max').replace(r'\text{Minimize}', '\\min') 
    content = content.replace(r'\t ', '\\t').replace(r' \f', '\\f') 
    content = content.replace(r'\{ ', '\\{').replace(r' \}', '\\}') 
    content = content.replace(r'\text{', r'\mathrm{')  # 将 \text{ 替换为 \mathrm{
    content = content.replace('\text', '\\mathrm')  # 确保 \text 变为 \mathrm
    content = content.replace('\t', '\\t')  # 保留原来的制表符替换

    return content

def extract_problem_type(output_text):
    pattern = r'(Network Revenue Management|Network Revenue Management Problem|Resource Allocation|Resource Allocation Problem|Transportation|Transportation Problem|Facility Location Problem|Assignment Problem|AP|Uncapacited Facility Location Problem|NRM|RA|TP|FLP|UFLP|Others without CSV|Sales-Based Linear Programming|SBLP)'
    match = re.search(pattern, output_text, re.IGNORECASE)
    return match.group(0) if match else None

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Tuple
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def retrieve_similar_docs(query,retriever):
    similar_docs = retriever.get_relevant_documents(query)
    results = []
    for doc in similar_docs:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    return results


def process_dataset_address(dataset_address: str) -> List[Document]:

    documents = []
    file_addresses = dataset_address.strip().split('\n')  
    for file_idx, file_address in enumerate(file_addresses, start=1):
        try:
            df = pd.read_csv(file_address.strip())  
            file_name = file_address.strip().split('/')[-1]  
            for row_idx, row in df.iterrows():
                page_content = ", ".join([f"{col} = {row[col]}" for col in df.columns])
                documents.append(Document(page_content=page_content))
            # print("-"*50+'documents'+"-"*50)
            # print(documents)
            # print('-'*110)
                
        except Exception as e:
            print(f"Error processing file {file_address}: {e}")
            continue
    
    return documents


def get_NRM_response(query,api_key,uploaded_files):

    retrieve='product'
    loader = CSVLoader(file_path="Large_Scale_Or_Files/RAG_Example_NRM2_MD.csv", encoding="utf-8")
    data = loader.load()
    documents = data
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_documents(documents, embeddings)
    retriever = vectors.as_retriever(search_kwargs={'k': 1})
    few_shot_examples = []

    similar_results = retrieve_similar_docs(query,retriever)

    for i, result in enumerate(similar_results, 1):
        content = result['content']
        split_at_formulation = content.split("Data_address:", 1)
        problem_description = split_at_formulation[0].replace("prompt:", "").strip()  
        split_at_address = split_at_formulation[1].split("Label:", 1)
        data_address = split_at_address[0].strip()

        split_at_label = split_at_address[1].split("Related:", 1)
        label = split_at_label[0].strip()  
        Related = split_at_label[1].strip()
        information = pd.read_csv(data_address)
        information_head = information[:36]

        example_data_description = "\nHere is the product data:\n"
        for i, r in information_head.iterrows():
            example_data_description += f"Product {i + 1}: {r['Product Name']}, revenue w_{i + 1} = {r['Revenue']}, demand rate a_{i + 1} = {r['Demand']}, initial inventory c_{i + 1} = {r['Initial Inventory']}\n"


        label = label.replace("{", "{{").replace("}", "}}")
        few_shot_examples.append(f"""

Question: Based on the following problem description and data, please formulate a complete mathematical model using real data from retrieval. {problem_description}

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. If the data to be retrieved is not specified, retrieve the whole dataset instead. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula. The final expressions should not be simplified or abbreviated.

Action: CSVQA

Action Input: Retrieve all the {retrieve} data related to {Related} to formulate the mathematical model.

Observation:

Thought: Now that I have the necessary data, construct the objective function and constraints using the retrieved data as parameters of the formula. Ensure to include any additional detailed constraints present in the problem description. Do NOT include any explanations, notes, or extra text. Format the expressions strictly in markdown, following this example. Besides, I need to use the $$ or $ to wrap the mathematical expressions instead of \[, \], \( or \). I also should avoid using align, align* and other latex environments. 
Final Answer: 
{label}
""")
    
    data = []
    for df_index, (file_name, df) in enumerate(uploaded_files):
        data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")
        for i, r in df.iterrows():
            description = ""
            description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
            data.append(description + "\n")
    document = [content for content in data]
    # document=data

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_texts(document, embeddings)
    retriever = vectors.as_retriever(search_kwargs={'k': 300})
    llm2 = ChatOpenAI(temperature=0.0, model_name='gpt-4.1', openai_api_key=api_key)

    system_prompt = (
        "Retrieve the documents in order. Use the given context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, retrieve all the data instead. Do not return source documents. Only present final answer."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm2, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    def qa_wrapper(query: str):
        return qa_chain.invoke({"input": query})
    qa_tool = Tool(
        name="CSVQA",
        func=qa_wrapper,
        description="Use this tool to answer Querys based on the provided CSV data and retrieve product data similar to the input query."
    )

    prefix = f"""You are an assistant that generates a mathematical models based on the user's description and provided CSV data.

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

    result = agent2.invoke(query)

    return result


def get_RA_response(query,api_key,uploaded_files):
    retrieve="product"
    loader = CSVLoader(file_path="Large_Scale_Or_Files/RAG_Example_RA2_MD.csv", encoding="utf-8")
    data = loader.load()

    # Each line is a document
    documents = data

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_documents(documents, embeddings)

    # Create a retriever
    retriever = vectors.as_retriever(search_kwargs={'k': 3})
    few_shot_examples = []
    similar_results =  retrieve_similar_docs(query,retriever)
    for i, result in enumerate(similar_results, 1):
        content = result['content']

        split_at_formulation = content.split("Data_address:", 1)
        problem_description = split_at_formulation[0].replace("prompt:", "").strip() 

        split_at_address = split_at_formulation[1].split("Label:", 1)
        data_address = split_at_address[0].strip()

        split_at_label = split_at_address[1].split("Related:", 1)
        label = split_at_label[0].strip()  
        Related = split_at_label[1].strip()

        datas=data_address.split()
        information = []

        for data in datas:
            information.append(pd.read_csv(data))
        example_data_description = "\nHere is the data:\n"
        for df_index, df in enumerate(information):
            if df_index == 0:
                example_data_description += f"\nDataFrame {df_index + 1} - Capacity\n"
            elif df_index == 1:
                example_data_description += f"\nDataFrame {df_index + 1} - Products\n"

            for z, r in df.iterrows():
                description = ""
                description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                example_data_description += description + "\n"
        label = label.replace("{", "{{").replace("}", "}}")
        few_shot_examples.append( f"""
Question: Based on the following problem description and data, please formulate a complete mathematical model using real data from retrieval. {problem_description}

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. If the data to be retrieved is not specified, retrieve the whole dataset instead. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula. The final expressions should not be simplified or abbreviated.

Action: CSVQA

Action Input: Retrieve all the {retrieve} data related to {Related} to formulate the mathematical model with no simplification or abbreviation.

Observation: {example_data_description}

Thought: Now that I have the necessary data, I would construct the objective function and constraints using the retrieved data as parameters of the formula. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula according to the retrieved 'product id'. Do NOT include any explanations, notes, or extra text. Format the expressions strictly in markdown, following this example. Besides, I need to use the $$ or $ to wrap the mathematical expressions instead of \[, \], \( or \). I also should avoid using align, align* and other latex environments. Besides, I should also avoid using \begin, \end, \text.

Final Answer: 
{label}
            """)

    data = []
    for df_index, (file_name, df) in enumerate(uploaded_files):
        data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")
        # for i, r in df.iterrows():
        #     description = ""
        #     description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
        #     data.append(description + "\n")
        if file_name=='products.csv' or file_name=='Products.csv':
            for i, r in df.iterrows():
                description = f"Product id: {i+1}; "
                description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                data.append(description + "\n")
        else:
            for i, r in df.iterrows():
                description = f""
                description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                data.append(description + "\n")
    documents = [content for content in data]

#    documents = process_dataset_address(dataset_address)
#    print(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_texts(documents, embeddings)
    retriever = vectors.as_retriever(search_kwargs={'k': 220})
    llm2 = ChatOpenAI(temperature=0.0, model_name='gpt-4.1', top_p=1,n = 1, openai_api_key=api_key)


    system_prompt = (
        "Retrieve the documents in order from top to bottom. Use the retrieved context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, retrieve all the data instead."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm2, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    def qa_wrapper(query: str):
        return qa_chain.invoke({"input": query})
    qa_tool = Tool(
        name="CSVQA",
        func=qa_wrapper,
        description="Use this tool to answer Querys based on the provided CSV data and retrieve product data similar to the input query."
    )

    prefix = f"""You are an assistant that generates a mathematical models based on the user's description and provided CSV data.

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

    result = agent2.invoke(query)

    return result


def get_TP_response(query,api_key,uploaded_files):
    retrieve="capacity data and products data, "
    # Load and process the data
    loader = CSVLoader(file_path="Large_Scale_Or_Files/RAG_Example_TP2_MD.csv", encoding="utf-8")
    data = loader.load()
    documents = data
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_documents(documents, embeddings)
    retriever = vectors.as_retriever(search_kwargs={'k': 3})
    few_shot_examples = []
    similar_results = retrieve_similar_docs(query,retriever)
    for i, result in enumerate(similar_results, 1):
        content = result['content']
        split_at_formulation = content.split("Data_address:", 1)
        problem_description = split_at_formulation[0].replace("prompt:", "").strip()

        split_at_address = split_at_formulation[1].split("Label:", 1)
        data_address = split_at_address[0].strip()

        split_at_label = split_at_address[1].split("Related:", 1)
        label = split_at_label[0].strip()  
        Related = split_at_label[1].strip()

        datas=data_address.split()
        information = []

        for data in datas:
            information.append(pd.read_csv(data))
        example_data_description = "\nHere is the data:\n"
        for df_index, df in enumerate(information):
            if df_index == 0:
                example_data_description += f"\nDataFrame {df_index + 1} - Customer Demand\n"
            elif df_index == 1:
                example_data_description += f"\nDataFrame {df_index + 1} - Supply Capacity\n"
            elif df_index == 2:
                example_data_description += f"\nDataFrame {df_index + 1} - Transportation Cost\n"

            for z, r in df.iterrows():
                description = ""
                description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                example_data_description += description + "\n"
            retrieve += ', '.join(df.columns)+', '
        label = label.replace("{", "{{").replace("}", "}}")
        few_shot_examples.append( f"""
Question: Based on the following problem description and data, please formulate a complete mathematical model using real data from retrieval. {problem_description}

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. Pay attention: 1. If the data to be retrieved is not specified, retrieve the whole dataset instead. 2. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula. 3. The final expressions should not be simplified or abbreviated.

Action: CSVQA

Action Input:  Retrieve all the {retrieve} data {Related} to formulate the mathematical model with no simplification or abbreviation. Retrieve the documents in order. Use the given context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, make sure that all the data is retrieved. Only present final answer in details of row, instead of giving a sheet format.

Observation: {example_data_description}

Thought: Now that I have the necessary data, I would construct the objective function and constraints using the retrieved data as parameters of the formula. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula. Do NOT include any explanations, notes, or extra text. Respond ONLY in this exact format: {label}. Following this example. The expressions should not be simplified or abbreviated. Besides, I need to use the $$ or $ to wrap the mathematical expressions instead of \[, \], \( or \). I also should avoid using align, align* and other latex environments. Besides, I should also avoid using \begin, \end, \text.

Final Answer: 
{label}
                """)
        
    data = []
    for df_index, (file_name, df) in enumerate(uploaded_files):
        data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")
        for i, r in df.iterrows():
            description = ""
            description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
            data.append(description + "\n")
    documents = [content for content in data]
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_texts(documents, embeddings)

    retriever = vectors.as_retriever(search_kwargs={'k': 300})

    llm2 = ChatOpenAI(temperature=0.0, model_name='gpt-4.1', top_p=1,n = 1, openai_api_key=api_key)


    system_prompt = (
        "Retrieve the documents in order. Use the given context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, make sure that all the data is retrieved."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm2, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    def qa_wrapper(query: str):
        return qa_chain.invoke({"input": query})['answer']
    qa_tool = Tool(
        name="CSVQA",
        func=qa_wrapper,
        description="Use this tool to answer Querys based on the provided CSV data and retrieve product data similar to the input query."
    )

    prefix = f"""You are an assistant that generates a mathematical models based on the user's description and provided CSV data.

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

    result = agent2.invoke(query)

    return result

def get_FLP_response(query,api_key,uploaded_files):
    retrieve='supplier'
    loader = CSVLoader(file_path="Large_Scale_Or_Files/RAG_Example_FLP2_MD.csv", encoding="utf-8")
    data = loader.load()

    documents = data

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_documents(documents, embeddings)
    retriever = vectors.as_retriever(max_tokens_limit=400,search_kwargs={'k': 1})
    few_shot_examples = []
    similar_results =  retrieve_similar_docs(query,retriever)
    for i, result in enumerate(similar_results, 1):
        content = result['content']
        split_at_formulation = content.split("Data_address:", 1)
        problem_description = split_at_formulation[0].replace("prompt:", "").strip() 

        split_at_address = split_at_formulation[1].split("Label:", 1)
        data_address = split_at_address[0].strip()

        file_addresses = data_address.strip().split('\n')
        dfs = []
        df_index = 0
        example_data_description = " "
        for file_address in file_addresses:
            try:
                df = pd.read_csv(file_address) 
                file_name = file_address.split('/')[-1]  
                if df_index == 0:
                    result = df['demand'].values.tolist()
                    example_data_description += "d=" + str(result) + "\n"
                elif df_index == 1:
                    result = df['fixed_costs'].values.tolist()
                    example_data_description +="c=" + str(result) + "\n"
                elif df_index == 2:
                    matrix = df.iloc[:,1:].values
                    example_data_description +="A=" + np.array_str(matrix)+ "."
                df_index += 1
                dfs.append((file_name, df))
            except Exception as e:
                print(f"Error reading file {file_address}: {e}")
        split_at_label = split_at_address[1].split("Related:", 1)
        label = split_at_label[0].strip() 
        label = label.replace("{", "{{").replace("}", "}}")
        Related=''

        few_shot_examples.append( f"""
Question: Based on the following problem description and data, please formulate a complete mathematical model using real data from retrieval. {problem_description}

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. If the data to be retrieved is not specified, retrieve the whole dataset instead. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula. The final expressions should not be simplified or abbreviated.

Action: CSVQA

Action Input: Retrieve all the {retrieve} data {Related} to formulate the mathematical model with no simplification or abbreviation. Retrieve the documents in order. Use the given context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, make sure that all the data is retrieved. Only present final answer in details of row, instead of giving a sheet format.

Observation: {example_data_description}

Thought: Now that I have the necessary data, I would construct the objective function and constraints using the retrieved data as parameters of the formula. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula. Do NOT include any explanations, notes, or extra text. Respond ONLY in this exact format: {label}. Following this example. The expressions should not be simplified or abbreviated. Besides, I need to use the $$ or $ to wrap the mathematical expressions instead of \[, \], \( or \). I also should avoid using align, align* and other latex environments. Besides, I should also avoid using \begin, \end, \text.

Final Answer: 
{label}

               """)

    data = []
    data_description = ""
    for df_index, (file_name, df) in enumerate(uploaded_files):
        if 'demand' in file_name:
                result = df['demand'].values.tolist()
                data_description += "d=" + str(result) + "\n"
        elif 'fixed_cost' in file_name:
            result = df['fixed_costs'].values.tolist()
            data_description +="c=" + str(result) + "\n"
        elif 'transportation_cost' in file_name:
            matrix = df.iloc[:,1:].values
            data_description +="A=" + np.array_str(matrix)+ "\n"

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_texts([data_description], embeddings)

    retriever = vectors.as_retriever(max_tokens_limit=400, search_kwargs={'k': 1}) 
    llm2 = ChatOpenAI(temperature=0.0, model_name='gpt-4.1', top_p=1,n = 1,openai_api_key=api_key)
    
    system_prompt = (
        "Retrieve the documents in order. Use the given context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, make sure that all the data is retrieved."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm2, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    def qa_wrapper(query: str):
        return qa_chain.invoke({"input": query})
    qa_tool = Tool(
        name="CSVQA",
        func=qa_wrapper,
        description="Use this tool to answer Querys based on the provided CSV data and retrieve product data similar to the input query."
    )

    prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

            Please refer to the following example and generate the answer in the same format:

            {few_shot_examples}

            Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

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
        handle_parsing_errors=True
    )

    result = agent2.invoke(query)
    return result

def get_AP_response(query,api_key,uploaded_files):
    loader = CSVLoader(file_path="Large_Scale_Or_Files/RAG_Example_AP2_MD.csv", encoding="utf-8")
    data = loader.load()
    retrieve = ''
    documents = data

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_documents(documents, embeddings)
    retriever = vectors.as_retriever(max_tokens_limit=400,search_kwargs={'k': 1})
    few_shot_examples = []
    similar_results =  retrieve_similar_docs(query,retriever)
    for i, result in enumerate(similar_results, 1):
        content = result['content']
        split_at_formulation = content.split("Data_address:", 1)
        problem_description = split_at_formulation[0].replace("prompt:", "").strip() 

        split_at_address = split_at_formulation[1].split("Label:", 1)
        data_address = split_at_address[0].strip()

        file_addresses = data_address.strip().split('\n')
        dfs = []
        df_index = 0
        example_data_description = " "
        for file_address in file_addresses:
            try:
                df = pd.read_csv(file_address) 
                file_name = file_address.split('/')[-1]  
                matrix = df.iloc[:,1:].values
                example_data_description +="C=" + np.array_str(matrix)+ "."
                dfs.append((file_name, df))
            except Exception as e:
                print(f"Error reading file {file_address}: {e}")
        split_at_label = split_at_address[1].split("Related:", 1)
        label = split_at_label[0].strip() 
        Related = split_at_label[1].strip()
        label = label.replace("{", "{{").replace("}", "}}")
        few_shot_examples.append( f"""
Question: Based on the following problem description and data, please formulate a complete mathematical model using real data from retrieval. {problem_description}

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. If the data to be retrieved is not specified, retrieve the whole dataset instead. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula. The final expressions should not be simplified or abbreviated.

Action: CSVQA

Action Input: Retrieve all the {retrieve} data {Related} to formulate the mathematical model with no simplification or abbreviation. Retrieve the documents in order. Use the given context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, make sure that all the data is retrieved. Only present final answer in details of row, instead of giving a sheet format.

Observation:{example_data_description}

Thought: Now that I have the necessary data, I would construct the objective function and constraints using the retrieved data as parameters of the formula. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula according to the retrieved 'product id'. Do NOT include any explanations, notes, or extra text. Respond ONLY in this exact format: {label}. Following this example. The expressions should not be simplified or abbreviated. Besides, I need to use the $$ or $ to wrap the mathematical expressions instead of \[, \], \( or \). I also should avoid using align, align* and other latex environments. Besides, I should also avoid using \begin, \end, \text.
Final Answer: 
{label}

                """)
    

    data_description = " "
    data = []
    dfs = []
    for df_index, (file_name, df) in enumerate(uploaded_files):
        matrix = df.iloc[:,1:].values
        data_description +="C=" + np.array_str(matrix)+ "."
        dfs.append((file_name, df))
        df_index += 1
        dfs.append((file_name, df))
        # data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")
        # for i, r in df.iterrows():
        #     matrix = df.iloc[:,1:].values
        #     data_description +="C=" + np.array_str(matrix)+ "."
 
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_texts([data_description], embeddings)

    retriever = vectors.as_retriever(max_tokens_limit=400, search_kwargs={'k': 1}) 
    llm2 = ChatOpenAI(temperature=0.0, model_name='gpt-4.1',top_p=1,n = 1, openai_api_key=api_key)

    system_prompt = (
        "Retrieve the documents in order from top to bottom. Use the retrieved context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, retrieve all the data instead."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm2, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    def qa_wrapper(query: str):
        return qa_chain.invoke({"input": query})
    qa_tool = Tool(
        name="CSVQA",
        func=qa_wrapper,
        description="Use this tool to answer Querys based on the provided CSV data and retrieve product data similar to the input query."
    )


    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm2,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=False,
    # )

    # qa_tool = Tool(
    #     name="CSVQA",
    #     func=qa_chain.run,
    #     description="Use this tool to answer queries based on the provided CSV data and retrieve data similar to the input query."
    # )

    prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

            Please refer to the following example and generate the answer in the same format:

            {few_shot_examples}

            Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

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
        handle_parsing_errors=True
    )

    result = agent2.invoke(query)
    return result

def get_code(output,selected_problem,api_key):
    llm_code = ChatOpenAI(
        temperature=0.0, model_name="gpt-4.1", top_p=1,n = 1, openai_api_key=api_key
    )

    prompt = f"""
    You are an expert in mathematical optimization and Python programming. Your task is to write Python code to solve the provided mathematical optimization model using the Gurobi library. The code should include the definition of the objective function, constraints, and decision variables. Please don't add additional explanations. Please don't include ```python and ```.Below is the provided mathematical optimization model:

    Mathematical Optimization Model:
    {output}
    """

    if selected_problem == "Network Revenue Management" or selected_problem == "NRM" or selected_problem == "Network Revenue Management Problem":

        prompt += """
For example, here is a simple instance for reference:

Mathematical Optimization Model:

##### Objective Function: 

$\quad \quad \max \quad \sum_i A_i \cdot x_i$

##### Constraints

###### 1. Inventory Constraints: 

$\quad \quad x_i \leq I_i, \quad \forall i $

###### 2. Demand Constraints: 

$x_i \leq d_i, \quad \forall i $

###### 3. Variable Constraints: 

$x_i \in \mathbb Z, x_i \geq 0, \quad \forall i $

###### Retrieved Information
$\small I = [7550, 6244]$
$\small A =  [149, 389]$
$\small d =  [15057, 12474]$
$\small s = 100$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Product_Optimization")

# Decision variables for the number of units of each product
x_1 = m.addVar(vtype=GRB.INTEGER, name="x_1") # Number of units of product 1
x_2 = m.addVar(vtype=GRB.INTEGER, name="x_2") # Number of units of product 2

# Objective function: Maximize 149 x_1 + 389 x_2
m.setObjective(149 * x_1 + 389 * x_2, GRB.MAXIMIZE)

# Constraints
m.addConstr(x_1 <= 7550, name="inventory_constraint_1")
m.addConstr(x_2 <= 6244, name="inventory_constraint_2")
m.addConstr(x_1 <= 15057, name="demand_constraint_1")
m.addConstr(x_2 <= 12474, name="demand_constraint_2")

# Non-negativity constraints are implicitly handled by the integer constraints (x_1, x_2 >= 0)

# Solve the model
m.optimize()

        """

    elif selected_problem == "Facility Location Problem" or selected_problem == "FLP" or selected_problem == "Facility Location":
        prompt += """
For example, here is a simple instance for reference:

Mathematical Optimization Model:


Objective Function:
$\quad \quad \min \quad \sum_{i} \sum_{j} A_{ij} \cdot x_{ij} + \sum_{i} c_i \cdot y_i$

Constraints
1. Demand Constraint:
$\quad \quad \sum_i x_{ij} = d_j, \quad \forall j$
2. Capacity Constraint:
$\quad \quad \sum_j x_{ij} \leq M \cdot y_i, \quad \forall i$
3. Non-negativity:
$\quad \quad x_{ij} \geq 0, \quad \forall i,j$
4. Binary Requirement:
$\quad \quad y_i \in \{0,1\}, \quad \forall i$

Retrieved Information
$\small d = [1083, 776, 16214, 553, 17106, 594, 732]$
$\small c = [102.33, 94.92, 91.83, 98.71, 95.73, 99.96, 98.16]$
$\small A = \begin{bmatrix}
1506.22 & 70.90 & 8.44 & 260.27 & 197.47 & 71.71 & 61.19 \\  
1732.65 & 1780.72 & 567.44 & 448.68 & 29.00 & 1484.91 & 963.92 \\  
115.66 & 100.76 & 64.68 & 1324.53 & 64.99 & 134.88 & 2102.83 \\  
1254.78 & 1115.63 & 52.31 & 1036.16 & 892.63 & 1464.04 & 1383.41 \\  
42.90 & 891.01 & 1013.94 & 1128.72 & 58.91 & 42.89 & 1570.31 \\  
0.70 & 139.46 & 70.03 & 79.15 & 1482.00 & 0.91 & 110.46 \\  
1732.30 & 1780.44 & 486.50 & 523.74 & 522.08 & 82.48 & 826.41
\end{bmatrix}$
$\small M = \sum_j d_j = 1083 + 776 + 16214 + 553 + 17106 + 594 + 732 = 38058 $


The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
d = np.array([1083, 776, 16214, 553, 17106, 594, 732])
c = np.array([102.33, 94.92, 91.83, 98.71, 95.73, 99.96, 98.16])
A = np.array([[1506.22, 70.90, 8.44, 260.27, 197.47, 71.71, 61.19],  
[1732.65, 1780.72, 567.44, 448.68, 29.00, 1484.91, 963.92],  
[115.66, 100.76, 64.68, 1324.53, 64.99, 134.88, 2102.83],  
[1254.78, 1115.63, 52.31, 1036.16, 892.63, 1464.04, 1383.41],  
[42.90, 891.01, 1013.94, 1128.72, 58.91, 42.89, 1570.31],  
[0.70, 139.46, 70.03, 79.15, 1482.00, 0.91, 110.46],  
[1732.30, 1780.44, 486.50, 523.74, 522.08, 82.48, 826.41]])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(A.shape[0], A.shape[1], lb=0, name="x")
y = m.addVars(A.shape[0], vtype=GRB.BINARY, name="y")

# Objective function
m.setObjective(gp.quicksum(A[i, j]*x[i, j] for i in range(A.shape[0]) for j in range(A.shape[1])) + gp.quicksum(c[i]*y[i] for i in range(A.shape[0])), GRB.MINIMIZE)

# Constraints
for j in range(A.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(A.shape[0])) == d[j], name=f"demand_constraint_{j}")

M = 1000000  # large number
for i in range(A.shape[0]):
    m.addConstr(-M*y[i] + gp.quicksum(x[i, j] for j in range(A.shape[1])) <= 0, name=f"M_constraint_{i}")

# Solve the model
m.optimize()
        """

    elif selected_problem == "Assignment Problem" or selected_problem == "AP" or selected_problem == "Assignment":
        prompt += """
For example, here is a simple instance for reference:

Mathematical Optimization Model:


Objective Function:
$\quad \quad \min \quad \sum_{i=1}^3 \sum_{j=1}^3 c_{ij} \cdot x_{ij}$

Constraints
1. Row Assignment Constraint:
$\quad \quad \sum_{j=1}^3 x_{ij} = 1, \quad \forall i \in \{1,2,3\}$
2. Column Assignment Constraint:
$\quad \quad \sum_{i=1}^3 x_{ij} = 1, \quad \forall j \in \{1,2,3\}$
3. Binary Constraint:
$\quad \quad x_{ij} \in \{0,1\}, \quad \forall i,j$

Retrieved Information
$\small c = \begin{bmatrix}
3000 & 3200 & 3100 \\
2800 & 3300 & 2900 \\
2900 & 3100 & 3000 
\end{bmatrix}$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
c = np.array([
    [3000, 3200, 3100],
    [2800, 3300, 2900],
    [2900, 3100, 3000]
])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(c.shape[0], c.shape[1], vtype=GRB.BINARY, name="x")

# Objective function
m.setObjective(gp.quicksum(c[i, j]*x[i, j] for i in range(c.shape[0]) for j in range(c.shape[1])), GRB.MINIMIZE)

# Constraints
for i in range(c.shape[0]):
    m.addConstr(gp.quicksum(x[i, j] for j in range(c.shape[1])) == 1, name=f"row_constraint_{i}")

for j in range(c.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(c.shape[0])) == 1, name=f"col_constraint_{j}")

# Solve the model
m.optimize()
"""

    
    elif selected_problem == "Transportation Problem" or selected_problem == "TP" or selected_problem == "Transportation":
        prompt += """
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\quad \quad \min \quad \sum_i \sum_j c_{ij} \cdot x_{ij}$

Constraints
1. Demand Constraint:
$\quad \quad \sum_i x_{ij} \geq d_j, \quad \forall j$
2. Capacity Constraint:
$\quad \quad \sum_j x_{ij} \leq s_i, \quad \forall i$

Retrieved Information
$\small d = [94, 39, 65, 435]$
$\small s = [2531, 20, 210, 241]$
$\small c = \begin{bmatrix}
883.91 & 0.04 & 0.03 & 44.45 \\
543.75 & 23.68 & 23.67 & 447.75 \\
537.34 & 23.76 & 498.95 & 440.60 \\
1791.49 & 68.21 & 1432.48 & 1527.76
\end{bmatrix}$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Optimization")

# Decision variables
x_S1_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C1")
x_S1_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C2")
x_S1_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C3")
x_S1_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C4")
x_S2_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C1")
x_S2_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C2")
x_S2_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C3")
x_S2_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C4")
x_S3_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C1")
x_S3_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C2")
x_S3_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C3")
x_S3_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C4")
x_S4_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C1")
x_S4_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C2")
x_S4_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C3")
x_S4_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C4")

# Objective function
m.setObjective(883.91 * x_S2_C1 + 0.04 * x_S2_C2 + 0.03 * x_S2_C3 + 44.45 * x_S2_C4 + 543.75 * x_S1_C1 + 23.68 * x_S1_C2 + 23.67 * x_S1_C3 + 447.75 * x_S1_C4 + 537.34 * x_S3_C1 + 23.76 * x_S3_C2 + 498.95 * x_S3_C3 + 440.60 * x_S3_C4 + 1791.49 * x_S4_C1 + 68.21 * x_S4_C2 + 1432.48 * x_S4_C3 + 1527.76 * x_S4_C4, GRB.MINIMIZE)

# Constraints
m.addConstr(x_S1_C1 + x_S2_C1 + x_S3_C1 + x_S4_C1 >= 94, name="demand_constraint1")
m.addConstr(x_S1_C2 + x_S2_C2 + x_S3_C2 + x_S4_C2 >= 39, name="demand_constraint2")
m.addConstr(x_S1_C3 + x_S2_C3 + x_S3_C3 + x_S4_C3 >= 65, name="demand_constraint3")
m.addConstr(x_S1_C4 + x_S2_C4 + x_S3_C4 + x_S4_C4 >= 435, name="demand_constraint4")
m.addConstr(x_S1_C1 + x_S1_C2 + x_S1_C3 + x_S1_C4 <= 2531, name="capacity_constraint1")
m.addConstr(x_S2_C1 + x_S2_C2 + x_S2_C3 + x_S2_C4 <= 20, name="capacity_constraint2")
m.addConstr(x_S3_C1 + x_S3_C2 + x_S3_C3 + x_S3_C4 <= 210, name="capacity_constraint3")
m.addConstr(x_S4_C1 + x_S4_C2 + x_S4_C3 + x_S4_C4 <= 241, name="capacity_constraint4")

# Solve the model
m.optimize()
        """
    
    elif selected_problem == "Resource Allocation" or selected_problem == "RA" or selected_problem == "Resource Allocation Problem":
        prompt += """
For example, here is a simple instance for reference:

Always remember: If not specified. All the variables are non-negative interger.

Mathematical Optimization Model:

Objective Function:
$\quad \quad \max \quad \sum_i \sum_j p_i \cdot x_{ij}$

Constraints
1. Capacity Constraint:
$\quad \quad \sum_i a_i \cdot x_{ij} \leq c_j, \quad \forall j$
2. Non-negativity Constraint:
$\quad \quad x_{ij} \geq 0, \quad \forall i,j$

Retrieved Information
$\small p = [321, 309, 767, 300, 763, 318, 871, 522, 300, 275, 858, 593, 126, 460, 685, 443, 700, 522, 940, 598]$
$\small a = [495, 123, 165, 483, 472, 258, 425, 368, 105, 305, 482, 387, 469, 341, 318, 104, 377, 213, 56, 131]$
$\small c = [4466]$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(20, vtype=GRB.INTEGER, name="x")

# Objective function
m.setObjective(sum(x[i]*c[i] for i in range(20)), GRB.MAXIMIZE)

# Constraints
m.addConstr(sum(x[i]*w[i] for i in range(20)) <= 4466, name="capacity_constraint")

# Coefficients for the objective function
c = [321, 309, 767, 300, 763, 318, 871, 522, 300, 275, 858, 593, 126, 460, 685, 443, 700, 522, 940, 598]

# Coefficients for the capacity constraint
w = [495, 123, 165, 483, 472, 258, 425, 368, 105, 305, 482, 387, 469, 341, 318, 104, 377, 213, 56, 131]

# Solve the model
m.optimize()
```

-----
Here is another simple instance for reference:

Objective Function:
$\quad \quad \max \quad \sum_i p_i \cdot x_i$

Constraints
1. Capacity Constraint:
$\quad \quad \sum_i a_i \cdot x_i \leq 180$
2. Dependency Constraint:
$\quad \quad x_1 \leq x_3$
3. Non-negativity Constraint:
$\quad \quad x_i \geq 0, \quad \forall i$

Retrieved Information
$\small p = [888, 134, 129, 370, 921, 765, 154, 837, 584, 365]$
$\small a = [4, 2, 4, 3, 2, 1, 2, 1, 3, 3]$

The corresponding Python code for this instance is as follows:

import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(10, vtype=GRB.INTEGER, name="x")

# Objective function
p = [888, 134, 129, 370, 921, 765, 154, 837, 584, 365]
m.setObjective(sum(x[i]*p[i] for i in range(10)), GRB.MAXIMIZE)

# Constraints
a = [4, 2, 4, 3, 2, 1, 2, 1, 3, 3]
m.addConstr(sum(x[i]*a[i] for i in range(10)) <= 180, name="capacity_constraint")
m.addConstr(x[0] <= x[2], name="dependency_constraint")

# Solve the model
m.optimize()
        
        """





        
    else:
        prompt += """
For example, here is a simple instance for reference:

Mathematical Optimization Model:
\begin{aligned}
\text{Maximize} \quad & Z = 5x_S + 8x_F \\
\text{Subject to} \quad & \\
& 2x_S + 5x_F \leq 200 \\
& x_S \leq 0.3(x_S + x_F) \\
& x_F \geq 10 \\
\text{Decision variables} \quad & x_S, x_F \in \mathbb{Z}^+ 
\end{aligned}

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Worker_Optimization")

# Decision variables for the number of seasonal (x_S) and full-time (x_F) workers
x_S = m.addVar(vtype=GRB.INTEGER, name="x_S")  # Number of seasonal workers
x_F = m.addVar(vtype=GRB.INTEGER, name="x_F")  # Number of full-time workers

# Objective function: Maximize Z = 5x_S + 8x_F
m.setObjective(5 * x_S + 8 * x_F, GRB.MAXIMIZE)

# Constraints
m.addConstr(2 * x_S + 5 * x_F <= 200, name="resource_constraint")
m.addConstr(x_S <= 0.3 * (x_S + x_F), name="seasonal_ratio_constraint")
m.addConstr(x_F >= 10, name="full_time_minimum_constraint")

# Non-negativity constraints are implicitly handled by the integer constraints (x_S, x_F >= 0)

# Solve the model
m.optimize()
```











The another example is:

Mathematical Optimization Model:
\begin{aligned}
\text{Minimize}\quad & \sum_{i=1}^3\sum_{j=1}^3 c_{ij}\,x_{ij} \\
\text{Subject to}\quad
& \sum_{j=1}^3 x_{ij} = 1,\quad i=1,2,3,\\
& \sum_{i=1}^3 x_{ij} = 1,\quad j=1,2,3,\\
& x_{ij}\in\{0,1\},\quad i,j=1,2,3,\\
\text{where}\quad & C = (c_{ij}) =
\begin{bmatrix}
919 & 556 & 951\\
21  & 640 & 409\\
59  & 786 & 304
\end{bmatrix}.
\end{aligned}


The corresponding Python code for this instance is as follows:

```python

import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
c = np.array([
    [919, 556, 951],
    [21, 640, 409],
    [59, 786, 304]
])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(c.shape[0], c.shape[1], vtype=GRB.BINARY, name="x")

# Objective function
m.setObjective(gp.quicksum(c[i, j]*x[i, j] for i in range(c.shape[0]) for j in range(c.shape[1])), GRB.MINIMIZE)

# Constraints
for i in range(c.shape[0]):
    m.addConstr(gp.quicksum(x[i, j] for j in range(c.shape[1])) == 1, name=f"row_constraint_{i}")

for j in range(c.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(c.shape[0])) == 1, name=f"col_constraint_{j}")

# Solve the model
m.optimize() 
```
"""


    messages = [
        HumanMessage(content=prompt) 
    ]

    response = llm_code(messages)

    return response.content

def process_problem_type(query, api_key, problem_type):
    # Read global uploaded files
    global uploaded_files

    if problem_type == "Network Revenue Management" or problem_type == "NRM" or problem_type == "Network Revenue Management Problem":
        result = get_NRM_response(query, api_key, uploaded_files)
        model = convert_to_typora_markdown(result['output'])
        code_response = get_code(output=result,selected_problem=problem_type,api_key=api_key)
        output = """### The Mathematical model is as follows:\n\n""" + model + """\n\n ### The Corresonding code is as follows: \n\n ```python \n\n """ + code_response + """\n\n ``` \n\n"""   
    elif problem_type == "Resource Allocation" or problem_type == "RA" or problem_type == "Resource Allocation Problem":
        result = get_RA_response(query, api_key, uploaded_files)
        output = result['output']
        model = convert_to_typora_markdown(result['output'])
        code_response = get_code(output=result,selected_problem=problem_type,api_key=api_key)
        output = """### The Mathematical model is as follows:\n\n""" + model + """\n\n ### The Corresonding code is as follows: \n\n ```python \n\n """ + code_response + """\n\n ``` \n\n"""
    elif problem_type == "Transportation" or problem_type == "TP" or problem_type == "Transportation Problem":
        result = get_TP_response(query, api_key, uploaded_files)
        output = result['output']
        model = convert_to_typora_markdown(result['output'])
        code_response = get_code(output=result,selected_problem=problem_type,api_key=api_key)
        output = """### The Mathematical model is as follows:\n\n""" + model + """\n\n ### The Corresonding code is as follows: \n\n ```python \n\n """ + code_response + """\n\n ``` \n\n"""

    elif problem_type == "Facility Location Problem" or problem_type == "FLP" or problem_type == "Uncapacited Facility Location" or problem_type == "UFLP":
        result = get_FLP_response(query, api_key, uploaded_files)
        output = result['output']
        model = convert_to_typora_markdown(result['output'])
        code_response = get_code(output=result,selected_problem=problem_type,api_key=api_key)
        output = """### The Mathematical model is as follows:\n\n""" + model + """\n\n ### The Corresonding code is as follows: \n\n ```python \n\n """ + code_response + """\n\n ``` \n\n"""

    elif problem_type == "Assignment Problem" or problem_type == "AP":
        result = get_AP_response(query, api_key, uploaded_files)
        output = result['output']
        model = convert_to_typora_markdown(result['output'])
        code_response = get_code(output=result,selected_problem=problem_type,api_key=api_key)
        output = """### The Mathematical model is as follows:\n\n""" + model + """\n\n ### The Corresonding code is as follows: \n\n ```python \n\n """ + code_response + """\n\n ``` \n\n"""

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
            loader = CSVLoader(file_path="Large_Scale_Or_Files/RAG_Example_SBLP_Flow.csv", encoding="utf-8")
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

            llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)
            prefix = f"""You are an assistant that generates a SBLP mathematical model and the corresponding Gurobi Python code based on the user's description and provided CSV data.  When you need to retrieve information from the CSV file, please use the CSVQA tool. When you need to retrieve flight information, please use the CName tool.

            Please refer to the following example and generate the answer in the same format:

            {fewshot_example}

            Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.


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
            # llm_code = ChatOpenAI(
            #     temperature=0.0, model_name="gpt-4.1", openai_api_key=api_key
            # )
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

            loader = CSVLoader(file_path="Large_Scale_Or_Files/RAG_Example_SBLP_CA.csv", encoding="utf-8")
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

            Thought: I need to extract all relevant information from the files 'flight.csv', 'od_demand.csv', 'v1.csv', and 'v2.csv' based on the required ticket attributes. It is essential that I do not overlook any useful information, and the retrieved parameters should remain unfolded.
            
            Action: CSVQA

            Action Input:  {example_matches}

            Observation: {example_data_description}

            Thought: I now have all the required information from the CSV, including p,d,v,r_0,r,v_0,v,c_f,C and so on for the specified flights. I will now formulate the SBLP model in markdown format, following the example provided, including all constraints, retrieved information, and generated code.Remember that all the retrieved parameters should remain unfolded in the final answer.


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

            llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)
            prefix = f"""You are an assistant that generates a SBLP mathematical model and the corresponding Gurobi Python code based on the user's description and provided CSV data.  When you need to retrieve information from the CSV file, please use the CSVQA tool. When you need to retrieve flight information, please use the CName tool.

            Please refer to the following example and generate the answer in the same format:

            {fewshot_example}

            Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

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
            result = ProcessPolicyFlow(query)
            output = convert_to_typora_markdown(result)
        else:
            result = ProcessCA(query)
            output = convert_to_typora_markdown(result['output'])
    else:
        return "Automatic modeling for this problem type is not supported yet."
    return output

# def open_browser():
#     webbrowser.open_new('http://127.0.0.1:5000/')

# if __name__ == '__main__':
#     threading.Timer(1.0, open_browser).start()  # Delay 1 second to open browser
#     app.run(host='0.0.0.0', port=5050,debug=True) 

# def open_browser():
#     webbrowser.open_new('http://127.0.0.1:5050/')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5050, debug=True)

def open_browser():
    webbrowser.open_new('https://xxxx.cpolar.cn')  
    webbrowser.open_new_tab('http://localhost:5050/')

@app.route('/')
def home():
    return "服务已启动！通过公网URL访问我吧！"

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        threading.Timer(1.0, open_browser).start()
    app.run(host='0.0.0.0', port=5050, debug=True)