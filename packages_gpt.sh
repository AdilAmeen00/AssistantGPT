#!/bin/bash

# Update the system
sudo apt-get update

# Install necessary packages
sudo apt-get install -y python3-venv python3-pip git-all sqlite3

# Clone the git repository
git clone https://github.com/AdilAmeen00/SupportGPT.git

# Navigate to the cloned directory (Assuming the requirements.txt is inside the cloned directory)
cd SupportGPT

# Install Python packages
pip install -r requirements.txt
pip3 install python-dotenv urllib3 chardet xformers chromadb pysqlite3-binary transformers accelerate scipy sentence_transformers flask --ignore-installed blinker==1.4
pip install --upgrade langchain

echo "All packages and dependencies have been installed!"
