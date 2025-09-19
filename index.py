# api/index.py
from streamlit.web.bootstrap import run
import sys
import os

def main():
    # Set the working directory to the root of the project
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    # Get the command line arguments for Streamlit
    sys.argv = ["streamlit", "run", "appv2.py", "--server.port", "8282", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
    # Run Streamlit
    run(__name__, [], [])
