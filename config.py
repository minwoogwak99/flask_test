import os
from dotenv import load_dotenv


def load_config():

    load_dotenv()
    # LANG SMITH
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # SET API KEY
    # NEED TO PUT API KEY IN .env FILE (refer to README.md)
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

    print('config.py loaded')


load_config()
