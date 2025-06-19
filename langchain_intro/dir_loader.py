from langchain_community.document_loaders import (
    DirectoryLoader,
    # TextLoader,
    # PyPDFLoader,
    # CSVLoader,
)

# ------- This section required to resovle ssl issue -------
# export SSL_CERT_FILE=$(python3 -m certifi)

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt", download_dir="/tmp/nltk_data")
# ------- This section required to resovle ssl issue -------


# create a dir-loader
dir_loader = DirectoryLoader("./docs/", glob="**/*.txt")

# load all file-names in the directory
dir_docs = dir_loader.load()

print(dir_docs)
