def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyPDFLoader

def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

# file_path = "A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficien.pdf"

# doc = document_loader(file_path)
# print(type(doc))
# print(f"No. of pages in the pdf = {len(doc)}") # 11
# print(type(doc[0]))
# print("First 1000 characers = \n", doc[0].page_content[:1000])