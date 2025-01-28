from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from text_split import text_splitter

def watsonx_embeddingg():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

query = "How are you?"
chunk = text_splitter.split_text(query)
watsonx_embedding = watsonx_embeddingg()
query_result = watsonx_embedding.embed_query(query)
# print(len(query_result))
# print(query_result[:5])
