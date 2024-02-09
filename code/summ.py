import boto3
import json
from langchain.llms.bedrock import Bedrock
from typing import Literal
from argparse import ArgumentParser
import os

GPT_3 = "gpt-3.5-turbo"
GPT_4 = "gpt-4-0125-preview"
CLAUDE2 = "anthropic.claude-v2"
CLAUDE1 = "anthropic.claude-instant-v1"
CLAUDE21 = "anthropic.claude-v2:1"
TITAN_LITE = "amazon.titan-text-lite-v1"
TITAN_EXPRESS = "amazon.titan-text-express-v1"

args = None


def temp():
    document = None

    # payload = {
    #     "prompt": "[INST]" + document + "[/INST]",
    #     "max_gen_len": 512,
    #     "temperature": 0.5,
    #     "top_p": 0.9,
    # }
    # body = json.dumps(payload)

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
    )

    docs = text_splitter.create_documents([document])

    num_docs = len(docs)

    num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

    print(
        f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens"
    )

    # Set verbose=True if you want to see the prompts being used
    from langchain.chains.summarize import load_summarize_chain

    # summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)
    summary_chain = load_summarize_chain(llm=llm, chain_type="stuff", verbose=False)

    # output = summary_chain.invoke(docs)
    output = summary_chain.invoke(document)
    output


def summarize():
    # modelId = "amazon.titan-tg1-large"
    # modelId = "anthropic.claude-v2"
    # modelId = "meta.llama2-70b-chat-v1"
    modelId = CLAUDE2

    bedrock = boto3.client(service_name="bedrock-runtime")

    llm = Bedrock(
        model_id=modelId,
        model_kwargs={
            # "max_tokens_to_sample": 10,
            # "stopSequences": [],
            "temperature": 0,
            # "top_k": 250
            # "top_p": 1,
        },
        client=bedrock,
    )
    text_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "text",
    )
    doc_path = os.path.join(text_dir, "document" + ".txt")
    with open(doc_path, "r") as file:
        document = file.read()

    token_num = llm.get_num_tokens(document)
    print("Invoking LLM...")
    print(f"{token_num} tokens")

    summ = llm.invoke(document)

    out_path = os.path.join(text_dir, args.out_filename + ".txt")
    with open(out_path, "w") as file:
        file.write(summ)


def parseArgument():
    parser = ArgumentParser()
    parser.add_argument("-O", "--outfile", type=str, default="_summary")

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgument()
    summarize()
