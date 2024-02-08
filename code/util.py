from pyalex import Works
from langchain.document_loaders import PyPDFLoader
import os


def extractPdfToFile(doi: str, filename: str = "document"):
    print("extracting pdf data...")

    # get pdf data from openalex
    paper_work_object = Works()[doi]
    url = paper_work_object["open_access"]["oa_url"]

    # read pdf data
    loader = PyPDFLoader(url)
    pages = loader.load()
    outpath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "text",
        filename + ".txt",
    )
    # write pdf data to file
    with open(outpath, "w") as doc:
        for page in pages:
            doc.write(page.page_content)
    print("pdf extraction finished!")
