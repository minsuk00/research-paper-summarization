from typing import Literal, Union
import os


def handle_input(document, summaries, mode):
    text_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "text",
    )
    doc_path = os.path.join(
        text_dir,
        document + ".txt",
    )
    if mode == "file":
        # parse document
        with open(doc_path, "r") as file:
            document = file.read()

        # parse summaries
        if isinstance(summaries, list):
            res = []
            for summ in summaries:
                summ_file = os.path.join(text_dir, summ + ".txt")
                with open(summ_file, "r") as file:
                    res.append(file.read())
            summaries = res
        elif isinstance(summaries, str):
            summ_file = os.path.join(text_dir, summaries + ".txt")
            with open(summ_file, "r") as file:
                # store summary to list
                summaries = [file.read()]
    elif mode == "text":
        # account for when only summary is text input. try to get document from file
        if len(document) < 20:
            try:
                print("Trying to read document from file...")
                with open(doc_path, "r") as file:
                    document = file.read()
                print("success!")
            except:
                print("Failed to read document from file. using document as text input")
        if isinstance(summaries, str):
            summaries = [summaries]

    return document, summaries


# =====Blanc=====
from blanc import BlancHelp


# typically 0~0.3 (0: useless, 0.3: useful)
def blanc(
    document: str,
    summaries: Union[list[str], str],
    mode: Literal["file", "text"] = "file",
):
    document, summaries = handle_input(document, summaries, mode)
    # print(document)
    blanc_help = BlancHelp()
    print("=====Blanc Score=====")
    # scores = blanc_help.eval_pairs(document, summaries)
    for i, summ in enumerate(summaries):
        score = blanc_help.eval_once(document, summ)
        print(f"Summary {i}: ", score)
    # for i, score in enumerate(scores):
    #     print(f"Summary {i}: ", score)


# =====Estime=====
from blanc import Estime


# focus on accuracy. number of alarms
def estime(
    document: str,
    summaries: Union[list[str], str],
    mode: Literal["file", "text"] = "file",
):
    document, summaries = handle_input(document, summaries, mode)

    output = ["alarms", "alarms_adjusted", "soft", "coherence"]
    # output = ["alarms", "soft"]
    estimator = Estime(output=output)
    scores = estimator.evaluate_claims(document, summaries)
    print("=====Estime Score=====")
    for i, *score in enumerate(scores):
        print(f"Summary {i}:")
        for j, crit in enumerate(output):
            print(f"{crit}: {score[0][j]}")


# =====Shannon Score=====
from blanc import Shannon


# typically 0~1. higher the better
def shannon_score(
    document: str,
    summaries: Union[list[str], str],
    mode: Literal["file", "text"] = "file",
):
    document, summaries = handle_input(document, summaries, mode)
    judge = Shannon()
    print("=====Shannon Score=====")
    for i, summ in enumerate(summaries):
        b, h, f, *_ = judge.go(document, summ)
        ss = (h - b) / (f - b)
        print(f"Summary {i}: ", ss)


# =====G-eval=====
from openai import OpenAI
import pandas as pd
import os
import geval_prompt as P

# import boto3


def g_eval(
    document: str,
    summaries: Union[list[str], str],
    mode: Literal["file", "text"] = "file",
    verbose: bool = False,
):
    """implementation of G-eval framework. can take in both file or text of doc/summary(s)

    Args:
        document (str): filename or actual text of document (file should be stored in /text dir. exclude .txt)
        summary (Union[list[str], str]): 1 or list of files/actual text.
        mode (Literal[&quot;file&quot;, &quot;text&quot;], optional): whether to parse file or the actual text. Defaults to "file".
        verbose (bool): log intermediate results
    """
    document, summaries = handle_input(document, summaries, mode)
    # document & summaries are stored as str | list[str]. summaries are stored in list

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # bedrock = boto3.client(service_name="bedrock-runtime")

    evaluation_metrics = {
        "Relevance": (P.RELEVANCY_SCORE_CRITERIA, P.RELEVANCY_SCORE_STEPS),
        "Coherence": (P.COHERENCE_SCORE_CRITERIA, P.COHERENCE_SCORE_STEPS),
        "Consistency": (P.CONSISTENCY_SCORE_CRITERIA, P.CONSISTENCY_SCORE_STEPS),
        "Fluency": (P.FLUENCY_SCORE_CRITERIA, P.FLUENCY_SCORE_STEPS),
    }

    # summaries = {"Summary 1": doc["summary1"], "Summary 2": doc["summary2"]}
    summ_dict = {}
    for i, summ in enumerate(summaries):
        summ_dict[f"Summary {i}"] = summ
    data = {"Evaluation Type": [], "Summary Type": [], "Score": []}

    print("Evaluating...")
    for eval_type, (criteria, steps) in evaluation_metrics.items():
        if verbose:
            print(f"evaluating {eval_type}...")
        for summ_type, summary in summ_dict.items():
            data["Evaluation Type"].append(eval_type)
            data["Summary Type"].append(summ_type)
            result = get_geval_score(
                criteria, steps, document, summary, eval_type, client
            )
            try:
                score_num = int(result.strip())
            except:
                # if result does not contain valid output
                score_num = -1
            if verbose:
                print(f"{summ_type}: {score_num}")
            data["Score"].append(score_num)

    pivot_df = pd.DataFrame(data, index=None).pivot(
        index="Evaluation Type", columns="Summary Type", values="Score"
    )
    print("=========================================")
    print(pivot_df)
    # styled_pivot_df = pivot_df.style.apply(highlight_max, axis=1)
    # display(styled_pivot_df)


def get_geval_score(
    criteria: str, steps: str, document: str, summary: str, metric_name: str, client
):
    prompt = P.EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        document=document,
        summary=summary,
    )
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        # model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content
