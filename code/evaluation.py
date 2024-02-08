from blanc import BlancHelp, BlancTune
import time
from typing import Literal, Union

# =====Blanc=====

# =====Estime=====

# =====Shannon Score=====

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
):
    """implementation of G-eval framework. can take in both file or text of doc/summary(s)

    Args:
        document (str): filename or actual text of document (file should be stored in /text dir. exclude .txt)
        summary (Union[list[str], str]): 1 or list of files/actual text.
        mode (Literal[&quot;file&quot;, &quot;text&quot;], optional): whether to parse file or the actual text. Defaults to "file".
    """
    if mode == "file":
        # parse document
        text_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "text",
        )
        doc_path = os.path.join(
            text_dir,
            document + ".txt",
        )
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
        if isinstance(summaries, str):
            summaries = [summaries]
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

    for eval_type, (criteria, steps) in evaluation_metrics.items():
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
            print(f"{summ_type}: {score_num}")
            data["Score"].append(score_num)

    pivot_df = pd.DataFrame(data, index=None).pivot(
        index="Evaluation Type", columns="Summary Type", values="Score"
    )
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
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content
