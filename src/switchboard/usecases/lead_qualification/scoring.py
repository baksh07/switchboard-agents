"""Step 1 of the pipeline: turn a raw lead message into a validated LeadScore."""

import argparse

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from switchboard.core.llm import get_llm
from switchboard.core.structured import invoke_structured
from switchboard.usecases.lead_qualification.prompts import SCORING_SYSTEM_PROMPT
from switchboard.usecases.lead_qualification.state import LeadScore

SAMPLE_LEADS = [
    (
        "Hi, I'm Priya Sharma, CTO at Acme Robotics (priya@acmerobotics.example). We're 450 people and "
        "want to automate our ops approvals this quarter. Budget is approved. Can we talk next week?"
    ),
    "hey im a student doing a thesis on CRMs, can u send me free stuff? - tom, tom.k@gmail.com",
    (
        "Hello, Ada Park here, Head of Operations at Initech (ada@initech.example). "
        "Just exploring options for next year, no rush."
    ),
]


def score_lead(lead_text: str, llm: BaseChatModel) -> LeadScore:
    messages = [SystemMessage(SCORING_SYSTEM_PROMPT), HumanMessage(lead_text)]
    return invoke_structured(llm, LeadScore, messages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the sample leads.")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    llm = get_llm(args.model)
    for lead in SAMPLE_LEADS:
        score = score_lead(lead, llm)
        print(score.model_dump_json(indent=2), "\n")
