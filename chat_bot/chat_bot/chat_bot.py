import pynecone as pc
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)
from chat_bot.chatapp import style

import os
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

class State(pc.State):

    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    def answer(self):
        # Our chatbot is not very smart right now...
        answer = "I don't know!"
        self.chat_history.append((self.question, answer))

def qa(question: str, answer: str) -> pc.Component:
    return pc.box(
        pc.box(
            pc.text(question, style=style.question_style),
            text_align="right",
        ),
        pc.box(
            pc.text(answer, style=style.answer_style),
            text_align="left",
        ),
        margin_y="1em",
    )


def chat() -> pc.Component:
    qa_pairs = [
        (
            "What is Reflex?",
            "A way to build web apps in pure Python!",
        ),
        (
            "What can I make with it?",
            "Anything from a simple website to a complex web app!",
        ),
    ]
    return pc.box(
        *[
            qa(question, answer)
            for question, answer in qa_pairs
        ]
    )


def action_bar() -> pc.Component:
    return pc.hstack(
        pc.input(
            placeholder="Ask a question",
            style=style.input_style,
        ),
        pc.button("Ask", style=style.button_style),
    )


def index() -> pc.Component:
    return pc.container(
        chat(),
        action_bar(),
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()