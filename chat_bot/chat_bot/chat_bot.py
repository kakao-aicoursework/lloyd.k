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

chat = ChatOpenAI(temperature=0.8)
system_message = "assistant는 API를 설명하는 챗봇이다."
system_message_prompt = SystemMessage(content=system_message)

with open("./chat_bot/project_data_카카오싱크.txt", "rt") as f:
    assistant_template = f.read()

human_template = ("아래 질문에 간단하게 답해주세요\n"
                  "{qa}\n"
                  )


assistant_message_prompt = HumanMessagePromptTemplate.from_template(assistant_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, assistant_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)


class State(pc.State):

    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[list[str]]

    def answer(self):
        response = chain.run(qa=self.question)
        answer = ""
        self.chat_history.append([self.question, answer])
        self.question = ""
        yield

        if response:
            answer += response
            self.chat_history[-1] = [
                self.chat_history[-1][0],
                answer,
            ]
        yield


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
    return pc.box(
        pc.foreach(
            State.chat_history,
            lambda messages: qa(messages[0], messages[1]),
        )
    )


def action_bar() -> pc.Component:
    return pc.hstack(
        pc.input(
            value=State.question,
            placeholder="Ask a question",
            on_change=State.set_question,
            style=style.input_style,
        ),
        pc.button(
            "Ask",
            on_click=State.answer,
            style=style.button_style,
        ),
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