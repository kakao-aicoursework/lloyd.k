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

human_template = ("해당 질문에 간단하게 답해주세요\n"
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
    chat_history: str

    def answer(self):
        # Our chatbot is not very smart right now...
        answer = "I don't know!"
        self.chat_history.append((self.question, answer))

    def handle_submit(self, form_data):
        self.question = form_data['qa']
        ans = chain.run(qa=self.question)
        self.chat_history += f"question: {self.question} \n answer: {self.answer}"
        print(ans)

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
    qa_str = State.chat_history
    qa_pairs = []

    return pc.box(
        *[
            qa(question, answer)
            for question, answer in qa_pairs
        ]
    )


def action_bar() -> pc.Component:
    return pc.form(
        pc.hstack(
            pc.input(
                placeholder="Ask a question",
                style=style.input_style,
                id="qa",
            ),
            pc.button("Ask", style=style.button_style, type_="submit"),
        ),
        on_submit=State.handle_submit
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