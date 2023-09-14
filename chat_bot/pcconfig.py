import pynecone as pc

class ChatbotConfig(pc.Config):
    pass

config = ChatbotConfig(
    app_name="chat_bot",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)