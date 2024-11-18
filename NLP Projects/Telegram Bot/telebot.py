import logging
from aiogram import Bot, Dispatcher, executor, types
# To read commands from .env
from dotenv import load_dotenv
import os
import openai
import sys

class Reference:
    '''
    A class to store previous responses from the ChatGPT API
    '''
    def __init__(self) -> None:
        self.response = ""


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

reference = Reference()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# model name
MODEL_NAME = "gpt-3.5-turbo"

# Initialize bot and dispatcher
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot)

def clear_paste():
    """
    A fucntion to clear the previous conversation and context.
    """
    reference.response = ""


@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    This handler receives messages with `/start` or `/help`command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.reply(f"Hello, I am Echo Bot!\nCreated by GOD\nHow can I assist you?")


@dispatcher.message_handler(commands=['clear'])
async def clear(message: types.Message):
    """
    A handler to clear the previous conversation and context
    """
    clear_paste()
    await message.reply("I have cleared the past conversation and context")


@dispatcher.message_handler(commands=["help"])
async def helper(message: types.Message):
    """
    A handler to display the help menu
    """
    help_command = """
    Hi There, I am ChatGPT Telegram bot! Please follow these commands - 
    /start - to start the conversation
    /clear - to clear the past conversation and context
    /help - to get this help menu
    """
    await message.reply(help_command )


@dispatcher.message_handler()
async def chatgpt(message: types.Message):
    """
    A handler to process the user's input and generate a response using ChatGPT's API.
    """
    print(f">>> USER: \n\t{message.text}")
    try:
        response = openai.ChatCompletion.create(
        model = MODEL_NAME,
        messages = [
            {"role": "assistant", "content": reference.response}, # role assistant
            {"role": "user", "content": message.text} #our query 
        ]
    )
        reference.response = response.choices[0]['message']['content']
        print(f">>> ChatGPT: \n\t{reference.response}")
        await bot.send_message(chat_id=message.chat.id, text=reference.response)

    except openai.error.RateLimitError as e:
        if "quota" in str(e).lower():
            await message.reply("It seems my usage quota for this month has been exhausted. Please try again later or contact the admin.")
        else:
            await message.reply("Too many requests! Please wait a moment before trying again.")
    except openai.error.AuthenticationError:
        await message.reply("Authentication failed. Please contact the admin.")
    except openai.error.Timeout:
        await message.reply("OpenAI is taking too long to respond. Please try again later.")
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API Error: {e}")
        await message.reply("Something went wrong with the AI system. Please try again later.")
    except Exception as e:
        logging.error(f"Unexpected Error: {e}")
        await message.reply("An unexpected error occurred. Please contact the support team.")


if __name__=="__main__":
    executor.start_polling(dispatcher, skip_updates=False)
