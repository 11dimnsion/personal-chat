import asyncio
import os
from collections.abc import AsyncGenerator

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient
from schema import ChatHistory, ChatMessage

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Rendered.ai Introduction"
APP_ICON = "🧰"



async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        agent_url = os.getenv("AGENT_URL", "http://localhost")
        st.session_state.agent_client = AgentClient(agent_url)
        print("URL:",agent_url)

    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            history: ChatHistory = agent_client.get_history(thread_id=thread_id)
            messages = history.messages
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    models = {
        "OpenAI GPT-4o-mini (streaming)": "gpt-4o-mini",
        # "Gemini 1.5 Flash (streaming)": "gemini-1.5-flash",
        # "Claude 3 Haiku (streaming)": "claude-3-haiku",
        # "llama-3.1-70b on Groq": "llama-3.1-70b",
    }
    # Config options
    with st.sidebar:
        st.image("media/Rendered-stacked-logo.webp", use_column_width=True)

        st.header(f"{APP_TITLE}")
        ""
        "Helpful chatbot for learning about Rendered.ai and how to integrate our tools"
        with st.popover(":material/settings: Settings", use_container_width=True):
            m = st.radio("LLM to use", options=models.keys())
            model = models[m]
            use_streaming = st.toggle("Stream results", value=True)

        # New: Predefined Response Buttons
        st.markdown("### Quick Responses")
        predefined_prompts = {
            "About RenderedAI": "What does RenderedAI do?",
            "Synthetic data platform": "Why would I need a synthetic data platform?",
            "Synthetic Data Automation": "How can a platform like RenderedAI help me with automation?",
        }

        for label, prompt in predefined_prompts.items():
            if st.button(label):
                await send_predefined_prompt(prompt)

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        # st.markdown(
        #     f"Thread ID: **{st.session_state.thread_id}**",
        #     help=f"Set URL query parameter ?thread_id={st.session_state.thread_id} to continue this conversation",
        # )

        "This tool was based on the [Agent Service Toolkit] :material/favorite: (https://github.com/JoshuaC215/agent-service-toolkit)"

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. I may take a few seconds to boot up when you send your first message. Ask me anything!"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        if use_streaming:
            stream = agent_client.astream(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
            )
            await draw_messages(stream, is_new=True)
        else:
            response = await agent_client.ainvoke(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
            )
            messages.append(response)
            st.chat_message("ai").write(response.content)
        st.rerun()  # Clear stale containers

    # If messages have been generated, show feedback widget
    if len(messages) > 0:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        #print("CHECKING MESSAGE TYPE:", type(msg))
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            #print("WROTE A STRING!!!!!!!")
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()
        #print("CHECKING MESSAGE TYPE AGAIN:", type(msg))
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage|str = await anext(messages_agen)
                            if isinstance(tool_result, str):
                                if not streaming_placeholder:
                                    if last_message_type != "ai":
                                        last_message_type = "ai"
                                        st.session_state.last_message = st.chat_message("ai")
                                    with st.session_state.last_message:
                                        streaming_placeholder = st.empty()

                                streaming_content = msg
                                streaming_placeholder.write(streaming_content)
                                print("WROTE A STRING!!!!!!!")
                                continue
                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        await agent_client.acreate_feedback(
            run_id=latest_run_id,
            key="human-feedback-stars",
            score=normalized_score,
            kwargs={"comment": "In-line human feedback"},
        )
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")

async def send_predefined_prompt(prompt: str) -> None:
    """
    Sends a predefined prompt by appending it to the chat history and processing it.

    Args:
        prompt (str): The predefined prompt to send.
    """
    # Append the predefined prompt as a human message
    st.session_state.messages.append(ChatMessage(type="human", content=prompt))
    
    # Retrieve the agent client from the session state
    agent_client: AgentClient = st.session_state.agent_client

    # Retrieve the current model and streaming settings
    # It's assumed that these are stored in the session state
    model = st.session_state.get("model")

    response = await agent_client.ainvoke(
        message=prompt,
        model=model,
        thread_id=st.session_state.thread_id,
    )
    # Append the agent's response to the chat history
    st.session_state.messages.append(response)
    # Display the agent's response in the chat
    st.chat_message("ai").write(response.content)

    # Rerun the Streamlit app to update the UI
    st.rerun()


if __name__ == "__main__":
    asyncio.run(main())
