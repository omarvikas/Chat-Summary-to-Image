import streamlit as st
import os
import json
import google.generativeai as genai
import requests
from PIL import Image
import io

# Langchain imports with comprehensive fallbacks
ChatGoogleGenerativeAI = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    try:
        from langchain.llms import GooglePalm as ChatGoogleGenerativeAI
    except ImportError:
        pass

ChatOpenAI = None
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        pass

# Message classes with fallbacks
HumanMessage = None
AIMessage = None
try:
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    try:
        from langchain.schema import HumanMessage, AIMessage
    except ImportError:
        # Create simple message classes if langchain is not available
        class HumanMessage:
            def __init__(self, content):
                self.content = content
                
        class AIMessage:
            def __init__(self, content):
                self.content = content

# PromptTemplate with fallbacks
PromptTemplate = None
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
    except ImportError:
        pass

# Page configuration
st.set_page_config(
    page_title="Langchain Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""
if "conversation_sentiment" not in st.session_state:
    st.session_state.conversation_sentiment = ""

def initialize_gemini_chat(api_key):
    """Initialize Gemini chat model"""
    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Always use direct API since langchain integration has issues
        return "direct_api"
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

def get_gemini_response(prompt, api_key, conversation_history=None):
    """Get response from Gemini using direct API if langchain fails"""
    try:
        genai.configure(api_key=api_key)
        
        # Try different model names - Gemini has updated their model names
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro'
        ]
        
        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except Exception:
                continue
        
        if not model:
            # List available models for debugging
            try:
                available_models = genai.list_models()
                model_list = [m.name for m in available_models if 'generateContent' in m.supported_generation_methods]
                st.error(f"Available models: {model_list}")
                if model_list:
                    model = genai.GenerativeModel(model_list[0])
                else:
                    st.error("No suitable models found")
                    return None
            except Exception as e:
                st.error(f"Error listing models: {str(e)}")
                return None
        
        # Add context if available
        if conversation_history:
            context = ""
            for msg in conversation_history[-6:]:  # Last 6 messages
                if isinstance(msg, dict):
                    if msg.get('role') == 'user':
                        context += f"Human: {msg.get('content', '')}\n"
                    elif msg.get('role') == 'assistant':
                        context += f"Assistant: {msg.get('content', '')}\n"
                elif hasattr(msg, 'content'):
                    if isinstance(msg, HumanMessage):
                        context += f"Human: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        context += f"Assistant: {msg.content}\n"
            
            if context:
                contextualized_prompt = f"""Previous conversation context:
{context}

Current question: {prompt}

Please respond naturally, taking into account the conversation history."""
            else:
                contextualized_prompt = prompt
        else:
            contextualized_prompt = prompt
        
        response = model.generate_content(contextualized_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting Gemini response: {str(e)}")
        return None

def initialize_openai_chat():
    """Initialize OpenAI chat model for summary"""
    try:
        # Get OpenAI API key from session state
        openai_api_key = st.session_state.openai_api_key
        if not openai_api_key:
            st.error("OpenAI API key not provided!")
            return None
        
        # Always use direct API approach for better compatibility
        return "direct_openai"
    except Exception as e:
        st.error(f"Error initializing OpenAI: {str(e)}")
        return None

def get_openai_response(prompt, api_key):
    """Get response from OpenAI using direct API if langchain fails"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error with OpenAI direct API: {str(e)}")
        return None

def generate_image_with_dalle(prompt, api_key):
    """Generate image using DALL-E 2 API"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Create a more visual prompt for DALL-E
        visual_prompt = f"""Create a visual representation of this conversation summary: {prompt}. 
        Make it artistic, colorful, and engaging. Focus on the key themes and emotions."""
        
        response = client.images.generate(
            model="dall-e-2",
            prompt=visual_prompt[:1000],  # DALL-E has prompt length limits
            size="512x512",
            n=1,
        )
        
        # Get the image URL
        image_url = response.data[0].url
        
        # Download and display the image
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            image = Image.open(io.BytesIO(image_response.content))
            return image, image_url
        else:
            st.error("Failed to download the generated image")
            return None, None
            
    except Exception as e:
        st.error(f"Error generating image with DALL-E: {str(e)}")
        return None, None

def create_image_prompt_from_summary(summary):
    """Create a better visual prompt for DALL-E based on conversation summary"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.session_state.openai_api_key)
        
        prompt_creation_request = f"""
        Based on this conversation summary, create a detailed visual prompt for DALL-E image generation.
        The prompt should be descriptive, artistic, and capture the essence of the conversation.
        Keep it under 800 characters and focus on visual elements, colors, mood, and style.
        
        Conversation Summary: {summary}
        
        Visual Prompt for DALL-E:
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_creation_request}],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error creating image prompt: {str(e)}")
        return f"Artistic visualization of: {summary[:500]}"

def format_conversation_for_summary():
    """Format conversation history for summary"""
    conversation_text = ""
    for message in st.session_state.messages:
        if isinstance(message, dict):
            if message.get('role') == 'user':
                conversation_text += f"Human: {message.get('content', '')}\n"
            elif message.get('role') == 'assistant':
                conversation_text += f"AI: {message.get('content', '')}\n"
        elif hasattr(message, 'content'):
            if isinstance(message, HumanMessage):
                conversation_text += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                conversation_text += f"AI: {message.content}\n"
    return conversation_text

def generate_summary_and_sentiment():
    """Generate conversation summary and sentiment analysis"""
    if not st.session_state.messages:
        return "No conversation to summarize.", "Neutral"
    
    openai_llm = initialize_openai_chat()
    if not openai_llm:
        return "Error: Could not initialize OpenAI for summary.", "Error"
    
    conversation_text = format_conversation_for_summary()
    
    try:
        # Summary prompt
        summary_prompt = f"""
        Please provide a concise summary of the following conversation in under 150 words:
        
        {conversation_text}
        
        Summary:
        """
        
        # Sentiment analysis prompt
        sentiment_prompt = f"""
        Analyze the sentiment of the following conversation and provide a brief sentiment analysis.
        Classify the overall sentiment as: Positive, Negative, Neutral, or Mixed.
        Provide a one-sentence explanation.
        
        {conversation_text}
        
        Sentiment Analysis:
        """
        
        openai_api_key = st.session_state.openai_api_key
        
        # Use direct API approach
        summary = get_openai_response(summary_prompt, openai_api_key)
        sentiment = get_openai_response(sentiment_prompt, openai_api_key)
        
        return summary, sentiment
    except Exception as e:
        return f"Error generating summary: {str(e)}", "Error in sentiment analysis"

def main():
    st.title("🤖 Langchain Chatbot with Gemini & OpenAI")
    st.markdown("---")
    
    # Sidebar for API key input
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        if not st.session_state.chat_started:
            st.markdown("**Enter your API keys to get started:**")
            
            gemini_api_key = st.text_input(
                "Gemini API Key:",
                type="password",
                help="Get your API key from Google AI Studio"
            )
            
            openai_api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                help="Get your API key from OpenAI Platform"
            )
            
            if st.button("Start Chat"):
                if gemini_api_key and openai_api_key:
                    st.session_state.gemini_api_key = gemini_api_key
                    st.session_state.openai_api_key = openai_api_key
                    st.session_state.chat_started = True
                    st.rerun()
                else:
                    if not gemini_api_key:
                        st.error("Please enter your Gemini API key!")
                    if not openai_api_key:
                        st.error("Please enter your OpenAI API key!")
        
        else:
            st.success("✅ Chat session active")
            st.info("💡 Your conversation is being tracked for context awareness")
            
            # Add a button to test API and show available models
            if st.button("🔍 Test Gemini API & Show Models"):
                try:
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    models = genai.list_models()
                    available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                    st.success(f"✅ Gemini API Key is valid!")
                    st.info(f"Available models: {', '.join(available_models)}")
                except Exception as e:
                    st.error(f"❌ Gemini API test failed: {str(e)}")
            
            if st.button("🔚 End Conversation", type="primary"):
                st.session_state.conversation_ended = True
                st.rerun()
    
    # Main chat interface
    if not st.session_state.chat_started and not st.session_state.conversation_ended:
        st.info("👈 Please enter both API keys in the sidebar to start chatting!")
        
        # Instructions
        st.markdown("""
        ### How to use this chatbot:
        
        1. **Get API Keys**: 
           - **Gemini API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to get your free API key
           - **OpenAI API Key**: Visit [OpenAI Platform](https://platform.openai.com/api-keys) to get your API key
        2. **Enter Both Keys**: Paste both API keys in the sidebar
        3. **Start Chatting**: Click "Start Chat" to begin your conversation
        4. **Context Awareness**: The bot remembers your conversation history within the session
        5. **End & Summarize**: Click "End Conversation" to get an AI-generated summary and sentiment analysis
        6. **Generate Visual**: Create an artistic image representation of your conversation using DALL-E 2
        
        ### Features:
        - 🧠 **Context-aware conversations** using Langchain
        - 🚀 **Powered by Google Gemini** for chat responses
        - 📊 **OpenAI-powered summaries** and sentiment analysis
        - 🎨 **DALL-E 2 image generation** from conversation summaries
        - 💾 **Session-based memory** for natural conversations
        
        ### Why both API keys?
        - **Gemini**: Powers the main chat conversation
        - **OpenAI**: Generates summaries, sentiment analysis, and DALL-E 2 images
        """)
    
    elif st.session_state.conversation_ended:
        st.header("📋 Conversation Summary")
        
        # Generate summary and sentiment if not already done
        if not st.session_state.conversation_summary:
            with st.spinner("Generating summary and sentiment analysis..."):
                summary, sentiment = generate_summary_and_sentiment()
                st.session_state.conversation_summary = summary
                st.session_state.conversation_sentiment = sentiment
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("💬 Conversation Summary")
            st.write(st.session_state.conversation_summary)
        
        with col2:
            st.subheader("😊 Sentiment Analysis")
            st.write(st.session_state.conversation_sentiment)
        
        st.markdown("---")
        
        # Image generation section
        st.header("🎨 Visual Representation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🖼️ Generate Image from Summary", type="secondary"):
                with st.spinner("Creating visual representation with DALL-E 2..."):
                    # Create a better prompt for DALL-E
                    visual_prompt = create_image_prompt_from_summary(st.session_state.conversation_summary)
                    st.write(f"**Image Prompt:** {visual_prompt}")
                    
                    # Generate image
                    image, image_url = generate_image_with_dalle(visual_prompt, st.session_state.openai_api_key)
                    
                    if image:
                        st.success("✅ Image generated successfully!")
                        st.image(image, caption="AI-Generated Visual Representation of Your Conversation", use_column_width=True)
                        
                        # Provide download link
                        if image_url:
                            st.markdown(f"[🔗 Download Full Resolution Image]({image_url})")
                    else:
                        st.error("Failed to generate image. Please try again.")
        
        with col2:
            if st.button("🔄 Start New Conversation"):
                # Reset session state
                st.session_state.messages = []
                st.session_state.chat_started = False
                st.session_state.gemini_api_key = ""
                st.session_state.openai_api_key = ""
                st.session_state.conversation_ended = False
                st.session_state.conversation_summary = ""
                st.session_state.conversation_sentiment = ""
                st.rerun()
    
    else:
        # Chat interface
        st.header("💬 Chat with Gemini")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if isinstance(message, dict):
                    if message.get('role') == 'user':
                        with st.chat_message("user"):
                            st.write(message.get('content', ''))
                    elif message.get('role') == 'assistant':
                        with st.chat_message("assistant"):
                            st.write(message.get('content', ''))
                elif hasattr(message, 'content'):
                    if isinstance(message, HumanMessage):
                        with st.chat_message("user"):
                            st.write(message.content)
                    elif isinstance(message, AIMessage):
                        with st.chat_message("assistant"):
                            st.write(message.content)
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Initialize Gemini model
            gemini_llm = initialize_gemini_chat(st.session_state.gemini_api_key)
            
            if gemini_llm:
                # Add user message to history
                user_message = {"role": "user", "content": prompt}
                st.session_state.messages.append(user_message)
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Generate response with context
                try:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            # Always use direct API for better compatibility
                            response_text = get_gemini_response(
                                prompt, 
                                st.session_state.gemini_api_key, 
                                st.session_state.messages[:-1]  # Exclude current message
                            )
                            
                            if response_text:
                                # Display response
                                st.write(response_text)
                                
                                # Add AI response to history
                                ai_message = {"role": "assistant", "content": response_text}
                                st.session_state.messages.append(ai_message)
                            else:
                                st.error("Failed to get response from Gemini.")
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info("Please check your API key and try again.")
            else:
                st.error("Failed to initialize Gemini model. Please check your API key.")

if __name__ == "__main__":
    main()