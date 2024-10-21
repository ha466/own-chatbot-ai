from api import client, MODEL
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('vader_lexicon')

def generate_response(user_input, entities, conversation_history):
    # Prepare the messages for the API
    messages = [
        {"role": "system", "content": "You are Iris, an advanced AI assistant. Respond concisely and helpfully."}
    ]
    messages.extend(conversation_history[-5:])  # Include last 5 messages for context
    messages.append({"role": "user", "content": f"User input: {user_input}\nDetected entities: {entities}"})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Error calling API: {str(e)}")

def analyze_sentiment(text):
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        
        if sentiment_scores['compound'] > 0.05:
            return "positive"
        elif sentiment_scores['compound'] < -0.05:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        raise Exception(f"Error analyzing sentiment: {str(e)}")

def extract_keywords(text, num_keywords=5):
    try:
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]

        # Get frequency distribution
        fdist = FreqDist(words)

        # Return top N keywords
        keywords = [word for word, _ in fdist.most_common(num_keywords)]
        return keywords
    except Exception as e:
        raise Exception(f"Error extracting keywords: {str(e)}")

def summarize_text(text, max_length=50):
    try:
        # Tokenize sentences
        sentences = nltk.sent_tokenize(text)
        
        # Get the first two sentences or up to max_length characters
        summary = ' '.join(sentences[:2])
        if len(summary) > max_length:
            summary = summary[:max_length] + '...'
        return summary
    except Exception as e:
        raise Exception(f"Error summarizing text: {str(e)}")

def extract_named_entities(text):
    try:
        # Tokenize and tag parts of speech
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        
        # Perform named entity recognition
        named_entities = ne_chunk(pos_tags)
        
        # Extract named entities
        entities = []
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                entity = ' '.join(c[0] for c in chunk)
                entity_type = chunk.label()
                entities.append((entity, entity_type))
        
        return entities
    except Exception as e:
        raise Exception(f"Error extracting named entities: {str(e)}")

def analyze_text_complexity(text):
    try:
        words = word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Determine complexity based on these metrics
        if avg_word_length > 5 and avg_sentence_length > 20:
            return "High"
        elif avg_word_length > 4 and avg_sentence_length > 15:
            return "Medium"
        else:
            return "Low"
    except Exception as e:
        raise Exception(f"Error analyzing text complexity: {str(e)}")