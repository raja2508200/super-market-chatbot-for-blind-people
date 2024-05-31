# super-market-chatbot-for-blind-people
Objectives:
Assist visually impaired users in navigating and interacting within a supermarket.
Answer questions about the location of items, prices, billing, and final total.
Key Features:
Intent Recognition: Classify user queries into predefined intents.
Conversational Flow: Guide users through a series of questions and responses.
Accessibility: Provide audio responses and navigation assistance.
Steps to Develop the Chatbot:
Define Intents:

Location Inquiry: "Where is the vegetable section?"
Price Inquiry: "How much is [item]?"
Billing Inquiry: "What's my total bill?"
Final Rate Inquiry: "What is the final amount?"
Dataset Creation:

Collect Data: Gather sample queries for each intent.
Label Data: Annotate each query with its corresponding intent.
Preprocessing:

Tokenization: Split sentences into words.
Normalization: Convert text to lowercase, remove punctuation, etc.
Embedding: Use word embeddings like Word2Vec or GloVe to convert words into vectors.
Model Design (Using CNN):

Input Layer: Sequence of word vectors.
Convolutional Layers: Apply multiple filters to extract features.
Pooling Layers: Reduce dimensionality while retaining important features.
Fully Connected Layers: Classify the features into one of the intents.
Output Layer: Softmax activation to output intent probabilities.
Training the Model:

Loss Function: Use categorical cross-entropy.
Optimizer: Use Adam optimizer.
Evaluation: Split the dataset into training and validation sets, use metrics like accuracy.
Deploying the Chatbot:

Integration: Connect the model to a chatbot framework (e.g., Rasa, Dialogflow).
Voice Interaction: Use Text-to-Speech (TTS) and Speech-to-Text (STT) for audio interaction.
Testing: Conduct thorough testing to ensure the chatbot accurately understands and responds to user queries.
