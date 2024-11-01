After running all 3 sentiment analysis we concluded that Machine learning performed better

Example: “Don’t buy!!!”
Review Body: "Constantly falls off of my vents, my phone constantly falls out. It’s caused me to almost get into an accident. Don’t buy!!!! Not safe!!!"
Outputs: Machine Learning Model:
Overall Sentiment: 1 star, with a strong negative score (0.91).
Aspect Sentiments:
“buy” – 1 star (0.90)
“falls” – 1 star (0.90)
“cause” – 1 star (0.89)
“get” – 1 star (0.91)
Why It’s Better: The machine learning model pinpoints multiple specific issues ("falls," "cause") and applies clear negative sentiments to each, which reflect the user’s frustration. It highlights multiple problems with a consistent sentiment score, effectively mirroring the user’s dissatisfaction.
Local LLM:

Overall Sentiment: Neutral (Score: 0.5)
Aspect Sentiments:
“Safety” – Neutral (Score: 0.5)
Why It Falls Short: The Local LLM misinterprets the intensity of the user's negative feedback. By labeling the sentiment as "Neutral," it loses the strong dissatisfaction expressed by the reviewer. It also identifies only a single aspect ("Safety"), missing out on critical points like “buy,” “falls,” and “cause.”
SpaCy:

Overall Sentiment: Negative
Aspect Sentiments:
“falls” – Negative
Why It Falls Short: SpaCy does pick up on a negative sentiment for "falls," but it lacks the depth of coverage. It identifies fewer aspects, missing important points like “cause” and “get,” making the output less comprehensive.


Improvements:


### SpaCy (15 minutes)
- Capture More Aspects**: Include nouns in aspect extraction, not just verbs. This way, we catch important words like “price” and “quality.”
- Use Dependency Parsing**: By analyzing word relationships, we can better link sentiment words to their aspects, like “bad quality.”
- Add a Sentiment Lexicon**: SpaCy lacks built-in sentiment, so using a lexicon of positive/negative words or training it on a small sentiment dataset could boost accuracy.

### Machine Learning Model ( 3 minutes)
- Wider Aspect Identification**: Add named entity recognition to identify specific product features and brand names, not just actions.
- Fine-Tune Sentiment Scoring**: Try other models or fine-tune BERT to capture nuanced opinions, like “fairly durable.”
- More Training Data**: Training the model on custom, labeled data would make it more precise for real-world reviews.

### Local LLM ( 5 hours)
- Prompt Engineering**: Design clearer prompts to ensure the LLM provides structured, detailed outputs.