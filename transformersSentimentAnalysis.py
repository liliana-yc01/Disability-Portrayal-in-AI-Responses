from transformers import pipeline

# Initialize pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Sample short inputs
data = ["Hello", "I hate you"]
short_results = sentiment_pipeline(data)
print("Short results:", short_results)

# Long text
text = """
Olivia Breen: Breaking Barriers in the World of Paralympic Athletics Olivia Breen, a name synonymous with resilience, determination, and excellence, is a prominent figure in the world of Paralympic athletics. Living with cerebral palsy, a neurological condition that affects movement and coordination, Olivia has never let her disability define her potential. Instead, she has used it as a driving force, pushing past limitations and shattering barriers to become one of the leading sprinters in the world.
Early Life and Diagnosis
Born on April 26, 1996, in the United Kingdom, Olivia Breen was diagnosed with cerebral palsy at a young age. The condition impacts muscle control, which in her case, resulted in challenges with coordination and balance. Despite the obstacles that cerebral palsy presented, Olivia’s parents supported her in every way possible, encouraging her to pursue her dreams and take part in activities that brought her joy.
From a young age, Olivia had an interest in sports. Initially, she started participating in a variety of physical activities, from swimming to athletics. It was in athletics, specifically sprinting, where she would find her true passion and shine on the global stage.
Path to Paralympic Glory
Olivia's journey into Paralympic athletics began when she was introduced to the sport by a local disability sports club. The supportive environment and competitive nature of athletics appealed to Olivia, and she began to show promise in sprinting. She joined a training group and started working with coaches who could help her harness her potential.
Her breakthrough moment came in 2012 when she was selected to represent Great Britain at the London Paralympic Games. Although she did not win a medal, the experience ignited a fierce desire to succeed at the highest level of competition. The event proved to be a turning point, marking the beginning of her rise in the sport.
Success on the Track
Olivia Breen's dedication and hard work soon paid off as she began dominating in the T38 classification for athletes with cerebral palsy. Her specialties are the 100 meters and the long jump, events in which she has excelled on the world stage. In 2014, Olivia made a significant impact by winning a gold medal at the IPC Athletics European Championships in Swansea, Wales. Her performance was a testament to her years of preparation, grit, and commitment to her craft.
Over the years, Olivia continued to build her medal collection at major international competitions, including the World Championships and the European Championships. However, it was the 2020 Tokyo Paralympics (held in 2021) that solidified Olivia’s place among the greats. She earned a bronze medal in the T38 100 meters, adding another prestigious honor to her career. Her achievement at Tokyo also placed her among the top athletes in her classification, further establishing her as a global contender.
Overcoming Challenges
One of the remarkable aspects of Olivia Breen's career is her ability to overcome personal and physical challenges. Cerebral palsy affects her daily life in numerous ways, requiring constant attention to her physical health and fitness. Despite these challenges, Olivia approaches every race and competition with a mindset of determination, refusing to let anything hold her back.
Her experience with disability discrimination has also been a source of motivation. Olivia has spoken out about the challenges she faces as a female Paralympic athlete, highlighting issues such as the lack of coverage and recognition for women in disability sports. She has become an advocate for greater visibility and equality, using her platform to encourage young athletes and raise awareness about the opportunities for those with disabilities to succeed in athletics.
The Future of Olivia Breen
As of 2025, Olivia Breen continues to be a formidable force in Paralympic athletics, aiming to achieve even greater feats in the years ahead. Her goal is to compete in the Paris 2024 Paralympic Games, where she will look to add more medals to her already impressive collection. With her commitment to the sport and her unwavering spirit, there is little doubt that Olivia will continue to inspire the world with her achievements.
Inspiring a Generation
Olivia Breen’s story is one of perseverance, hard work, and passion. She has become a role model not just for athletes with disabilities, but for anyone facing adversity in life. Her journey teaches us all that with the right mindset, determination, and support, no obstacle is insurmountable. Olivia’s continued success on the track and her advocacy for inclusivity in sports prove that athletes like her are redefining what it means to be an Olympian in the 21st century.
For Olivia Breen, the race is never over. It’s a race for equality, for recognition, and for proving to the world that no matter the circumstances, anyone can achieve greatness.
"""

# Split the long text into chunks
def split_text(text, max_length=400):
    sentences = text.split('. ')
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_length:
            chunk += sentence + '. '
        else:
            chunks.append(chunk.strip())
            chunk = sentence + '. '
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Run sentiment analysis on chunks
chunks = split_text(text)
long_results = sentiment_pipeline(chunks)
for i, (chunk, result) in enumerate(zip(chunks, long_results)):
    print(f"\nChunk {i+1} ({len(chunk)} chars): {result}")
    print(chunk)
